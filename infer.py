# infer.py

import fitz
import json
import os
import re
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from transformers import DataCollatorWithPadding

LABEL2ID = {"H1": 0, "H2": 1, "H3": 2, "junk": 3}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


def is_visual_garbage(text):
    text = text.strip()
    if len(text) < 3:
        return True
    if re.fullmatch(r"[\W_]+", text):  # only non-alphanumeric
        return True
    if re.search(r"(\.{2,}|-{2,}|={2,}|\*{2,}|~{2,}|/{2,}|\\{2,})", text):
        return True
    if text in {"•", "·", "●", "|", "_", "–", "-", "=", ">>>", "<<<"}:
        return True
    return False

def extract_title_from_first_page(pdf_path):
    doc = fitz.open(pdf_path)
    first_page = doc[0]
    blocks = first_page.get_text("dict")["blocks"]

    candidates = []
    for block in blocks:
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = clean_line_text(span["text"])
                size = span["size"]
                font = span.get("font", "").lower()

                if not text or len(text) < 4 or is_visual_garbage(text):
                    continue
                if size < 10:
                    continue
                if "bold" not in font and size < 13:
                    continue
                candidates.append((text, size))

    if not candidates:
        return "Untitled Document"

    # Pick the bold text with the largest size
    sorted_candidates = sorted(candidates, key=lambda x: x[1], reverse=True)
    return sorted_candidates[0][0].strip()



def clean_line_text(text):
    # Remove leading and trailing garbage like ...---===
    text = re.sub(r"^[\.\-_~•·→«»<>|\\✓=+*#:;,!?(){}\[\]]+", "", text)
    text = re.sub(r"[\.\-_~•·→«»<>|\\✓=+*#:;,!?(){}\[\]]+$", "", text)
    text = re.sub(r"\s{2,}", " ", text)  # collapse extra spaces
    return text.strip()


class HeadingDataset(Dataset):
    def __init__(self, texts, tokenizer):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=128)

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}


def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    grouped_lines = []

    for i, page in enumerate(doc):
        spans_by_page = []
        blocks = page.get_text("dict")["blocks"]

        for block in blocks:
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    raw_text = span["text"]
                    text = clean_line_text(raw_text)
                    size = span["size"]
                    font = span.get("font", "").lower()
                    bbox = span["bbox"]
                    y_coord = bbox[1]

                    # ─────── Heuristic Junk Filtering ───────
                    if not text or len(text) < 4:
                        continue
                    if is_visual_garbage(text):
                        continue
                    if any(p in text.lower() for p in [
                        "page", "slide", "copyright", "confidential", ".com", ".pdf", "www.",
                        "figure", "table", "all rights"
                    ]):
                        continue
                    if size < 9:
                        continue
                    if "bold" not in font and size < 12:
                        continue
                    if len(text.split()) > 25:
                        continue

                    spans_by_page.append({
                        "text": text,
                        "size": size,
                        "font": font,
                        "y": y_coord,
                        "page": i + 1
                    })

        # ─────── Sort by y position ───────
        spans_by_page.sort(key=lambda x: x["y"])

        current_group = []
        last_span = None

        for span in spans_by_page:
            if not last_span:
                current_group.append(span)
            else:
                same_font = span["font"] == last_span["font"]
                same_size = abs(span["size"] - last_span["size"]) < 0.5
                close_vertically = abs(span["y"] - last_span["y"]) < 10

                if same_font and same_size and close_vertically:
                    current_group.append(span)
                else:
                    merged_text = clean_line_text(" ".join([s["text"] for s in current_group]))
                    if len(merged_text) >= 4 and not is_visual_garbage(merged_text):
                        grouped_lines.append({
                            "text": merged_text,
                            "page": last_span["page"]
                        })
                    current_group = [span]

            last_span = span

        # Final group on page
        if current_group:
            merged_text = clean_line_text(" ".join([s["text"] for s in current_group]))
            if len(merged_text) >= 4 and not is_visual_garbage(merged_text):
                grouped_lines.append({
                    "text": merged_text,
                    "page": current_group[-1]["page"]
                })

    return grouped_lines


def predict_on_pdf(pdf_path, output_path):
    tokenizer = BertTokenizer.from_pretrained("./model")
    model = BertForSequenceClassification.from_pretrained("./model")
    model.eval()

    title = extract_title_from_first_page(pdf_path)

    extracted = extract_text_from_pdf(pdf_path)
    texts = [e["text"] for e in extracted]

    if not texts:
        print("⚠️ No text extracted. Check filters or PDF content.")
        return

    dataset = HeadingDataset(texts, tokenizer)
    dataloader = DataLoader(dataset, batch_size=16, collate_fn=DataCollatorWithPadding(tokenizer))

    all_preds = []
    with torch.no_grad():
        for batch in dataloader:
            outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.tolist())

    outline = []
    for entry, pred in zip(extracted, all_preds):
        label = ID2LABEL[pred]
        cleaned_text = entry["text"].strip()

        if label != "junk" and len(cleaned_text) >= 4:
            outline.append({
                "level": label,
                "text": cleaned_text,
                "page": entry["page"]
            })

    final_output = {
        "title": title,
        "outline": outline
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)

    print(f"✅ Saved to {output_path} (title: '{title}', {len(outline)} headings)")


if __name__ == "__main__":
    input_dir = "/app/input"
    output_dir = "/app/output"

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(input_dir, filename)
            json_name = filename.rsplit(".", 1)[0] + ".json"
            output_path = os.path.join(output_dir, json_name)

            print(f"📄 Processing {filename}...")
            predict_on_pdf(pdf_path, output_path)