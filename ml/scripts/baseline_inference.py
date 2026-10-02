"""
Baseline inference for VakyaSetu.

Runs the UNMODIFIED IndicTrans2 distilled Indic->English model on Sanskrit
sentences and saves the results to a CSV.

Usage:
    python baseline_inference.py sentences.txt baseline_outputs.csv

sentences.txt: one Sanskrit sentence per line (UTF-8, Devanagari).

Adapted from the IndicTransToolkit README / IndicTrans2 huggingface_interface
example.py. If a call signature differs in the version you install, treat the
repo's own example.py as the source of truth and adjust this file to match.
"""

import csv
import sys

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit import IndicProcessor

CKPT = "ai4bharat/indictrans2-indic-en-dist-200M"  # distilled 200M, Indic -> English
SRC_LANG = "san_Deva"   # Sanskrit, Devanagari script
TGT_LANG = "eng_Latn"   # English, Latin script
BATCH_SIZE = 4
MAX_LEN = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_sentences(path):
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def main(in_path, out_path):
    sentences = load_sentences(in_path)
    print(f"Loaded {len(sentences)} sentences. Device: {DEVICE}")

    ip = IndicProcessor(inference=True)
    tokenizer = AutoTokenizer.from_pretrained(CKPT, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(CKPT, trust_remote_code=True).to(DEVICE)
    model.eval()

    rows = []
    for i in range(0, len(sentences), BATCH_SIZE):
        chunk = sentences[i : i + BATCH_SIZE]

        # 1. IndicProcessor: normalisation + adds the language tags the model expects
        batch = ip.preprocess_batch(chunk, src_lang=SRC_LANG, tgt_lang=TGT_LANG)

        # 2. Tokenise into tensors
        inputs = tokenizer(
            batch,
            padding="longest",
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt",
        ).to(DEVICE)

        # 3. Generate with beam search
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_length=MAX_LEN,
                num_beams=5,
                num_return_sequences=1,
            )

        # 4. Decode token ids back to text, then post-process
        decoded = tokenizer.batch_decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        translations = ip.postprocess_batch(decoded, lang=TGT_LANG)

        rows.extend(zip(chunk, translations))
        print(f"  translated {min(i + BATCH_SIZE, len(sentences))}/{len(sentences)}")

    # utf-8-sig so Excel displays Devanagari correctly when you double-click the CSV
    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sanskrit", "model_output", "reference_english", "notes"])
        for src, hyp in rows:
            writer.writerow([src, hyp, "", ""])  # fill reference + notes by hand

    print(f"Saved {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python baseline_inference.py sentences.txt baseline_outputs.csv")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])