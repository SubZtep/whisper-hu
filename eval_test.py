import os
import re
import torch
import evaluate
from torch.utils.data import DataLoader, Subset

from dataset import CommonVoiceDataset
from collator import WhisperCollator
from transformers import WhisperProcessor, WhisperForConditionalGeneration

DATA = "data/cv-corpus-26.0-2026-06-12/hu"
MODEL = os.environ.get("EVAL_MODEL", "SubZtep/whisper-small-hu")
BATCH = int(os.environ.get("EVAL_BATCH", "16"))

# optional cap for smoke testing, e.g. EVAL_SAMPLES=8 python eval_test.py
EVAL_SAMPLES = int(os.environ.get("EVAL_SAMPLES", "0"))


def normalize(texts):
    def clean(t):
        t = t.lower().strip()
        t = re.sub(r"\s+", " ", t)
        return t
    return [clean(t) for t in texts]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = WhisperProcessor.from_pretrained(MODEL, language="hu", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(
        MODEL,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)
    model.eval()

    ds = CommonVoiceDataset(DATA, "test")
    if EVAL_SAMPLES and len(ds) > EVAL_SAMPLES:
        ds = Subset(ds, range(EVAL_SAMPLES))
    print(f"[eval] {MODEL} on {len(ds)} test samples ({device})")

    loader = DataLoader(
        ds,
        batch_size=BATCH,
        collate_fn=WhisperCollator(processor),
        num_workers=2,
    )

    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")
    preds, refs = [], []

    for i, batch in enumerate(loader):
        features = batch["input_features"].to(device, dtype=model.dtype)
        with torch.no_grad():
            ids = model.generate(features, language="hu", task="transcribe", max_length=225)

        labels = batch["labels"].masked_fill(
            batch["labels"].eq(-100), processor.tokenizer.pad_token_id
        )
        preds += normalize(processor.batch_decode(ids, skip_special_tokens=True))
        refs += normalize(processor.batch_decode(labels, skip_special_tokens=True))

        if (i + 1) % 20 == 0:
            wer = wer_metric.compute(predictions=preds, references=refs) * 100
            print(f"[eval] {len(preds)}/{len(ds)} running WER {wer:.2f}", flush=True)

    wer = wer_metric.compute(predictions=preds, references=refs) * 100
    cer = cer_metric.compute(predictions=preds, references=refs) * 100
    print(f"\n[result] test WER {wer:.2f} | CER {cer:.2f} ({len(preds)} samples)")


if __name__ == "__main__":
    main()
