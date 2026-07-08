import os
import re
import torch
import logging
import warnings
import evaluate
import numpy as np
from dataset import CommonVoiceDataset
from collator import WhisperCollator
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers.utils import logging as hf_logging

DATA = "data/cv-corpus-26.0-2026-06-12/hu"
MODEL = "openai/whisper-small"
LANG = "hu"

# --- persistent output: mount Drive on Colab, else local ---
try:
    from google.colab import drive
    drive.mount("/content/drive")
    OUT = "/content/drive/MyDrive/whisper-hu"
except Exception:
    OUT = "./whisper-hu"

OUT_FINAL = OUT + "-final"

torch.backends.cuda.matmul.allow_tf32 = True

# --- quiet the DOM-killing spam ---
warnings.filterwarnings("ignore")
hf_logging.set_verbosity_error()
try:
    import datasets
    datasets.utils.logging.set_verbosity_error()
except Exception:
    pass

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
wer_metric = evaluate.load("wer")

def normalize(texts):
    def clean(t):
        t = t.lower().strip()
        t = re.sub(r"\s+", " ", t)
        return t
    return [clean(t) for t in texts]

def compute_metrics_factory(processor):
    def compute_metrics(pred):
        pred_ids = pred.predictions

        label_ids = pred.label_ids.copy()
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = normalize(
            processor.batch_decode(pred_ids, skip_special_tokens=True)
        )

        label_str = normalize(
            processor.batch_decode(label_ids, skip_special_tokens=True)
        )

        return {
            "wer": wer_metric.compute(
                predictions=pred_str,
                references=label_str,
            ) * 100
        }

    return compute_metrics

def main():
    from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

    processor = WhisperProcessor.from_pretrained(MODEL)
    processor.tokenizer.pad_token = processor.tokenizer.eos_token

    model = WhisperForConditionalGeneration.from_pretrained(MODEL)
    model.config.use_cache = False
    model.generation_config.language = "hungarian"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None

    train_ds = CommonVoiceDataset(DATA, "train")
    eval_ds = CommonVoiceDataset(DATA, "dev")

    args = Seq2SeqTrainingArguments(
        output_dir=OUT,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=1,
        generation_max_length=225,
        generation_num_beams=1,
        gradient_accumulation_steps=1,
        gradient_checkpointing=False,
        warmup_steps=500,
        lr_scheduler_type="linear",
        learning_rate=5e-6,
        num_train_epochs=3,
        eval_strategy="steps",
        eval_steps=1000,
        eval_accumulation_steps=10,
        logging_strategy="steps",
        logging_steps=200,
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        predict_with_generate=True,
        fp16=True,
        bf16=False,
        max_grad_norm=1.0,
        dataloader_num_workers=4,
        disable_tqdm=True,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=WhisperCollator(processor),
        compute_metrics=compute_metrics_factory(processor),
    )

    trainer.train()

    processor.save_pretrained(OUT_FINAL)
    model.save_pretrained(OUT_FINAL)
    print(f"[saved] final model -> {OUT_FINAL}")

if __name__ == "__main__":
    main()
