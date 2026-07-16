import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

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
MODEL = os.environ.get("WHISPER_MODEL", "openai/whisper-small")
LANG = "hu"
HF_REPO = os.environ.get("HF_REPO", "SubZtep/whisper-small-hu")

# --- quick test mode: small subset, fast steps ---
QUICK_TEST = os.environ.get("QUICK_TEST", "0") == "1"
QUICK_TRAIN_SAMPLES = int(os.environ.get("QUICK_TRAIN_SAMPLES", "200"))
QUICK_EVAL_SAMPLES = int(os.environ.get("QUICK_EVAL_SAMPLES", "50"))

# --- push to HF Hub if a token is present ---
PUSH_TO_HUB = os.environ.get("HF_TOKEN") is not None

# --- persistent output: mount Drive on Colab, else local (loud now, not silent) ---
try:
    from google.colab import drive
    drive.mount("/content/drive")
    OUT = f"/content/drive/MyDrive/whisper-hu-{MODEL.split('/')[-1]}"
    print(f"[storage] Drive mounted -> {OUT}")
except Exception as e:
    OUT = f"./whisper-hu-{MODEL.split('/')[-1]}"
    print(f"[storage] Drive mount FAILED ({e}) -> using local: {OUT}")
    print("[storage] WARNING: local Colab disk is wiped on runtime reset!")

if QUICK_TEST:
    OUT += "-quicktest"

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

    if PUSH_TO_HUB:
        from huggingface_hub import login
        login(os.environ["HF_TOKEN"])
        print(f"[hub] checkpoints will push to -> {HF_REPO}")
    else:
        print("[hub] no HF_TOKEN set -> skipping Hub push, Drive/local only")

    processor = WhisperProcessor.from_pretrained(MODEL)
    processor.tokenizer.pad_token = processor.tokenizer.eos_token

    model = WhisperForConditionalGeneration.from_pretrained(MODEL)
    model.config.use_cache = False
    model.generation_config.language = "hungarian"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None

    train_ds = CommonVoiceDataset(DATA, "train")
    eval_ds = CommonVoiceDataset(DATA, "dev")

    if QUICK_TEST:
        from torch.utils.data import Subset
        train_ds = Subset(train_ds, range(min(QUICK_TRAIN_SAMPLES, len(train_ds))))
        eval_ds = Subset(eval_ds, range(min(QUICK_EVAL_SAMPLES, len(eval_ds))))
        print(f"[quicktest] train={len(train_ds)} eval={len(eval_ds)} samples")

    args = Seq2SeqTrainingArguments(
        output_dir=OUT,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=4,
        generation_max_length=225,
        generation_num_beams=1,
        gradient_accumulation_steps=2,  # keeps effective batch size 16
        gradient_checkpointing=True,
        warmup_steps=2 if QUICK_TEST else 500,
        lr_scheduler_type="linear",
        learning_rate=1e-5,
        num_train_epochs=1 if QUICK_TEST else 3,
        eval_strategy="steps",
        eval_steps=5 if QUICK_TEST else 1000,
        eval_accumulation_steps=10,
        logging_strategy="steps",
        logging_steps=2 if QUICK_TEST else 200,
        save_strategy="steps",
        save_steps=5 if QUICK_TEST else 1000,
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
        push_to_hub=PUSH_TO_HUB,
        hub_model_id=HF_REPO if PUSH_TO_HUB else None,
        hub_strategy="checkpoint" if PUSH_TO_HUB else "every_save",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=WhisperCollator(processor),
        compute_metrics=compute_metrics_factory(processor),
    )

    # resume automatically if a checkpoint already exists in OUT
    resume = os.path.isdir(OUT) and any(
        d.startswith("checkpoint-") for d in os.listdir(OUT)
    )
    if resume:
        print(f"[resume] found checkpoint in {OUT}, resuming")

    trainer.train(resume_from_checkpoint=resume)

    processor.save_pretrained(OUT_FINAL)
    model.save_pretrained(OUT_FINAL)
    print(f"[saved] final model -> {OUT_FINAL}")

    if PUSH_TO_HUB:
        trainer.push_to_hub()
        print(f"[hub] pushed final model -> https://huggingface.co/{HF_REPO}")


if __name__ == "__main__":
    main()
