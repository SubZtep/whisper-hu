import os
import re
import torch
import logging
import evaluate
import numpy as np
from dataset import CommonVoiceDataset
from collator import WhisperCollator
from transformers import WhisperProcessor, WhisperForConditionalGeneration

DATA_DIR_RELATIVE = "data"
MODEL = "openai/whisper-small"
LANG = "hu"

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

        pred_str = normalize(processor.batch_decode(pred_ids, skip_special_tokens=True))
        label_str = normalize(processor.batch_decode(label_ids, skip_special_tokens=True))

        wer = wer_metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer * 100}
    return compute_metrics

def main():
    from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

    processor = WhisperProcessor.from_pretrained(MODEL)
    processor.tokenizer.pad_token = processor.tokenizer.eos_token

    model = WhisperForConditionalGeneration.from_pretrained(MODEL)
    model.config.use_cache = False
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=LANG,
        task="transcribe",
    )

    train_ds = CommonVoiceDataset(DATA_DIR_RELATIVE, "train")
    eval_ds = CommonVoiceDataset(DATA_DIR_RELATIVE, "dev")

    args = Seq2SeqTrainingArguments(
        output_dir="./whisper-hu",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=1,
        generation_max_length=225,
        generation_num_beams=1,
        gradient_accumulation_steps=2,
        gradient_checkpointing=False,
        warmup_steps=500,
        lr_scheduler_type="linear",
        learning_rate=5e-6,
        num_train_epochs=3,
        eval_strategy="steps",
        eval_steps=1000,
        logging_strategy="steps",
        logging_steps=5,
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=2,
        predict_with_generate=True,
        fp16=True,
        bf16=False,
        max_grad_norm=1.0,
        dataloader_num_workers=4,
        disable_tqdm=False,
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

    processor.save_pretrained("./whisper-hu")
    model.save_pretrained("./whisper-hu")

if __name__ == "__main__":
    main()
