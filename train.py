import os
import torch
import logging
from dataset import CommonVoiceDataset
from collator import WhisperCollator
from transformers import WhisperProcessor, WhisperForConditionalGeneration

DATA = "cv-corpus-26.0-2026-06-12/hu"
MODEL = "openai/whisper-small"
LANG = "hu"

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

def main():
    from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

    processor = WhisperProcessor.from_pretrained(MODEL)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL)
    
    model.config.forced_decoder_ids = (
        processor.get_decoder_prompt_ids(
            language=LANG,
            task="transcribe",
        )
    )

    train_ds = CommonVoiceDataset(DATA, "train")
    eval_ds = CommonVoiceDataset(DATA, "dev")

    args = Seq2SeqTrainingArguments(
        output_dir="./whisper-hu",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=2,
        learning_rate=1e-5,
        num_train_epochs=3,
        eval_strategy="steps",
        eval_steps=200,
        logging_strategy="steps",
        logging_steps=5,
        save_strategy="steps",
        save_steps=200,
        fp16=False,
        max_grad_norm=1.0,
        dataloader_num_workers=2,
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
    )

    trainer.train()

    processor.save_pretrained("./whisper-hu")
    model.save_pretrained("./whisper-hu")

if __name__ == "__main__":
    main()
