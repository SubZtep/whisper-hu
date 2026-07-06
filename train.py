import os
import torch
from dataset import CommonVoiceDataset
from collator import WhisperCollator
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# from transformers import (
#     WhisperProcessor,
#     WhisperForConditionalGeneration,
#     # Seq2SeqTrainer,
#     # Seq2SeqTrainingArguments,
# )

# def main():
#     torch.cuda.init()
#     os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
#     print("1. script start")

# if __name__ == "__main__":
#     main()

MODEL = "openai/whisper-small"
LANG = "hu"
# torch.multiprocessing.set_start_method("spawn", force=True)

def main():
    from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
    # from transformers import (
    #     WhisperProcessor,
    #     WhisperForConditionalGeneration,
    #     Seq2SeqTrainer,
    #     Seq2SeqTrainingArguments,
    # ) 
    
    print("1. script start")
    processor = WhisperProcessor.from_pretrained(MODEL)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL)

    # print("before cuda")
    # model.to("cuda")
    # print("after cuda")
            
    print("2. run")
    model.config.use_cache = False
    # model.gradient_checkpointing_enable()
    
    model.config.forced_decoder_ids = (
        processor.get_decoder_prompt_ids(
            language=LANG,
            task="transcribe",
        )
    )
    
    train_ds = CommonVoiceDataset(
        "cv-corpus-26.0-2026-06-12/hu",
        "train",
    )
    
    eval_ds = CommonVoiceDataset(
        "cv-corpus-26.0-2026-06-12/hu",
        "dev",
    )
    
    print('3. lol')
    
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
        fp16=torch.cuda.is_available(),
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
