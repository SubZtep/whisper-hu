import time
import numpy as np

class WhisperCollator:
    def __init__(self, processor):
        self.processor = processor
        self.sot_id = processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")

    def __call__(self, batch):
        audio = []
        texts = []

        for x in batch:
            a = np.asarray(x["audio"], dtype=np.float32)

            # per-sample mono safety
            if a.ndim == 2:
                a = a.mean(axis=1)

            audio.append(a)
            texts.append(x["sentence"])

        features = self.processor.feature_extractor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",
        )

        enc = self.processor.tokenizer(
            texts,
            padding=True,
            return_tensors="pt",
        )

        # pad and eos share an id in Whisper: mask via attention mask so the
        # real eos at the end of each sentence stays in the labels
        labels = enc.input_ids.masked_fill(enc.attention_mask.eq(0), -100)

        # the trainer prepends the decoder start token when shifting, and the
        # tokenizer already emits it -> drop it so it doesn't appear twice
        if (labels[:, 0] == self.sot_id).all():
            labels = labels[:, 1:]

        return {
            "input_features": features.input_features,
            "labels": labels,
        }
