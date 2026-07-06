import time
import numpy as np

class WhisperCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, batch):
        audio = []
        texts = []

        for x in batch:
            a = np.asarray(x["audio"], dtype=np.float32)
            a = a / (np.max(np.abs(a)) + 1e-9) # normalize

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

        labels = self.processor.tokenizer(
            texts,
            padding=True,
            return_tensors="pt",
        ).input_ids

        # IMPORTANT for Whisper loss stability
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {
            "input_features": features.input_features,
            "labels": labels,
        }
