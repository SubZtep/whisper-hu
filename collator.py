import time
import torch
import numpy as np

class WhisperCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, batch):
        t0 = time.perf_counter()
        print("start batch")

        audio = [x["audio"] for x in batch]

        features = self.processor.feature_extractor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",
        )

        labels = self.processor.tokenizer(
            [x["sentence"] for x in batch],
            padding=True,
            return_tensors="pt",
        )

        labels = labels.input_ids.masked_fill(
            labels.input_ids == self.processor.tokenizer.pad_token_id,
            -100,
        )

        t1 = time.perf_counter()
        print(f"Collator: {(t1-t0):.3f}s")

        return {
            "input_features": features.input_features,
            "labels": labels,
        }
