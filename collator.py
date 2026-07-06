import time
import torch
import logging
import numpy as np

logger = logging.getLogger(__name__)

class WhisperCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, batch):
        t0 = time.perf_counter()
        logger.info("start batch")

        audio = [np.asarray(x["audio"], dtype=np.float32) for x in batch]

        # per-sample normalization (important)
        audio = [
            a / (np.max(np.abs(a)) + 1e-9)
            for a in audio
        ]

        features = self.processor.feature_extractor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",
        )

        labels = self.processor.tokenizer(
            [x["sentence"] for x in batch],
            padding=True,
            return_tensors="pt",
        ).input_ids

        return {
            "input_features": features.input_features,
            "labels": labels,
        }
