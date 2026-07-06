import os
import torch
import librosa
import logging
import functools
import pandas as pd
import soundfile as sf
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)
logging.getLogger("numba").setLevel(logging.WARNING)

@functools.lru_cache(maxsize=2000)
def load_audio(path):
    return sf.read(path, dtype="float32")

class CommonVoiceDataset(Dataset):
    def __init__(self, data_path: str, split: str):
        self.data_path = data_path

        df = pd.read_csv(
            os.path.join(data_path, f"{split}.tsv"),
            sep="\t",
            low_memory=False,
        )

        df = df.dropna(subset=["path", "sentence"])

        self.rows = list(
            zip(
                df["path"].tolist(),
                df["sentence"].tolist(),
            )
        )
        self.rows = self.rows[:200]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        if idx % 1000 == 0:
            logger.info(f"Loading sample {idx}")

        path, sentence = self.rows[idx]

        audio_path = os.path.join(
            self.data_path,
            "clips",
            path,
        )

        audio, sr = load_audio(audio_path)
        
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            sr = 16000

        if audio.ndim == 2:
            audio = audio.mean(axis=1)

        return {
            "audio": audio,
            "sampling_rate": sr,
            "sentence": sentence,
        }
 