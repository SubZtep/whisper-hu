---
language: hu
license: mit
library_name: transformers
pipeline_tag: automatic-speech-recognition
base_model: openai/whisper-small
tags:
- whisper
- hungarian
- common-voice
metrics:
- wer
- cer
model-index:
- name: whisper-small-hu
  results:
  - task:
      type: automatic-speech-recognition
      name: Automatic Speech Recognition
    dataset:
      name: Common Voice 26.0 (Hungarian)
      type: mozilla-foundation/common_voice_26_0
      config: hu
      split: test
    metrics:
    - type: wer
      value: 15.62
      name: WER
    - type: cer
      value: 3.33
      name: CER
---

# whisper-small-hu

[openai/whisper-small](https://huggingface.co/openai/whisper-small) fine-tuned for Hungarian
speech recognition on [Mozilla Common Voice 26.0](https://commonvoice.mozilla.org/).

| | WER | CER |
|---|---|---|
| openai/whisper-small (zero-shot) | 46.79 | 11.70 |
| **whisper-small-hu** | **15.62** | **3.33** |

Both models evaluated on the full Common Voice 26.0 Hungarian test split (13,127 clips) with
the same pipeline: greedy decoding, forced `<|hu|><|transcribe|>` prompt, predictions and
references lowercased and whitespace-normalized before scoring.

## Usage

```python
from transformers import pipeline

asr = pipeline("automatic-speech-recognition", model="SubZtep/whisper-small-hu")
print(asr("audio.mp3", generate_kwargs={"language": "hungarian", "task": "transcribe"})["text"])
```

## Training

- **Data:** Common Voice 26.0 Hungarian `train` split (62,284 clips), silence-trimmed, 16 kHz
- **Schedule:** 3 epochs (~11,700 steps), effective batch size 16 (8 × 2 gradient accumulation)
- **Optimizer:** AdamW, lr 1e-5, linear decay with 500 warmup steps, fp16, gradient checkpointing
- **Hardware:** single NVIDIA T4 (Google Colab)
- **Selection:** best checkpoint by WER on a fixed 1,500-sample subset of the `dev` split

Training code: https://github.com/SubZtep/whisper-hu

### Training progress

WER on the 1,500-sample `dev` subset during training:

| Epoch | Step  | Train Loss | Val Loss | WER     |
|:-----:|:-----:|:----------:|:--------:|:-------:|
| 0.26  | 1000  | 0.3302     | 0.3284   | 30.54   |
| 0.51  | 2000  | 0.2490     | 0.2674   | 25.36   |
| 0.77  | 3000  | 0.2130     | 0.2337   | 23.12   |
| 1.03  | 4000  | 0.1380     | 0.2017   | 18.60   |
| 1.28  | 5000  | 0.1011     | 0.1960   | 18.16   |
| 1.54  | 6000  | 0.0923     | 0.1866   | 17.29   |
| 1.80  | 7000  | 0.0913     | 0.1779   | 16.28   |
| 2.06  | 8000  | 0.0390     | 0.1765   | 15.71   |
| 2.31  | 9000  | 0.0388     | 0.1781   | 15.32   |
| 2.57  | 10000 | 0.0394     | 0.1762   | 15.38   |
| 2.83  | 11000 | 0.0369     | 0.1745   | 15.31   |

### Framework versions

- Transformers 4.44.2
- PyTorch 2.11.0+cu128
- Datasets 4.0.0
- Tokenizers 0.19.1

## Limitations

- Trained on read speech (Common Voice sentences) — expect higher error rates on spontaneous
  conversation, noisy audio, or domain-specific vocabulary
- WER is measured on lowercased text; the model itself outputs casing and punctuation, but
  their accuracy is not reflected in the score
- Hungarian is agglutinative, so WER punishes single-suffix mistakes as whole-word errors —
  the CER of 3.33 is a better sense of how close transcripts are
