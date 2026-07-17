## Run on Google Colab

🔗 [My Interactive Python Notebook](https://colab.research.google.com/drive/1YvY3lClsRgdtOR5prBtf_xTcYZ21Vn-a?usp=sharing)
([CT2 conversion](https://colab.research.google.com/drive/1OF4xFBVoJZUn1Az232zNOR6P-QOLbxW-?usp=sharing))

## Run locally

> [!IMPORTANT]  
> Python 3.11 required.

1. Create a `.env` file based on [`.env.example`](.env.example), and populate it with your [API key](https://mozilladatacollective.com/profile/credentials).

2. Create a virtual environment:
   ```sh
   python -m venv .venv
   ```

2. Activate it:
   ```sh
   source .venv/bin/activate
   ```

3. Install dependencies
   ```sh
   pip install -r requirements.txt
   ```

4. Install CUDA-specific PyTorch **(optional)**:
   ```sh
   pip install torch==2.2.2+cu118 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

5. Download dataset:
   ```sh
   python download_dataset.py
   ```

6. Run the project:
   ```sh
   python train.py
   ```

## Training a different model size

`train.py` is parameterized via env vars, so the same script trains any Whisper
checkpoint and pushes to a matching HF repo — no code changes needed.

| Variable        | Default                     | Purpose                                  |
| ---------------- | ---------------------------- | ----------------------------------------- |
| `WHISPER_MODEL`  | `openai/whisper-small`       | Base model to fine-tune                   |
| `HF_REPO`        | `SubZtep/whisper-small-hu`   | HF Hub repo to push checkpoints/model to  |
| `HF_TOKEN`       | —                             | HF write token; if unset, Hub push is skipped |
| `QUICK_TEST`     | `0`                           | `1` = small subset, fast smoke-test run   |

Example:

```bash
# default (small)
python train.py

# large-v3, separate HF repo
WHISPER_MODEL=openai/whisper-large-v3 HF_REPO=SubZtep/whisper-large-v3-hu python train.py

# quick smoke test before a full run
QUICK_TEST=1 python train.py
```

Each model size gets its own HF Hub repo (e.g. `whisper-small-hu`,
`whisper-large-v3-hu`) since model cards, weights, and repo size differ per
checkpoint. Local/Drive output dirs are also namespaced per model
(`whisper-hu-<model-name>`), so runs don't overwrite each other.

---

[Download Dataset](https://mozilladatacollective.com/datasets/cmqinob6900wknr07s6fgcprx)
