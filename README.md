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

---

[Download Dataset](https://mozilladatacollective.com/datasets/cmqinob6900wknr07s6fgcprx)
