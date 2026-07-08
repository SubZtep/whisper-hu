from datacollective import download_dataset
from pathlib import Path
import tarfile

# DATASET_ID = "cmqinob6900wknr07s6fgcprx"
# Dataset slug also accepted
DATASET_ID = "common-voice-scripted-speech-26-0-hungar-22d743b4"

DATA_DIR_RELATIVE = "data"
ROOT = Path(__file__).parent
DATA_DIR = ROOT / DATA_DIR_RELATIVE

print(f"Using data directory: {DATA_DIR}")

DATA_DIR.mkdir(exist_ok=True)

if any(DATA_DIR.iterdir()):
    raise RuntimeError(
        f"{DATA_DIR} is not empty. "
        "Please remove existing files before downloading the dataset."
    )

print("Downloading dataset...")

download_dataset(
    DATASET_ID,
    download_directory=DATA_DIR_RELATIVE,
)

archives = list(DATA_DIR.glob("*.tar.gz"))

if len(archives) != 1:
    raise RuntimeError(
        f"Expected exactly one tar.gz archive, found {len(archives)}"
    )

archive = archives[0]

print(f"Extracting {archive.name}...")

with tarfile.open(archive, "r:gz") as tar:
    tar.extractall(path=DATA_DIR, filter="fully_trusted")

print("Removing archive...")
archive.unlink()

print("Dataset ready.")
