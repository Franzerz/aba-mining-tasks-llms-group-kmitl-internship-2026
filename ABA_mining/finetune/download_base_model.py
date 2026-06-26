#### PATCHARAKORN ####

"""Download the base model for fine-tuning into the local HF cache."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

os.environ.setdefault("HF_HOME", str((ROOT.parent / ".hf_cache").resolve()))

from huggingface_hub import snapshot_download

MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"

if __name__ == "__main__":
    path = snapshot_download(
        repo_id=MODEL_ID,
        token=os.environ["HF_TOKEN"],
    )
    print("Downloaded to:", path)
