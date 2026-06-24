#### PATCHARAKORN ####

"""Merge LoRA adapter with base model and save merged model."""
import os
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
os.environ.setdefault("HF_HOME", str((ROOT.parent / ".hf_cache").resolve()))

import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
ADAPTER_PATH = ROOT / "finetune" / "output" / "task1_lora_adapter"
OUTPUT_PATH = ROOT / "finetune" / "output" / "task1_merged"

print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="cpu",
    token=os.environ.get("HF_TOKEN"),
)

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(model, str(ADAPTER_PATH))

print("Merging adapter into base model...")
merged_model = model.merge_and_unload()

OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
print(f"Saving merged model to {OUTPUT_PATH}...")
merged_model.save_pretrained(str(OUTPUT_PATH), safe_serialization=True)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=os.environ.get("HF_TOKEN"))
tokenizer.save_pretrained(str(OUTPUT_PATH))

print("Done!")
