#### HUGGINGFACE ####

import sys
print(f"Python: {sys.executable}")
print(f"Path: {sys.path[:3]}")

from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
print(f"ROOT: {ROOT}")

import os
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

print("About to import peft...")
from peft import PeftModel
print("peft imported successfully!")
