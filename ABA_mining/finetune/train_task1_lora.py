#### HUGGINGFACE ####
#### https://huggingface.co/blog/ImranzamanML/fine-tuning-1b-llama-32-a-comprehensive-article ####

"""QLoRA fine-tuning of Llama-3.2-3B-Instruct on the Task 1 dataset.

Usage:
  python finetune/train_task1_lora.py                       # full training run
  python finetune/train_task1_lora.py --limit 10 --max-steps 5 --max-seq-length 512  # smoke test

OOM fallback chain (try in order if training runs out of VRAM):
  1. --max-seq-length 768
  2. --lora-r 8 --lora-target-modules q_proj,v_proj
  3. set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  4. close other GPU apps to reclaim VRAM
  5. --base-model meta-llama/Llama-3.2-1B-Instruct
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd  # noqa: F401  (import before transformers avoids a DLL load-order crash on Windows)
import torch
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]  # ABA_mining
load_dotenv(ROOT / ".env")
os.environ.setdefault("HF_HOME", str((ROOT.parent / ".hf_cache").resolve()))

from datasets import load_dataset  # noqa: E402
from peft import LoraConfig  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig  # noqa: E402
from trl import SFTConfig, SFTTrainer  # noqa: E402

DATA_DIR = ROOT / "finetune" / "data"
DEFAULT_OUTPUT_DIR = ROOT / "finetune" / "output" / "task1_lora_adapter"
CHAT_TEMPLATE_PATH = ROOT.parent / ".venv" / "Lib" / "site-packages" / "trl" / "chat_templates" / "llama3_training.jinja"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-model", default="meta-llama/Llama-3.2-3B-Instruct")
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    p.add_argument("--max-seq-length", type=int, default=1024)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument(
        "--lora-target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated list of LoRA target module names.",
    )
    p.add_argument("--num-train-epochs", type=float, default=3.0)
    p.add_argument("--limit", type=int, default=None, help="Limit number of training examples (smoke test).")
    p.add_argument("--max-steps", type=int, default=-1, help="Override max optimizer steps (smoke test).")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    train_ds = load_dataset("json", data_files=str(DATA_DIR / "task1_train.jsonl"), split="train")
    val_ds = load_dataset("json", data_files=str(DATA_DIR / "task1_val.jsonl"), split="train")
    if args.limit is not None:
        train_ds = train_ds.select(range(min(args.limit, len(train_ds))))
        val_ds = val_ds.select(range(min(args.limit, len(val_ds))))

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, token=os.environ.get("HF_TOKEN"))

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        token=os.environ.get("HF_TOKEN"),
    )
    model.config.use_cache = False

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_target_modules.split(","),
        bias="none",
        task_type="CAUSAL_LM",
    )

    sft_config = SFTConfig(
        output_dir=args.output_dir,
        max_length=args.max_seq_length,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        bf16=True,
        max_grad_norm=0.3,
        eval_strategy="steps",
        eval_steps=50,
        save_steps=50,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        assistant_only_loss=True,
        chat_template_path=str(CHAT_TEMPLATE_PATH),
        packing=False,
        logging_steps=10,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
