#### HUGGINGFACE ####
#### https://huggingface.co/blog/ImranzamanML/fine-tuning-1b-llama-32-a-comprehensive-article ####

"""Run Task 1 with fine-tuned model on validation set (supports multiple versions)."""
import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
os.environ.setdefault("HF_HOME", str((ROOT.parent / ".hf_cache").resolve()))

sys.path.insert(0, str(ROOT))
from internship.ABA_mining.src.prompts import load_prompt, render_prompt
from internship.ABA_mining.src.utils import try_parse_json, normalize_parsed_json

import pandas as pd
import yaml
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MERGED_MODEL_PATH = ROOT / "finetune" / "output" / "task1_merged"
VAL_IDS_PATH = ROOT / "finetune" / "data" / "val_ids.json"
CSV_PATH = ROOT / "Dataset" / "Original ABA Dataset for Version 2 [Oct 23, 2025], Senior Project, MUICT - Sheet2_.csv"
OUTPUT_BASE_DIR = ROOT.parent / "outputs" / "task1" / "llama3.2_ft"

def main():
    parser = argparse.ArgumentParser(description="Run Task 1 with fine-tuned model")
    parser.add_argument("--num-versions", type=int, default=3,
                        help="Number of versions to run (default: 3)")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print(f"Task 1 - Fine-tuned Model Inference")
    print(f"Versions : {args.num_versions}")
    print("=" * 60 + "\n")

    print("Loading validation IDs...")
    with open(VAL_IDS_PATH) as f:
        val_ids = set(json.load(f))
    print(f"Loaded {len(val_ids)} validation review IDs")

    print("Loading review data from CSV...")
    df = pd.read_csv(CSV_PATH, dtype=str, keep_default_na=False)
    df = df[df["ID"].str.strip() != ""]

    # Group by ID to build review instances
    reviews_dict = {}
    for review_id, g in df.groupby("ID", sort=False):
        if review_id not in val_ids:
            continue
        pos = g["PositiveReview"].iloc[0] or ""
        neg = g["NegativeReview"].iloc[0] or ""
        review_text = f'PositiveReview — "{pos.strip()}"\nNegativeReview — "{neg.strip()}"'
        reviews_dict[review_id] = review_text

    print(f"Loaded {len(reviews_dict)} validation review instances")

    print("Loading fine-tuned model...")
    tokenizer = AutoTokenizer.from_pretrained(str(MERGED_MODEL_PATH))
    model = AutoModelForCausalLM.from_pretrained(
        str(MERGED_MODEL_PATH),
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    print("Loading prompt template...")
    header = load_prompt(ROOT, "prompts/task1/finetune_header.txt")
    footer = load_prompt(ROOT, "prompts/task1/footer.txt")
    prompt_template = header.rstrip() + "\n\n" + footer

    print("Loading topics...")
    with open(ROOT / "configs" / "topics.yaml") as f:
        topics_cfg = yaml.safe_load(f)
    topics = topics_cfg["schemas"][topics_cfg["active_schema"]]["topics"]
    topics_str = ", ".join(topics)

    output_paths = []
    OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)

    for run_num in range(1, args.num_versions + 1):
        run_label = f"run{run_num}"

        print(f"\n[{run_num}/{args.num_versions}] Running inference - {run_label}...")

        results = {}
        parse_errors = 0
        norm_errors = 0
        for i, (review_id, review_text) in enumerate(reviews_dict.items()):
            if i % 10 == 0:
                print(f"  {i}/{len(reviews_dict)}...", flush=True)

            prompt = render_prompt(
                prompt_template,
                TOPICS=topics_str,
                REVIEW_TEXT=review_text,
            )

            messages = [{"role": "user", "content": prompt}]
            inputs = tokenizer.apply_chat_template(messages, tokenize=True, return_tensors="pt")

            with torch.no_grad():
                outputs = model.generate(
                    inputs["input_ids"].to(model.device),
                    attention_mask=inputs["attention_mask"].to(model.device),
                    max_new_tokens=1024,
                    temperature=0.0,
                    top_p=0.05,
                    do_sample=False,
                )

            response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

            # Try to parse JSON with improved recovery
            ok, parsed, err = try_parse_json(response)
            if not ok:
                result = {"annotations": []}
                parse_errors += 1
            else:
                # Normalize the parsed JSON to match the expected schema
                normalized, norm_errs = normalize_parsed_json(parsed, output_schema="full", topics=topics)
                if normalized is None:
                    result = {"annotations": []}
                    norm_errors += 1
                else:
                    result = normalized

            results[review_id] = {
                "id": review_id,
                "review_text": review_text,
                "prediction": result,
            }

        output_file = OUTPUT_BASE_DIR / f"task1_predictions_{run_label}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"  → Wrote: {output_file}")
        print(f"    Parse errors: {parse_errors}, Normalization errors: {norm_errors}")
        output_paths.append(output_file)

    print("\n" + "=" * 60)
    print(f"All {args.num_versions} version(s) completed:")
    for i, path in enumerate(output_paths, 1):
        print(f"  version{i}: {path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
