#### PATCHARAKORN ####

"""Run Task 1 baseline on validation set with standard llama3.2 (supports multiple versions)."""
import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

sys.path.insert(0, str(ROOT))
from internship.ABA_mining.src.prompts import load_prompt, render_prompt

import pandas as pd
import yaml
import requests

VAL_IDS_PATH = ROOT / "finetune" / "data" / "val_ids.json"
CSV_PATH = ROOT / "Dataset" / "Original ABA Dataset for Version 2 [Oct 23, 2025], Senior Project, MUICT - Sheet2_.csv"
OUTPUT_BASE_DIR = ROOT.parent / "outputs" / "task1" / "llama3.2_baseline"

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2"

def main():
    parser = argparse.ArgumentParser(description="Run Task 1 baseline with llama3.2")
    parser.add_argument("--num-versions", type=int, default=3,
                        help="Number of versions to run (default: 3)")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print(f"Task 1 - Baseline Model Inference (llama3.2)")
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

    print("Loading prompt template...")
    header = load_prompt(ROOT, "prompts/task1/header.txt")
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
        for i, (review_id, review_text) in enumerate(reviews_dict.items()):
            if i % 10 == 0:
                print(f"  {i}/{len(reviews_dict)}...", flush=True)

            prompt = render_prompt(
                prompt_template,
                TOPICS=topics_str,
                REVIEW_TEXT=review_text,
            )

            try:
                response = requests.post(
                    OLLAMA_URL,
                    json={"model": MODEL_NAME, "prompt": prompt, "stream": False},
                    timeout=120,
                )
                response.raise_for_status()
                result_text = response.json()["response"]
            except Exception as e:
                print(f"Error calling Ollama: {e}")
                result_text = "{}"

            try:
                result = json.loads(result_text)
            except json.JSONDecodeError:
                result = {"annotations": []}

            results[review_id] = {
                "id": review_id,
                "review_text": review_text,
                "prediction": result,
            }

        output_file = OUTPUT_BASE_DIR / f"task1_predictions_{run_label}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"  → Wrote: {output_file}")
        output_paths.append(output_file)

    print("\n" + "=" * 60)
    print(f"All {args.num_versions} version(s) completed:")
    for i, path in enumerate(output_paths, 1):
        print(f"  version{i}: {path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
