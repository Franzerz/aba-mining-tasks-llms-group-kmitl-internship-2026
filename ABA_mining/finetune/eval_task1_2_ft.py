#### PATCHARAKORN ####

"""Evaluate fine-tuned model on Task 1.2 (span extraction) vs baseline.

Compares fine-tuned Llama-3.2-3B-Instruct with baseline on validation set.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")
os.environ.setdefault("HF_HOME", str((ROOT.parent / ".hf_cache").resolve()))

from internship.ABA_mining.src import load_topics_config, build_client
from internship.ABA_mining.src.prompts import build_modular_prompt
from internship.ABA_mining.src.task1 import load_task1_instances_from_input, build_task1_schema, local_validate_task1

MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
FT_MODEL_DIR = ROOT / "finetune" / "output" / "task1_merged"
VAL_IDS_FILE = ROOT / "finetune" / "data" / "val_ids.json"


def load_validation_ids() -> list[str]:
    """Load validation IDs from fine-tuning data split."""
    with open(VAL_IDS_FILE, "r") as f:
        return json.load(f)


def run_ft_model(review_texts: dict[str, str], val_ids: list[str], num_runs: int = 1) -> dict:
    """Run fine-tuned model on validation set."""
    print("\n" + "=" * 70)
    print("FINE-TUNED MODEL EVALUATION (Task 1.2 - Span Extraction)")
    print("=" * 70)

    # Load fine-tuned model
    print("\nLoading fine-tuned model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        str(FT_MODEL_DIR),
        torch_dtype=torch.bfloat16,
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(str(FT_MODEL_DIR))
    model.eval()
    print(f"  Device: {device}")
    print(f"  Model loaded from: {FT_MODEL_DIR}")

    # Load config
    topics_cfg = load_topics_config(ROOT)
    topics = topics_cfg.topics

    # Build Task 1.2 prompt (span only)
    rule_list = [1, 7]  # Rules for span extraction only
    output_schema = "span_only"
    prompt_template = build_modular_prompt(ROOT, rule_list, output_schema)

    results = {"correct": 0, "total": 0, "errors": []}

    # Run on validation set
    print(f"\nRunning on {len(val_ids)} validation IDs...")
    for review_id in val_ids:
        review_id_str = str(review_id)
        if review_id_str not in review_texts:
            continue

        review_text = review_texts[review_id_str]

        # Prepare prompt
        from internship.ABA_mining.src.prompts import render_prompt

        topics_str = ", ".join(topics)
        prompt = render_prompt(
            prompt_template,
            TOPICS=topics_str,
            REVIEW_TEXT=review_text,
        )

        # Generate
        try:
            messages = [{"role": "user", "content": prompt}]
            rendered = tokenizer.apply_chat_template(messages, tokenize=False)
            inputs = tokenizer(rendered, return_tensors="pt").to(device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.0,
                    top_p=0.05,
                    do_sample=False,
                )

            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Extract JSON from response
            if "{" in response:
                json_str = response[response.rfind("{") :]
                result = json.loads(json_str)

                # Validate span
                errors = local_validate_task1(
                    result, topics=topics, review_text=review_text, output_schema=output_schema
                )
                if not errors:
                    results["correct"] += 1
                results["total"] += 1

        except Exception as e:
            results["errors"].append({"id": review_id_str, "error": str(e)})

    # Calculate metrics
    accuracy = (results["correct"] / results["total"] * 100) if results["total"] > 0 else 0
    print(f"\nFine-Tuned Model Results:")
    print(f"  Valid spans: {results['correct']} / {results['total']}")
    print(f"  Span accuracy: {accuracy:.1f}%")
    print(f"  Errors: {len(results['errors'])}")

    return {
        "model": "fine-tuned",
        "correct": results["correct"],
        "total": results["total"],
        "accuracy": accuracy,
    }


def run_baseline_model(
    review_texts: dict[str, str], val_ids: list[str]
) -> dict:
    """Run baseline model on validation set using Ollama."""
    print("\n" + "=" * 70)
    print("BASELINE MODEL EVALUATION (Task 1.2 - Span Extraction)")
    print("=" * 70)

    client = build_client("ollama")

    # Load config
    topics_cfg = load_topics_config(ROOT)
    topics = topics_cfg.topics

    # Build Task 1.2 prompt (span only)
    rule_list = [1, 7]  # Rules for span extraction only
    output_schema = "span_only"
    prompt_template = build_modular_prompt(ROOT, rule_list, output_schema)

    results = {"correct": 0, "total": 0, "errors": []}

    print(f"\nRunning on {len(val_ids)} validation IDs with llama3.2...")
    for i, review_id in enumerate(val_ids):
        review_id_str = str(review_id)
        if review_id_str not in review_texts:
            continue

        review_text = review_texts[review_id_str]

        # Prepare prompt
        from internship.ABA_mining.src.prompts import render_prompt

        topics_str = ", ".join(topics)
        prompt = render_prompt(
            prompt_template,
            TOPICS=topics_str,
            REVIEW_TEXT=review_text,
        )

        # Generate via Ollama
        try:
            response = client.generate(
                model="llama3.2",
                prompt=prompt,
                stream=False,
                options={
                    "temperature": 0.0,
                    "top_p": 0.05,
                },
            )

            result_text = response.get("response", "")
            if "{" in result_text:
                json_str = result_text[result_text.rfind("{") :]
                result = json.loads(json_str)

                # Validate span
                errors = local_validate_task1(
                    result, topics=topics, review_text=review_text, output_schema=output_schema
                )
                if not errors:
                    results["correct"] += 1
                results["total"] += 1

            if (i + 1) % 20 == 0:
                print(f"  Processed {i + 1}/{len(val_ids)}")

        except Exception as e:
            results["errors"].append({"id": review_id_str, "error": str(e)})

    # Calculate metrics
    accuracy = (results["correct"] / results["total"] * 100) if results["total"] > 0 else 0
    print(f"\nBaseline Model Results:")
    print(f"  Valid spans: {results['correct']} / {results['total']}")
    print(f"  Span accuracy: {accuracy:.1f}%")
    print(f"  Errors: {len(results['errors'])}")

    return {
        "model": "baseline",
        "correct": results["correct"],
        "total": results["total"],
        "accuracy": accuracy,
    }


def main():
    # Load validation IDs and review texts
    val_ids = load_validation_ids()
    print(f"Loaded {len(val_ids)} validation IDs")

    # Load review texts
    from internship.ABA_mining.src import load_paths_config
    paths_cfg = load_paths_config(ROOT)
    instances = load_task1_instances_from_input(paths_cfg, limit_reviews=None)
    review_texts = {inst.review_id: inst.review_text for inst in instances}
    print(f"Loaded {len(review_texts)} review texts")

    # Run fine-tuned model
    ft_results = run_ft_model(review_texts, val_ids)

    # Run baseline model
    baseline_results = run_baseline_model(review_texts, val_ids)

    # Compare
    print("\n" + "=" * 70)
    print("COMPARISON: Fine-Tuned vs Baseline (Task 1.2)")
    print("=" * 70)
    print(f"{'Metric':<20} {'Fine-Tuned':>20} {'Baseline':>20} {'Improvement':>15}")
    print("-" * 75)
    print(
        f"{'Span Accuracy':<20} {ft_results['accuracy']:>19.1f}% {baseline_results['accuracy']:>19.1f}% "
        f"{ft_results['accuracy'] - baseline_results['accuracy']:>+14.1f}%"
    )
    print(
        f"{'Valid Spans':<20} {ft_results['correct']:>20} {baseline_results['correct']:>20}"
    )
    print(
        f"{'Total Examples':<20} {ft_results['total']:>20} {baseline_results['total']:>20}"
    )
    print("=" * 75)


if __name__ == "__main__":
    main()
