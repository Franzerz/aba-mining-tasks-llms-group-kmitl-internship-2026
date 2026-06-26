#### PATCHARAKORN ####

"""Evaluate fine-tuned model on full dataset (all 784 IDs) vs baseline.

Compares fine-tuned model with baseline Llama-3.2 on full review dataset.
Uses 80/20 split was only for training - this test uses all 784 IDs.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")
os.environ.setdefault("HF_HOME", str((ROOT.parent / ".hf_cache").resolve()))

from transformers import AutoModelForCausalLM, AutoTokenizer
from internship.ABA_mining.src import load_paths_config, load_topics_config, build_client
from internship.ABA_mining.src.task1 import load_task1_instances_from_input, local_validate_task1
from internship.ABA_mining.src.prompts import build_modular_prompt, render_prompt

FT_MODEL_DIR = ROOT / "finetune" / "output" / "task1_merged"


def main():
    print("=" * 70)
    print("TASK 1.2 FULL DATASET EVALUATION")
    print("Fine-Tuned vs Baseline (all 784 review IDs)")
    print("=" * 70)

    # Load configs
    paths_cfg = load_paths_config(ROOT)
    topics_cfg = load_topics_config(ROOT)
    topics = topics_cfg.topics
    print(f"\nTopics: {topics}\n")

    # Load all instances
    instances = load_task1_instances_from_input(paths_cfg, limit_reviews=None)
    print(f"Loaded {len(instances)} review instances")

    # Build Task 1.2 prompt (span extraction only)
    rule_list = [1, 7]  # Span extraction rules
    output_schema = "span_only"
    prompt_template = build_modular_prompt(ROOT, rule_list, output_schema)
    topics_str = ", ".join(topics)

    # ========== FINE-TUNED MODEL ==========
    print("\n" + "=" * 70)
    print("FINE-TUNED MODEL EVALUATION")
    print("=" * 70)

    print("\nLoading fine-tuned model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        ft_model = AutoModelForCausalLM.from_pretrained(
            str(FT_MODEL_DIR),
            torch_dtype=torch.bfloat16,
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained(str(FT_MODEL_DIR))
        ft_model.eval()
        print(f"Device: {device}")
        print(f"Model loaded successfully\n")

        ft_correct = 0
        ft_total = 0
        ft_errors = []

        for i, inst in enumerate(instances):
            if (i + 1) % 100 == 0:
                print(f"Processing {i + 1}/{len(instances)}...")

            try:
                # Render prompt
                prompt = render_prompt(
                    prompt_template,
                    TOPICS=topics_str,
                    REVIEW_TEXT=inst.review_text,
                )

                # Generate
                messages = [{"role": "user", "content": prompt}]
                rendered = tokenizer.apply_chat_template(messages, tokenize=False)
                inputs = tokenizer(rendered, return_tensors="pt").to(device)

                with torch.no_grad():
                    outputs = ft_model.generate(
                        **inputs,
                        max_new_tokens=512,
                        temperature=0.0,
                        top_p=0.05,
                        do_sample=False,
                    )

                response = tokenizer.decode(outputs[0], skip_special_tokens=True)

                # Extract JSON
                if "{" in response:
                    json_str = response[response.rfind("{") :]
                    result = json.loads(json_str)

                    # Validate
                    errors = local_validate_task1(
                        result, topics=topics, review_text=inst.review_text, output_schema=output_schema
                    )
                    if not errors:
                        ft_correct += 1
                    ft_total += 1

            except Exception as e:
                ft_errors.append({"id": inst.review_id, "error": str(e)[:100]})

        ft_accuracy = (ft_correct / ft_total * 100) if ft_total > 0 else 0
        print(f"\nFine-Tuned Results (Full Dataset):")
        print(f"  Valid spans: {ft_correct} / {ft_total}")
        print(f"  Span accuracy: {ft_accuracy:.1f}%")
        print(f"  Errors: {len(ft_errors)}")

    except Exception as e:
        print(f"ERROR loading fine-tuned model: {e}")
        ft_accuracy = 0
        ft_correct = 0
        ft_total = 0

    # ========== BASELINE MODEL ==========
    print("\n" + "=" * 70)
    print("BASELINE MODEL EVALUATION")
    print("=" * 70)

    print("\nLoading baseline model (Ollama llama3.2)...")
    client = build_client("ollama")

    try:
        baseline_correct = 0
        baseline_total = 0
        baseline_errors = []

        for i, inst in enumerate(instances):
            if (i + 1) % 100 == 0:
                print(f"Processing {i + 1}/{len(instances)}...")

            try:
                # Render prompt
                prompt = render_prompt(
                    prompt_template,
                    TOPICS=topics_str,
                    REVIEW_TEXT=inst.review_text,
                )

                # Generate via Ollama
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

                # Extract JSON
                if "{" in result_text:
                    json_str = result_text[result_text.rfind("{") :]
                    result = json.loads(json_str)

                    # Validate
                    errors = local_validate_task1(
                        result, topics=topics, review_text=inst.review_text, output_schema=output_schema
                    )
                    if not errors:
                        baseline_correct += 1
                    baseline_total += 1

            except Exception as e:
                baseline_errors.append({"id": inst.review_id, "error": str(e)[:100]})

        baseline_accuracy = (baseline_correct / baseline_total * 100) if baseline_total > 0 else 0
        print(f"\nBaseline Results (Full Dataset):")
        print(f"  Valid spans: {baseline_correct} / {baseline_total}")
        print(f"  Span accuracy: {baseline_accuracy:.1f}%")
        print(f"  Errors: {len(baseline_errors)}")

    except Exception as e:
        print(f"ERROR with baseline model: {e}")
        baseline_accuracy = 0
        baseline_correct = 0
        baseline_total = 0

    # ========== COMPARISON ==========
    print("\n" + "=" * 70)
    print("COMPARISON: Fine-Tuned vs Baseline")
    print("=" * 70)
    improvement = ft_accuracy - baseline_accuracy
    print(f"\n{'Metric':<25} {'Fine-Tuned':>20} {'Baseline':>20} {'Difference':>15}")
    print("-" * 80)
    print(
        f"{'Span Accuracy':<25} {ft_accuracy:>19.1f}% {baseline_accuracy:>19.1f}% {improvement:>+14.1f}%"
    )
    print(
        f"{'Valid Spans':<25} {ft_correct:>20} {baseline_correct:>20}"
    )
    print(
        f"{'Total Evaluated':<25} {ft_total:>20} {baseline_total:>20}"
    )
    print("=" * 80)

    if improvement > 0:
        print(f"\n✅ Fine-tuned model improved span accuracy by {improvement:.1f}%!")
    elif improvement < 0:
        print(f"\n⚠️ Fine-tuned model is {abs(improvement):.1f}% lower than baseline")
    else:
        print(f"\n→ Models have same accuracy")


if __name__ == "__main__":
    main()
