#### PATCHARAKORN ####

"""Compare fine-tuned vs baseline results on Task 1.2 (span extraction)."""
import json
from pathlib import Path

FT_RESULTS = Path("outputs/task1/llama3.2_ft/task1_predictions.json")
BASELINE_RESULTS = Path("outputs/task1/llama3.2_baseline/task1_predictions.json")
VAL_IDS_PATH = Path("ABA_mining/finetune/data/val_ids.json")

print("=" * 70)
print("Task 1.2 COMPARISON: Fine-tuned vs Baseline (Standard llama3.2)")
print("=" * 70)

# Load results
with open(FT_RESULTS) as f:
    ft_data = json.load(f)
with open(BASELINE_RESULTS) as f:
    baseline_data = json.load(f)
with open(VAL_IDS_PATH) as f:
    val_ids = json.load(f)

print(f"\nValidation set: {len(val_ids)} review IDs")
print(f"Fine-tuned results: {len(ft_data)} predictions")
print(f"Baseline results: {len(baseline_data)} predictions")

# Count non-null spans
def count_spans(data):
    total = 0
    for pred in data.values():
        for topic_spans in pred.get("prediction", {}).get("Topics", {}).values():
            if isinstance(topic_spans, list):
                for item in topic_spans:
                    if isinstance(item, dict) and item.get("text") is not None:
                        total += 1
    return total

ft_spans = count_spans(ft_data)
baseline_spans = count_spans(baseline_data)

print("\n" + "=" * 70)
print("SPAN EXTRACTION (Task 1.2) SUMMARY")
print("=" * 70)
print(f"\nFine-tuned spans predicted:   {ft_spans:4d}")
print(f"Baseline spans predicted:     {baseline_spans:4d}")
print(f"Difference:                   {ft_spans - baseline_spans:+4d}")

if ft_spans > baseline_spans:
    improvement = ((ft_spans - baseline_spans) / baseline_spans * 100) if baseline_spans > 0 else 0
    print(f"Fine-tuned extraction: +{improvement:.1f}% more spans")
elif baseline_spans > ft_spans:
    diff_pct = ((baseline_spans - ft_spans) / baseline_spans * 100) if baseline_spans > 0 else 0
    print(f"Baseline extraction: +{diff_pct:.1f}% more spans")
else:
    print("Both extract same number of spans")

print("\n" + "=" * 70)
print("RESULTS FILES READY FOR DETAILED EVALUATION")
print("=" * 70)
print(f"Fine-tuned: {FT_RESULTS.resolve()}")
print(f"Baseline:   {BASELINE_RESULTS.resolve()}")
print("\nUse src/eval.py for detailed precision/recall/F1 metrics")
print("=" * 70)
