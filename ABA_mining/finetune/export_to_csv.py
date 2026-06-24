#### PATCHARAKORN ####

"""Export fine-tuned model results to CSV format."""
import json
from pathlib import Path
import pandas as pd

FT_RESULTS = Path("outputs/task1/llama3.2_ft/task1_predictions.json")
CSV_OUTPUT = Path("outputs/task1/llama3.2_ft/task1_predictions.csv")

print("Loading fine-tuned predictions...")
with open(FT_RESULTS, encoding="utf-8") as f:
    data = json.load(f)

rows = []
for review_id, pred_data in data.items():
    review_text = pred_data.get("review_text", "")
    topics_dict = pred_data.get("prediction", {}).get("Topics", {})

    for topic, spans_list in topics_dict.items():
        if not isinstance(spans_list, list):
            spans_list = [spans_list]

        for idx, span_item in enumerate(spans_list):
            if span_item is None:
                continue

            text = span_item.get("text") if isinstance(span_item, dict) else None
            label = span_item.get("label") if isinstance(span_item, dict) else None

            rows.append({
                "Review_ID": review_id,
                "Topic": topic,
                "Span_Index": idx + 1,
                "Text": text,
                "Label": label,
                "Review_Text": review_text[:100] + "..." if len(str(review_text)) > 100 else review_text
            })

df = pd.DataFrame(rows)
CSV_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(CSV_OUTPUT, index=False, encoding="utf-8")

print(f"Exported {len(rows)} spans to CSV")
print(f"  File: {CSV_OUTPUT.resolve()}")
print(f"  Columns: Review_ID, Topic, Span_Index, Text, Label, Review_Text")
print(f"\nSummary:")
print(f"  Total reviews: {len(data)}")
print(f"  Total spans: {len(rows)}")
print(f"  Null spans: {sum(1 for r in rows if r['Text'] is None)}")
