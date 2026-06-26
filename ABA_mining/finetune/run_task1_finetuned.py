#### HUGGINGFACE ####
#### https://huggingface.co/blog/ImranzamanML/fine-tuning-1b-llama-32-a-comprehensive-article ####

"""Run Task 1.2 span extraction with the fine-tuned model, writing the same
CSV/JSONL format as the baseline runs (run_task1.py), so the two can be
compared directly.
"""
from __future__ import annotations

import argparse
import os
from dataclasses import replace
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = ROOT / "finetune" / "output" / "task1_merged"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Task 1.2 with the fine-tuned model")
    parser.add_argument("--n", type=int, default=None, help="Limit number of IDs (default: all 784).")
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR), help="Path to merged fine-tuned model.")
    args = parser.parse_args()

    os.environ["HF_LOCAL_MODEL_DIR"] = args.model_dir
    load_dotenv(ROOT / ".env", override=False)
    os.environ.setdefault("HF_HOME", str((ROOT.parent / ".hf_cache").resolve()))

    import sys
    sys.path.insert(0, str(ROOT))
    from internship.ABA_mining.src import load_model_config, load_paths_config, load_topics_config, build_client, run_task1
    from internship.ABA_mining.src.prompts import build_modular_prompt

    topics_cfg = load_topics_config(ROOT)
    model_cfg = load_model_config(ROOT)
    paths_cfg = replace(load_paths_config(ROOT), task1_dir=ROOT / "outputs" / "task1")
    model_cfg = replace(model_cfg, task1_model="task1_finetuned", validator_model="task1_finetuned")

    client = build_client("hf_local")

    rule_list = [1, 7]  # Task 1.2: span extraction only
    output_schema = "span_only"
    prompt_template = build_modular_prompt(ROOT, rule_list, output_schema)

    print("\n" + "=" * 60)
    print(f"Model      : fine-tuned ({args.model_dir})")
    print(f"Experiment : subtask1_2")
    print(f"Rules      : {rule_list}")
    print(f"Schema     : {output_schema}")
    print(f"Dataset    : {'all IDs' if args.n is None else f'first {args.n} IDs (test mode)'}")
    print("=" * 60 + "\n")

    out_path = run_task1(
        repo_root=ROOT,
        client=client,
        topics_cfg=topics_cfg,
        model_cfg=model_cfg,
        paths_cfg=paths_cfg,
        limit_reviews=args.n,
        prompt_template=prompt_template,
        output_subdir="task1_finetuned/modular/subtask1_2",
        output_label="subtask1_2",
        output_schema=output_schema,
        runs_per_id=1,
    )

    print(f"\nWrote JSONL: {out_path}")
    print(f"Wrote CSV  : {out_path.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
