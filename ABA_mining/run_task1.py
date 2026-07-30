#### PATCHARAKORN ####

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import yaml
from dotenv import load_dotenv

from src import load_model_config, load_paths_config, load_topics_config
from src import build_client
from src import run_task1
from src.prompts import build_modular_prompt


def _load_experiments(repo_root: Path) -> dict:
    cfg_path = repo_root / "configs" / "experiments.yaml"
    with cfg_path.open(encoding="utf-8") as f:
        return yaml.safe_load(f).get("experiments", {})


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Task 1 mining pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full dataset — processes every ID one at a time, saves after each
  python3 run_task1.py                              # default: combined (all 7 rules)
  python3 run_task1.py --experiment combined        # rules 1-7, all subtasks
  python3 run_task1.py --experiment subtask1_1      # rules 2,5,6 — topic only
  python3 run_task1.py --experiment subtask1_2      # rules 1,7   — span only
  python3 run_task1.py --experiment subtask1_3      # rules 3,4   — sentiment only

  # Label the run (for comparing run 1 vs run 2)
  python3 run_task1.py --experiment combined --run 1
  python3 run_task1.py --experiment combined --run 2

  # Quick test with only 5 IDs
  python3 run_task1.py --experiment combined --n 5

  # Override model
  python3 run_task1.py --experiment combined --model llama4:scout
        """,
    )
    parser.add_argument(
        "--experiment",
        default="combined",
        help="Experiment name from configs/experiments.yaml (default: combined).",
    )
    parser.add_argument("--n", type=int, default=None, help="Limit number of IDs (default: whole dataset). Use --n 5 for quick testing.")
    parser.add_argument("--run", type=int, default=None, help="Run number for folder naming, e.g. 1, 2, 3")
    parser.add_argument("--num-versions", type=int, default=1, help="Run task multiple times to create versions (default: 1). Use --num-versions 3 to run 3 times automatically.")
    parser.add_argument("--model", default=None, help="Override model from model.yaml, e.g. --model llama4:scout")
    parser.add_argument("--runs", type=int, default=3, help="Number of times to call the LLM per ID (default: 3).")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    load_dotenv(repo_root / ".env", override=False)

    topics_cfg = load_topics_config(repo_root)
    model_cfg = load_model_config(repo_root)
    paths_cfg = load_paths_config(repo_root)

    # Override output path to use ABA_mining/outputs
    paths_cfg = replace(paths_cfg, task1_dir=repo_root / "outputs" / "task1")

    if args.model:
        model_cfg = replace(model_cfg, task1_model=args.model, validator_model=args.model)

    client = build_client(model_cfg.provider, ollama_options=model_cfg.ollama_options)

    # ── Load experiment ───────────────────────────────────────────────────────
    experiments = _load_experiments(repo_root)
    if args.experiment not in experiments:
        available = ", ".join(experiments.keys())
        parser.error(f"Unknown experiment '{args.experiment}'. Available: {available}")

    exp = experiments[args.experiment]
    rule_list = [int(r) for r in exp["rules"]]
    output_schema = exp["output_schema"]
    description = exp.get("description", "")

    # ── Assemble prompt from rule files ───────────────────────────────────────
    # Combines header + rule2.txt + rule5.txt + rule6.txt + footer (for subtask1_1 example)
    prompt_template = build_modular_prompt(repo_root, rule_list, output_schema)

    # ── Build output folder: model/modular/experiment/versionX ───────────────
    model_folder = model_cfg.task1_model.replace(":", "_").replace("/", "_").replace("-", "_")

    print("\n" + "=" * 60)
    print(f"Model      : {model_cfg.task1_model}")
    print(f"Experiment : {args.experiment}")
    print(f"Rules      : {rule_list}")
    print(f"Schema     : {output_schema}")
    print(f"Description: {description}")
    print(f"Dataset    : {'all IDs' if args.n is None else f'first {args.n} IDs (test mode)'}")
    print(f"Runs/ID    : {args.runs}")
    print(f"Versions   : {args.num_versions}")
    print("=" * 60 + "\n")

    # Run multiple rounds if requested
    output_paths = []
    for run_num in range(1, args.num_versions + 1):
        run_label = f"run{run_num}"
        output_subdir = f"{model_folder}/modular/{args.experiment}"

        print(f"[{run_num}/{args.num_versions}] Running {args.experiment} - {run_label}...")

        out_path = run_task1(
            repo_root=repo_root,
            client=client,
            topics_cfg=topics_cfg,
            model_cfg=model_cfg,
            paths_cfg=paths_cfg,
            limit_reviews=args.n,
            prompt_template=prompt_template,
            output_subdir=output_subdir,
            output_label=f"{args.experiment}_{run_label}",
            output_schema=output_schema,
            runs_per_id=args.runs,
        )
        output_paths.append(out_path)
        print(f"  → Wrote: {out_path}\n")

    print("=" * 60)
    print(f"All {args.num_versions} run(s) completed in same location:")
    for i, path in enumerate(output_paths, 1):
        print(f"  run{i}: {path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
