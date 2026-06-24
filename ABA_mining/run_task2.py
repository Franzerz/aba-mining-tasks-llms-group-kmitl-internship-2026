#### PATCHARAKORN ####

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

# Ensure the local src/ package is found before any installed 'src' PyPI package
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv

from internship.ABA_mining.src import (
    build_client,
    load_model_config,
    load_paths_config,
    load_task2_instances_gt,
    run_task2,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Task 2 — body literal generation (standalone, GT input)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_task2.py --model llama3.2          # run all 2265 GT instances
  python run_task2.py --model llama4:scout      # run with llama4_scout on server
  python run_task2.py --model llama3.2 --n 20   # quick test with 20 instances
  python run_task2.py --model llama3.2 --num-versions 3  # run 3 versions automatically
        """,
    )
    parser.add_argument("--n", type=int, default=None,
                        help="Max instances to process (default: all)")
    parser.add_argument("--offset", type=int, default=0,
                        help="Skip first N instances (default: 0)")
    parser.add_argument("--model", default=None,
                        help="Override model from model.yaml (e.g. --model llama3.2)")
    parser.add_argument("--prompt", default="prompts/task2/generator_v1.txt",
                        help="Prompt template path (default: prompts/task2/generator_v1.txt)")
    parser.add_argument("--num-versions", type=int, default=1,
                        help="Number of versions to run (default: 1). Use --num-versions 3 for 3 versions.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    load_dotenv(repo_root / ".env", override=False)

    model_cfg = load_model_config(repo_root)
    paths_cfg = load_paths_config(repo_root)

    # Override output path to use ABA_mining/outputs
    paths_cfg = replace(paths_cfg, task1_dir=repo_root / "outputs" / "task1")

    if args.model:
        model_cfg = replace(model_cfg, task1_model=args.model, validator_model=args.model)

    client = build_client(model_cfg.provider, ollama_options=model_cfg.ollama_options)
    model_folder = model_cfg.task1_model.replace(":", "_").replace("/", "_").replace("-", "_")
    prompt_label = Path(args.prompt).stem  # e.g. generator_v1

    print("\n" + "=" * 60)
    print(f"Task 2 — Body Literal Generation")
    print(f"Model    : {model_cfg.task1_model}")
    print(f"Prompt   : {args.prompt}")
    print(f"Dataset  : {'all instances' if args.n is None else f'first {args.n} instances'}")
    print(f"Versions : {args.num_versions}")
    print("=" * 60 + "\n")

    instances = load_task2_instances_gt(
        paths_cfg.gold_csv,
        limit=args.n,
        offset=args.offset,
    )
    print(f"Loaded {len(instances)} GT instances\n")

    # Extract version number from prompt file (e.g., generator_v1 → version1)
    version_match = prompt_label.replace("generator_", "")  # e.g., "v1" → "v1"
    version_folder = f"version{version_match[1]}" if version_match.startswith("v") else "version1"  # v1 → version1

    output_paths = []

    for run_num in range(1, args.num_versions + 1):
        run_label = f"run{run_num}"
        print(f"[{run_num}/{args.num_versions}] Running Task 2 [{version_folder}] - {run_label}...\n")

        out_path = run_task2(
            repo_root=repo_root,
            client=client,
            model_cfg=model_cfg,
            paths_cfg=paths_cfg,
            instances=instances,
            source="gt",
            prompt_path=args.prompt,
            output_subdir=f"gt/{model_folder}/{version_folder}",
            label=f"{model_folder}_{prompt_label}_{run_label}",
        )
        output_paths.append(out_path)
        print(f"  → Wrote: {out_path}\n")

    print("=" * 60)
    print(f"All {args.num_versions} run(s) completed in {version_folder}:")
    for i, path in enumerate(output_paths, 1):
        print(f"  run{i}: {path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
