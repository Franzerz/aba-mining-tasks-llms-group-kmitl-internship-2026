#### PATCHARAKORN ####

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

# Import this script's own sibling src/ folder directly (not via a hardcoded
# "internship.ABA_mining.src" dotted path) — location-independent regardless of what the
# containing folder is named or how deep it's nested, so cloning/copying this project
# anywhere still works without import errors.
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv

from src import (
    build_client,
    load_model_config,
    load_paths_config,
    load_task3_instances,
    run_task3,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Task 3 - Contrary (attack-relation) Yes/No classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_task3.py llama3.2                                   # baseline zero-shot, all aspects, 3 runs
  python run_task3.py llama4:scout --prompt prompts/task3/one_shot.txt
  python run_task3.py llama3.2 --prompt prompts/task3/contrary_v3.txt   # chain-of-thought version
  python run_task3.py llama3.2 --n 5 --aspects check-in           # quick test
  python run_task3.py llama3.2 --n-runs 1                         # single run (no consistency check)
  python run_task3.py llama3.2 --aspects facility --categories "Contrary(P)Body(P)"
        """,
    )
    parser.add_argument("model_name", nargs="?", default=None,
                        help="Optional positional model alias for --model")
    parser.add_argument("--model", default=None,
                        help="Override model from model.yaml (e.g. --model llama3.2)")
    parser.add_argument("--prompt", default="prompts/task3/zero_shot.txt",
                        help="Prompt template path (default: prompts/task3/zero_shot.txt). "
                             "Also see prompts/task3/one_shot.txt and contrary_v1..v6.txt")
    parser.add_argument("--n-runs", type=int, default=3,
                        help="Number of repeated runs for consistency checking (default: 3)")
    parser.add_argument("--n", type=int, default=None,
                        help="Max instances per (aspect, category) to process, applied after "
                             "--max-pairs-per-category (default: all)")
    parser.add_argument("--aspects", nargs="+", default=None,
                        help="Restrict to specific aspects, e.g. --aspects check-in staff")
    parser.add_argument("--categories", nargs="+", default=None,
                        help="Restrict to specific polarity categories, e.g. --categories \"Contrary(P)Body(N)\". "
                             "Default: all categories with finished human votes (see "
                             "INCOMPLETE_ASPECT_CATEGORIES in src/task3.py)")
    parser.add_argument("--max-pairs-per-category", type=int, default=500,
                        help="Cap on pairs per (aspect, category) group (default: 500; some categories "
                             "run into the tens of thousands of pairs). 0 = no cap.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Sampling seed for --max-pairs-per-category (default: 42, for reproducibility)")
    parser.add_argument("--max-tokens", type=int, default=300,
                        help="Max output tokens per call (default: 300; CoT/recipe prompts need room to reason)")
    parser.add_argument("--no-resume", action="store_true",
                        help="Ignore existing output files and re-run everything")
    args = parser.parse_args()

    repo_root = REPO_ROOT
    load_dotenv(repo_root / ".env", override=False)

    model_cfg = load_model_config(repo_root)
    paths_cfg = load_paths_config(repo_root)

    model_override = args.model or args.model_name
    if model_override:
        model_cfg = replace(model_cfg, task1_model=model_override, validator_model=model_override)

    client = build_client(model_cfg.provider, ollama_options=model_cfg.ollama_options)
    model_folder = model_cfg.task1_model.replace(":", "_").replace("/", "_").replace("-", "_")
    prompt_label = Path(args.prompt).stem  # e.g. "zero_shot", "contrary_v3"

    print("\n" + "=" * 60)
    print(f"Task 3 - Contrary Classification")
    print(f"Model      : {model_cfg.task1_model}")
    print(f"Prompt     : {args.prompt}")
    print(f"Aspects    : {'all' if args.aspects is None else ', '.join(args.aspects)}")
    print(f"Categories : {'all usable' if args.categories is None else ', '.join(args.categories)}")
    print(f"Runs       : {args.n_runs}")
    print("=" * 60 + "\n")

    instances = load_task3_instances(
        paths_cfg.task3_dir,
        aspects=args.aspects,
        categories=args.categories,
        max_per_group=args.max_pairs_per_category or None,
        seed=args.seed,
    )
    if args.n is not None:
        limited: dict[tuple[str, str], int] = {}
        capped = []
        for inst in instances:
            key = (inst.aspect, inst.category)
            count = limited.get(key, 0)
            if count >= args.n:
                continue
            capped.append(inst)
            limited[key] = count + 1
        instances = capped

    groups = sorted({(i.aspect, i.category) for i in instances})
    print(f"Loaded {len(instances)} instance(s) across {len(groups)} (aspect, category) group(s):")
    for aspect, category in groups:
        n = sum(1 for i in instances if i.aspect == aspect and i.category == category)
        print(f"  {aspect:<12} {category:<20} {n}")
    print()

    output_dir = repo_root / "outputs" / "task3" / model_folder / prompt_label

    for run_idx in range(1, args.n_runs + 1):
        print(f"[{run_idx}/{args.n_runs}] Running Task 3 [{prompt_label}]...\n")
        written = run_task3(
            client=client,
            model_cfg=model_cfg,
            repo_root=repo_root,
            instances=instances,
            prompt_path=args.prompt,
            label=prompt_label,
            run_idx=run_idx,
            n_runs=args.n_runs,
            output_dir=output_dir,
            max_output_tokens=args.max_tokens,
            resume=not args.no_resume,
        )
        for aspect, path in written.items():
            print(f"  -> [{aspect}] {path}")
        print()

    print("=" * 60)
    print(f"All {args.n_runs} run(s) completed under: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
