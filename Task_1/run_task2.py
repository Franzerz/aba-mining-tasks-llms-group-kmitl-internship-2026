from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from dotenv import load_dotenv

from src import (
    build_client,
    load_body_labels_config,
    load_model_config,
    load_paths_config,
    load_task2_instances_gt,
    load_task2_instances_llm,
    run_task2,
)

LLM_CSVS = {
    "llama4_scout": "outputs/task1/llama4_scout/modular/subtask1_2/task1_llama4_scout_extended11_subtask1_2.csv",
    "llama3.2":     "outputs/task1/llama3.2/modular/subtask1_2/task1_llama3.2_extended11_subtask1_2.csv",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Task 2 body label prediction")
    parser.add_argument(
        "--source",
        choices=["gt", "llm", "both"],
        default="both",
        help="gt = use GT selected content; llm = use Task 1.2 LLM output; both = run both (default: both)",
    )
    parser.add_argument("--n", type=int, default=20, help="Max instances to process (default: 20)")
    parser.add_argument("--offset", type=int, default=0, help="Skip first N instances (default: 0)")
    parser.add_argument(
        "--model",
        default=None,
        help="Override model from model.yaml (e.g. --model llama4:scout)",
    )
    parser.add_argument(
        "--llm-csv",
        default=None,
        help="Path to Task 1.2 LLM output CSV (for --source llm or both). "
             "If omitted, uses all known LLM CSVs.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    load_dotenv(repo_root / ".env", override=False)

    model_cfg     = load_model_config(repo_root)
    paths_cfg     = load_paths_config(repo_root)
    body_lbl_cfg  = load_body_labels_config(repo_root)

    if args.model:
        model_cfg = replace(model_cfg, task1_model=args.model, validator_model=args.model)

    client = build_client(model_cfg.provider, ollama_options=model_cfg.ollama_options)
    model_folder = model_cfg.task1_model.replace(":", "_").replace("/", "_").replace("-", "_")

    run_gt  = args.source in ("gt", "both")
    run_llm = args.source in ("llm", "both")

    if run_gt:
        print("\n" + "=" * 60)
        print(f"SOURCE: gt  |  model: {model_cfg.task1_model}  |  n={args.n}")
        print("=" * 60)
        instances = load_task2_instances_gt(
            paths_cfg.gold_csv,
            body_lbl_cfg,
            limit=args.n,
            offset=args.offset,
        )
        print(f"Loaded {len(instances)} GT instances")
        run_task2(
            repo_root=repo_root,
            client=client,
            body_labels_cfg=body_lbl_cfg,
            model_cfg=model_cfg,
            paths_cfg=paths_cfg,
            instances=instances,
            source="gt",
            output_subdir=f"{model_folder}/gt",
            label=model_folder,
        )

    if run_llm:
        llm_csv_paths: list[Path] = []
        if args.llm_csv:
            llm_csv_paths = [Path(args.llm_csv)]
        else:
            for name, rel in LLM_CSVS.items():
                p = repo_root / rel
                if p.exists():
                    llm_csv_paths.append(p)
                else:
                    print(f"[SKIP] LLM CSV not found: {p}")

        for llm_csv in llm_csv_paths:
            model_label = llm_csv.parent.parent.parent.name  # e.g. llama4_scout
            print("\n" + "=" * 60)
            print(f"SOURCE: llm  |  model: {model_cfg.task1_model}  |  csv: {llm_csv.name}  |  n={args.n}")
            print("=" * 60)
            instances = load_task2_instances_llm(
                llm_csv,
                body_lbl_cfg,
                limit=args.n,
                offset=args.offset,
            )
            print(f"Loaded {len(instances)} LLM instances from {model_label}")
            run_task2(
                repo_root=repo_root,
                client=client,
                body_labels_cfg=body_lbl_cfg,
                model_cfg=model_cfg,
                paths_cfg=paths_cfg,
                instances=instances,
                source="llm",
                output_subdir=f"{model_folder}/llm_{model_label}",
                label=f"{model_folder}_from_{model_label}",
            )

    print("\nDone.")


if __name__ == "__main__":
    main()
