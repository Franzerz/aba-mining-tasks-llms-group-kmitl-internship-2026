#### PATCHARAKORN ####

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

# Ensure the local src/ package is found before any installed 'src' PyPI package
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv

from src import (
    build_client,
    load_model_config,
    load_paths_config,
    load_task2_instances_gt,
    run_task2,
)


class _FastOllamaClient:
    """Ollama client that disables 'thinking' output.

    Reasoning-capable models (Qwen3.x, Gemma thinking variants, DeepSeek-R1,
    gpt-oss, ...) silently spend hundreds of extra tokens on a hidden
    chain-of-thought before the JSON answer even starts. Llama has no
    thinking mode, which is why it's the only model that runs fast here —
    passing think=False puts the other models on equal footing.
    """

    def __init__(self, options: dict) -> None:
        import ollama

        self._ollama = ollama
        self._options = dict(options or {})
        self._think_supported = True

    def complete(self, *, model: str, prompt: str, temperature: float, top_p: float, max_output_tokens: int):
        from src.llm import LLMResponse

        options = dict(self._options)
        options.setdefault("temperature", temperature)
        options.setdefault("top_p", top_p)
        options.setdefault("num_predict", max_output_tokens)

        if self._think_supported:
            try:
                resp = self._ollama.generate(
                    model=model, prompt=prompt, stream=False, think=False, options=options,
                )
                return LLMResponse(text=(resp.get("response") or "").strip())
            except TypeError:
                # Installed ollama-python client predates the `think` kwarg (<0.4) — disable for the rest of the run.
                self._think_supported = False

        resp = self._ollama.generate(model=model, prompt=prompt, stream=False, options=options)
        return LLMResponse(text=(resp.get("response") or "").strip())


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
  python run_task2.py --model llama3.2 --all-prompts     # run every prompts/task2/generator_v*.txt

Checkpointing:
  Each run writes its .jsonl row-by-row and fsyncs after every row. If the
  job is killed (e.g. SLURM time limit) and this script is resubmitted with
  the same arguments, already-finished versions are detected and skipped,
  and a partially-finished version resumes from its last completed row
  instead of starting over.
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
    parser.add_argument("--all-prompts", action="store_true",
                        help="Run every prompt template in prompts/task2/ (generator_v*.txt) instead of just --prompt.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    load_dotenv(repo_root / ".env", override=False)

    model_cfg = load_model_config(repo_root)
    paths_cfg = load_paths_config(repo_root)

    # Override output path to use ABA_mining/outputs
    paths_cfg = replace(paths_cfg, task1_dir=repo_root / "outputs" / "task1")

    if args.model:
        model_cfg = replace(model_cfg, task1_model=args.model, validator_model=args.model)

    if model_cfg.provider.lower().strip() == "ollama":
        client = _FastOllamaClient(options=model_cfg.ollama_options)
    else:
        client = build_client(model_cfg.provider, ollama_options=model_cfg.ollama_options)
    model_folder = model_cfg.task1_model.replace(":", "_").replace("/", "_").replace("-", "_")

    if args.all_prompts:
        prompt_paths = sorted(
            (repo_root / "prompts" / "task2").glob("generator_v*.txt"),
            key=lambda p: int(p.stem.replace("generator_v", "")),
        )
        if not prompt_paths:
            parser.error("--all-prompts was set but no prompts/task2/generator_v*.txt files were found")
    else:
        prompt_paths = [Path(args.prompt)]

    print("\n" + "=" * 60)
    print(f"Task 2 — Body Literal Generation")
    print(f"Model    : {model_cfg.task1_model}")
    print(f"Prompts  : {', '.join(str(p) for p in prompt_paths)}")
    print(f"Dataset  : {'all instances' if args.n is None else f'first {args.n} instances'}")
    print(f"Versions : {args.num_versions}")
    print("=" * 60 + "\n")

    instances = load_task2_instances_gt(
        paths_cfg.gold_csv,
        limit=args.n,
        offset=args.offset,
    )
    print(f"Loaded {len(instances)} GT instances\n")

    all_output_paths = {}

    for prompt_path in prompt_paths:
        prompt_label = prompt_path.stem  # e.g. generator_v1

        # Extract version number from prompt file (e.g., generator_v1 → version1)
        version_match = prompt_label.replace("generator_", "")  # e.g., "v1" → "v1"
        version_folder = f"version{version_match[1]}" if version_match.startswith("v") else "version1"  # v1 → version1

        output_paths = []

        for run_num in range(1, args.num_versions + 1):
            run_label = f"run{run_num}"
            print(f"[{version_folder} {run_num}/{args.num_versions}] Running Task 2 [{version_folder}] - {run_label}...\n")

            out_path = run_task2(
                repo_root=repo_root,
                client=client,
                model_cfg=model_cfg,
                paths_cfg=paths_cfg,
                instances=instances,
                source="gt",
                prompt_path=str(prompt_path),
                output_subdir=f"gt/{model_folder}/{version_folder}",
                label=f"{model_folder}_{prompt_label}_{run_label}",
            )
            output_paths.append(out_path)
            print(f"  → Wrote: {out_path}\n")

        all_output_paths[version_folder] = output_paths

    print("=" * 60)
    print(f"All prompt version(s) completed:")
    for version_folder, output_paths in all_output_paths.items():
        print(f"  {version_folder}:")
        for i, path in enumerate(output_paths, 1):
            print(f"    run{i}: {path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
