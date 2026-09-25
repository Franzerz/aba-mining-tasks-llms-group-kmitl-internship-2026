#### PATCHARAKORN ####

"""
Runs run_task3.py once per (model, prompt version) pair, so a full sweep across every
model and every prompt variant is a single command.

Checkpointed: every prediction is written to disk the instant it completes (see
src/task3.py::run_task3), and this script re-checks each (model, version)'s own resume state
before moving on, so if the process gets killed partway through (e.g. a remote server or tmux
session closing) — for ANY reason, mid-version, mid-model, or between them — just re-run the
exact same command. Already completed (model, version, run, pair) combinations are skipped
automatically; nothing is lost or redone.

If a single (model, version) errors out (as opposed to the whole process dying), it's logged
and skipped so the rest of the sweep still runs — re-run the same command afterward to retry
just the failed one(s).

--git-push: after each (model, version) finishes, commits and pushes outputs/ so results are
backed up even if the server/tmux session dies before the whole sweep completes. Requires the
repo to already have a remote configured with non-interactive auth (SSH key / stored
credentials) — this runs unattended, so a credential prompt would just hang forever. Test
`git push` manually once before relying on this.

Usage:
  python run_task3_all.py llama3.2 --aspects check-in check-out price staff
  python run_task3_all.py llama3.2 --versions zero_shot contrary_v3   # subset
  python run_task3_all.py llama4:scout --n-runs 1 --n 20              # quick test, all versions

  # Multiple models in order, all 5 gold-data topics, backed up to git after each step:
  python run_task3_all.py --models llama3.2 qwen3.8:27b gemma4:31b llama4:scout llama3.3:70b \\
      --aspects check-in check-out price staff facility --git-push
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
PROMPTS_DIR = REPO_ROOT / "prompts" / "task3"

ALL_PROMPT_VERSIONS = [
    "zero_shot", "one_shot",
    "contrary_v1", "contrary_v2", "contrary_v3", "contrary_v4", "contrary_v5", "contrary_v6",
]


def git_checkpoint(message: str) -> None:
    """Best-effort git add/commit/push of outputs/ — never fatal to the sweep itself.
    'Nothing to commit' (e.g. a fully-resumed no-op step) is expected and not an error."""
    try:
        subprocess.run(["git", "add", "outputs/"], cwd=REPO_ROOT, check=True)
        commit = subprocess.run(["git", "commit", "-m", message], cwd=REPO_ROOT)
        if commit.returncode != 0:
            print("[git] nothing new to commit — skipping push")
            return
        push = subprocess.run(["git", "push"], cwd=REPO_ROOT)
        if push.returncode != 0:
            print("[git] push failed (network? auth?) — will retry at the next checkpoint. Continuing sweep.")
    except Exception as e:  # noqa: BLE001 — this is a best-effort backup, never block the sweep on it
        print(f"[git] error during git checkpoint: {e} — continuing without git backup for this step.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Task 3 across every model and prompt version in one command.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("model_name", nargs="?", default=None,
                        help="Single model (positional). Use --models instead for multiple.")
    parser.add_argument("--model", default=None, help="Single model (alternative to positional).")
    parser.add_argument("--models", nargs="+", default=None,
                         help="Run multiple models in this exact order, e.g. "
                              "--models llama3.2 qwen3.8:27b gemma4:31b llama4:scout llama3.3:70b. "
                              "Overrides model_name/--model if given.")
    parser.add_argument("--aspects", nargs="+", default=None,
                        help="Restrict to specific aspects, e.g. --aspects check-in check-out price staff facility")
    parser.add_argument("--categories", nargs="+", default=None)
    parser.add_argument("--versions", nargs="+", default=ALL_PROMPT_VERSIONS,
                         help=f"Which prompt versions to run (default: all 8: {', '.join(ALL_PROMPT_VERSIONS)})")
    parser.add_argument("--n-runs", type=int, default=3)
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--max-pairs-per-category", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--git-push", action="store_true",
                         help="git add/commit/push outputs/ after each (model, version) completes.")
    args = parser.parse_args()

    unknown_versions = [v for v in args.versions if not (PROMPTS_DIR / f"{v}.txt").exists()]
    if unknown_versions:
        sys.exit(f"[ERROR] No prompt file for: {unknown_versions} "
                  f"(expected under {PROMPTS_DIR.relative_to(REPO_ROOT)}/)")

    if args.models:
        models = args.models
    else:
        single = args.model or args.model_name
        if not single:
            sys.exit("[ERROR] Provide a model (positional or --model), or a list via --models.")
        models = [single]

    failed: list[str] = []
    total_jobs = len(models) * len(args.versions)
    job_i = 0

    for model in models:
        for version in args.versions:
            job_i += 1
            cmd = [sys.executable, str(REPO_ROOT / "run_task3.py"), model,
                   "--prompt", f"prompts/task3/{version}.txt", "--n-runs", str(args.n_runs)]
            if args.aspects:
                cmd += ["--aspects", *args.aspects]
            if args.categories:
                cmd += ["--categories", *args.categories]
            if args.n is not None:
                cmd += ["--n", str(args.n)]
            if args.max_pairs_per_category is not None:
                cmd += ["--max-pairs-per-category", str(args.max_pairs_per_category)]
            if args.max_tokens is not None:
                cmd += ["--max-tokens", str(args.max_tokens)]
            if args.no_resume:
                cmd.append("--no-resume")

            print(f"\n{'#' * 70}")
            print(f"[{job_i}/{total_jobs}] Model: {model}   Prompt version: {version}")
            print(f"{'#' * 70}")
            try:
                subprocess.run(cmd, cwd=REPO_ROOT, check=True)
            except subprocess.CalledProcessError as e:
                failed.append(f"{model}/{version}")
                print(f"\n[ERROR] '{model}/{version}' exited with code {e.returncode} — "
                      f"continuing to the next one.")
                print(f"        Already-completed pairs are safe on disk (checkpointed). "
                      f"Re-run this same command later to retry the rest.")
                continue

            if args.git_push:
                git_checkpoint(f"Task3: {model} / {version} complete")

    print(f"\n{'=' * 70}")
    if failed:
        print(f"{total_jobs - len(failed)}/{total_jobs} (model, version) job(s) completed. "
              f"Failed: {', '.join(failed)}")
        print("Re-run this exact command to retry the failed job(s) and pick up any "
              "partially-completed ones — already-finished pairs are skipped automatically.")
    else:
        print(f"All {total_jobs} (model, version) job(s) completed.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
