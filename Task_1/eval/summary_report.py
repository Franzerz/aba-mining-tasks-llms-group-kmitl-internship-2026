import re
import subprocess
import sys
import time
from pathlib import Path

import sys as _sys

# Paths
TASK_DIR = Path(__file__).resolve().parent.parent
LLM_DIR  = TASK_DIR / "outputs" / "task1"
EVAL_DIR    = TASK_DIR / "outputs" / "eval"
SUMMARY_DIR = EVAL_DIR / "summary"

_venv_base = Path(__file__).resolve().parents[2] / ".venv"
PYTHON = (_venv_base / "Scripts" / "python.exe"
          if _sys.platform == "win32"
          else _venv_base / "bin" / "python")

EVAL_PY = Path(__file__).resolve().parent

RESULT_DIRS = {
    "1.1": EVAL_DIR / "topic",
    "1.2": EVAL_DIR / "content",
    "1.3": EVAL_DIR / "sentiment",
}
SECTION_HEADERS = {
    "1.1": "OVERALL SUMMARY",
    "1.2": "OVERALL ACCURACY BY METHOD",
    "1.3": "OVERALL SUMMARY",
}


def _stem_parts(llm_csv: Path, base_dir: Path) -> tuple:
    rel    = llm_csv.relative_to(base_dir)
    subdir = Path(*rel.parts[:-1]) if len(rel.parts) > 1 else Path(".")
    stem   = llm_csv.stem
    stem   = re.sub(r"^task\d+_", "", stem)
    stem   = re.sub(r"_extended\d+", "", stem)
    stem   = re.sub(r"_generator", "", stem)
    stem   = re.sub(r"_n\d+$", "", stem)
    return subdir, stem


def extract_section(txt_path: Path, header: str) -> str:
    """Extract from the ─── separator before `header` to the next blank line."""
    if not txt_path.exists():
        return f"  [File not found: {txt_path.name}]"
    content = txt_path.read_text(encoding="utf-8")
    pattern = rf"(─{{40,}}\n{re.escape(header)}\n─{{40,}}.*?)(?=\n\n|\Z)"
    m = re.search(pattern, content, re.DOTALL)
    if not m:
        return f"  [Section '{header}' not found in {txt_path.name}]"
    return m.group(1)


# Step 1 – Run all subtask scripts
print("=" * 72)
print("Running subtask evaluations...")
print("=" * 72)

total_start = time.perf_counter()

for name, script in [
    ("1.1", EVAL_PY / "subtask1.1.py"),
    ("1.2", EVAL_PY / "subtask1.2.py"),
    ("1.3", EVAL_PY / "subtask1.3.py"),
]:
    print(f"\n  Running subtask{name}... ", end="", flush=True)
    t0 = time.perf_counter()
    proc = subprocess.run(
        [str(PYTHON), str(script)],
        cwd=str(TASK_DIR),
        capture_output=True,
        text=True,
        env={**__import__("os").environ, "PYTHONUTF8": "1"},
    )
    elapsed = time.perf_counter() - t0
    if proc.returncode != 0:
        print(f"FAILED ({elapsed:.1f}s)")
        print(proc.stderr[-500:])
        sys.exit(1)
    print(f"done ({elapsed:.1f}s)")

# Step 2 – Write one summary file per LLM output
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

llm_csvs    = sorted(p for p in LLM_DIR.rglob("*.csv") if p.stem.endswith("_n20"))
stem_pairs  = [_stem_parts(p, LLM_DIR) for p in llm_csvs]

saved = []
for llm_csv, (subdir, stem) in zip(llm_csvs, stem_pairs):
    lines: list = []
    L = lines.append

    L("=" * 72)
    L(f"SUMMARY REPORT  –  {subdir / stem}")
    L("=" * 72)
    L(f"\nGenerated : {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    for task, label in [
        ("1.1", "Topic Classification"),
        ("1.2", "Content / Text Span Matching"),
        ("1.3", "Sentiment Classification"),
    ]:
        result_file = RESULT_DIRS[task] / subdir / f"{stem}_results.txt"
        section     = extract_section(result_file, SECTION_HEADERS[task])
        L(f"\n[Subtask {task} – {label}]")
        L(section)

    out_dir  = SUMMARY_DIR / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{stem}_summary.txt"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    saved.append(out_path)
    print(f"  Saved: {out_path.relative_to(TASK_DIR)}")

total_elapsed = time.perf_counter() - total_start
print(f"\n{len(saved)} summary files written to: {SUMMARY_DIR.relative_to(TASK_DIR)}/")
print(f"Total runtime: {total_elapsed:.1f}s")
