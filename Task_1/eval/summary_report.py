import re
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
TASK_DIR = Path(__file__).resolve().parent.parent
LLM_DIR  = TASK_DIR / "outputs" / "task1"
EVAL_DIR    = TASK_DIR / "outputs" / "eval"
SUMMARY_DIR = EVAL_DIR / "summary"
PYTHON   = Path(__file__).resolve().parents[2] / ".venv" / "bin" / "python"
EVAL_PY  = Path(__file__).resolve().parent

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _short_stem(llm_csv: Path) -> str:
    s = llm_csv.stem
    s = re.sub(r"^task\d+_", "", s)
    s = re.sub(r"_extended\d+", "", s)
    s = re.sub(r"_generator", "", s)
    s = re.sub(r"_n\d+$", "", s)
    return s


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


# ---------------------------------------------------------------------------
# Step 1 – Run all subtask scripts
# ---------------------------------------------------------------------------
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
    )
    elapsed = time.perf_counter() - t0
    if proc.returncode != 0:
        print(f"FAILED ({elapsed:.1f}s)")
        print(proc.stderr[-500:])
        sys.exit(1)
    print(f"done ({elapsed:.1f}s)")

# ---------------------------------------------------------------------------
# Step 2 – Write one summary file per LLM output
# ---------------------------------------------------------------------------
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

llm_csvs = sorted(LLM_DIR.rglob("*.csv"))
stems    = [_short_stem(p) for p in llm_csvs]

saved = []
for stem in stems:
    lines = []
    L = lines.append

    L("=" * 72)
    L(f"SUMMARY REPORT  –  {stem}")
    L("=" * 72)
    L(f"\nGenerated : {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    for task, label in [
        ("1.1", "Topic Classification"),
        ("1.2", "Content Matching"),
        ("1.3", "Sentiment Classification"),
    ]:
        result_file = RESULT_DIRS[task] / f"{stem}_results.txt"
        section     = extract_section(result_file, SECTION_HEADERS[task])
        L(f"\n[Subtask {task} – {label}]")
        L(section)

    out_path = SUMMARY_DIR / f"{stem}_summary.txt"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    saved.append(out_path)
    print(f"  Saved: {out_path.relative_to(TASK_DIR)}")

total_elapsed = time.perf_counter() - total_start
print(f"\n{len(saved)} summary files written to: {SUMMARY_DIR.relative_to(TASK_DIR)}/")
print(f"Total runtime: {total_elapsed:.1f}s")
