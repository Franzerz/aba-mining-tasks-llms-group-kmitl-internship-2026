"""Task 2 version-level averages — combines the run1/run2/run3 result files
that task2_eval_new.py already produced for each model/version into a single
per-version average report, saved into that same version folder.

For each scenario reported per run ("INCLUDING error instances" and
"EXCLUDING error instances"), two kinds of averages are computed across the
runs found in a version folder (normally run1, run2, run3):

  * Micro average = pool the runs' TP/FP/FN counts first, then compute one
    Precision/Recall/F1 from the pooled counts. E.g. for precision:
        micro avg precision = (TP_run1 + TP_run2 + TP_run3)
                             / (Pb_run1 + Pb_run2 + Pb_run3)
    where Pb = TP + FP (predicted count) for that run.

  * Macro average = compute each run's own macro (per-topic) Precision /
    Recall / F1 independently (already present in each run's results.txt),
    then average those three numbers directly:
        macro avg precision = (macro_P_run1 + macro_P_run2 + macro_P_run3) / 3

Nothing is re-evaluated against the ground truth here — this only parses the
numbers already written by task2_eval_new.py's per-run *_results.txt files.

Run (from repo root, ABA_mining/):
    python evaluator/task2_version_avg.py
"""

import re
import sys
from pathlib import Path

TASK_DIR = Path(__file__).resolve().parent.parent
EVAL_DIR = TASK_DIR / "outputs" / "eval" / "task2"

RUN_FILE_RE = re.compile(r"^(?P<prefix>.+)_run(?P<run>\d+)(?P<suffix>.*)_results\.txt$")

SCENARIOS = [
    ("INCLUDING error instances", "INCLUDING error instances"),
    ("EXCLUDING error instances", "EXCLUDING error instances"),
]

MICRO_RE = re.compile(r"Micro TP=(\d+)\s+FP=(\d+)\s+FN=(\d+)")
MACRO_P_RE = re.compile(r"Macro Precision \(per-topic\)\s*:\s*([\d.]+)")
MACRO_R_RE = re.compile(r"Macro Recall\s+\(per-topic\)\s*:\s*([\d.]+)")
MACRO_F_RE = re.compile(r"Macro F1\s+\(per-topic\)\s*:\s*([\d.]+)")


def prf1(tp: int, pb: int, rb: int) -> tuple:
    precision = tp / pb if pb else 0.0
    recall = tp / rb if rb else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def parse_scenario(block: str) -> dict:
    m = MICRO_RE.search(block)
    tp, fp, fn = (int(x) for x in m.groups()) if m else (0, 0, 0)
    pb, rb = tp + fp, tp + fn
    mp = float(MACRO_P_RE.search(block).group(1)) if MACRO_P_RE.search(block) else 0.0
    mr = float(MACRO_R_RE.search(block).group(1)) if MACRO_R_RE.search(block) else 0.0
    mf = float(MACRO_F_RE.search(block).group(1)) if MACRO_F_RE.search(block) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "pb": pb, "rb": rb,
            "macro_p": mp, "macro_r": mr, "macro_f": mf}


def parse_run_results(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    incl_start = text.index("INCLUDING error instances")
    excl_start = text.index("EXCLUDING error instances")
    runtime_idx = text.find("\nRuntime:")
    end = runtime_idx if runtime_idx != -1 else len(text)
    return {
        "INCLUDING error instances": parse_scenario(text[incl_start:excl_start]),
        "EXCLUDING error instances": parse_scenario(text[excl_start:end]),
    }


def find_version_dirs() -> list:
    """Every directory under EVAL_DIR that directly contains run result
    files (model/version leaf directories), discovered generically so newly
    added models/versions are picked up automatically."""
    dirs = {}
    for p in EVAL_DIR.rglob("*_results.txt"):
        if RUN_FILE_RE.match(p.name):
            dirs.setdefault(p.parent, []).append(p)
    return sorted(dirs.items(), key=lambda kv: str(kv[0]))


def write_version_avg(version_dir: Path, run_files: list) -> None:
    runs = []
    for p in sorted(run_files, key=lambda p: int(RUN_FILE_RE.match(p.name).group("run"))):
        m = RUN_FILE_RE.match(p.name)
        runs.append({"run": int(m.group("run")), "path": p, "name": p.name,
                      "data": parse_run_results(p)})
    if len(runs) < 2:
        return  # nothing to average

    m0 = RUN_FILE_RE.match(runs[0]["name"])
    out_name = f"{m0.group('prefix')}_avg{m0.group('suffix')}_results.txt"
    out_path = version_dir / out_name

    model = version_dir.parent.name
    version = version_dir.name

    R: list = []
    R.append("=" * 72)
    R.append(f"VERSION AVERAGE (v2) - {model} / {version} "
              f"(n={len(runs)} runs: {', '.join('run' + str(r['run']) for r in runs)})")
    R.append("=" * 72)
    R.append("\nCombines these per-run result files:")
    for r in runs:
        R.append(f"    {r['name']}")

    R.append(f"\n{'-' * 72}")
    R.append("BODY (LITERAL) TOKEN MATCHING - VERSION AVERAGE ACROSS RUNS")
    R.append(f"{'-' * 72}")
    R.append("  Micro avg = pool TP/FP/FN across all runs, then compute P/R/F1 once.")
    R.append("  Macro avg = mean of each run's own macro (per-topic) P/R/F1.")

    for label, key in SCENARIOS:
        R.append(f"\n  --- {label} ---")

        R.append(f"    Per-run Micro Precision : " +
                  "  ".join(f"run{r['run']}={prf1(r['data'][key]['tp'], r['data'][key]['pb'], r['data'][key]['rb'])[0]:.4f}" for r in runs))
        R.append(f"    Per-run Micro Recall    : " +
                  "  ".join(f"run{r['run']}={prf1(r['data'][key]['tp'], r['data'][key]['pb'], r['data'][key]['rb'])[1]:.4f}" for r in runs))
        R.append(f"    Per-run Micro F1        : " +
                  "  ".join(f"run{r['run']}={prf1(r['data'][key]['tp'], r['data'][key]['pb'], r['data'][key]['rb'])[2]:.4f}" for r in runs))
        R.append(f"    Per-run Macro Precision : " +
                  "  ".join(f"run{r['run']}={r['data'][key]['macro_p']:.4f}" for r in runs))
        R.append(f"    Per-run Macro Recall    : " +
                  "  ".join(f"run{r['run']}={r['data'][key]['macro_r']:.4f}" for r in runs))
        R.append(f"    Per-run Macro F1        : " +
                  "  ".join(f"run{r['run']}={r['data'][key]['macro_f']:.4f}" for r in runs))

        sum_tp = sum(r["data"][key]["tp"] for r in runs)
        sum_pb = sum(r["data"][key]["pb"] for r in runs)
        sum_rb = sum(r["data"][key]["rb"] for r in runs)
        micro_p, micro_r, micro_f = prf1(sum_tp, sum_pb, sum_rb)

        n = len(runs)
        macro_p_avg = sum(r["data"][key]["macro_p"] for r in runs) / n
        macro_r_avg = sum(r["data"][key]["macro_r"] for r in runs) / n
        macro_f_avg = sum(r["data"][key]["macro_f"] for r in runs) / n

        R.append("")
        R.append(f"    Micro avg Precision : {micro_p:.4f}  ({sum_tp}/{sum_pb})")
        R.append(f"    Micro avg Recall    : {micro_r:.4f}  ({sum_tp}/{sum_rb})")
        R.append(f"    Micro avg F1        : {micro_f:.4f}")
        R.append(f"    Macro avg Precision : {macro_p_avg:.4f}  (mean over {n} runs)")
        R.append(f"    Macro avg Recall    : {macro_r_avg:.4f}  (mean over {n} runs)")
        R.append(f"    Macro avg F1        : {macro_f_avg:.4f}  (mean over {n} runs)")

    out_path.write_text("\n".join(R), encoding="utf-8")
    print(f"  Saved: {out_path.relative_to(TASK_DIR)}")


if __name__ == "__main__":
    version_dirs = find_version_dirs()
    if not version_dirs:
        sys.exit(f"[ERROR] No per-run result files found under {EVAL_DIR}")

    print(f"Found {len(version_dirs)} version folder(s) under {EVAL_DIR.relative_to(TASK_DIR)}\n")
    for version_dir, run_files in version_dirs:
        print(f"Processing: {version_dir.relative_to(TASK_DIR)}  ({len(run_files)} run file(s))")
        write_version_avg(version_dir, run_files)

    print("\nAll done.")
