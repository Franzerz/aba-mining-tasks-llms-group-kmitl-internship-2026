"""Task 2 evaluator v2 — Triple-Gate head alignment + prefix-restricted
semantic token matching for body/cont literals.

Differences from the original evaluator (task2_eval.py), which this file
takes only as a reference for GT loading / output-CSV shape:

  * Parse-error rows are never filtered out of the dataset — they are only
    *reported* (row-based, not id-based) as an error-rate summary.
  * The Head field is not scored by similarity. It instead gates whether an
    LLM instance may be evaluated at all, via three sequential checks
    ("Triple-Gate"): Review ID match -> Head/claim+sentiment match ->
    same-underlying-content match.
  * Body ("Literal" rows) are scored with a prefix-restricted, greedy
    one-to-one token matching pipeline (TP/FP/FN as pair counts, not
    independent pred/gt ratios), reported both including and excluding
    rows that came from a JSON-parse error.

Run (from repo root, ABA_mining/):
    python evaluator/task2_eval_new.py
    python evaluator/task2_eval_new.py --glob "gt/llama3.2/version1/*.csv"
    python evaluator/task2_eval_new.py --limit 60
"""

import argparse
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ----------------------------------------------------------------------
# Paths / constants
# ----------------------------------------------------------------------

TASK_DIR = Path(__file__).resolve().parent.parent
GT_CSV = TASK_DIR / "Dataset" / (
    "Original ABA Dataset for Version 2 "
    "[Oct 23, 2025], Senior Project, MUICT - Sheet2_.csv"
)
LLM_DIR = TASK_DIR / "outputs" / "task2"
EVAL_DIR = TASK_DIR / "outputs" / "eval" / "task2"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

N_BODY_COLS = 15
NO_LITERAL_PLACEHOLDER = "(no literals)"

TOKEN_SIM_THRESHOLD = 0.50    # body/cont literal-token cosine acceptance gate
CONTENT_SIM_THRESHOLD = 0.75  # Gate 3 fallback: "same underlying content"

PREFIX_NO_EVIDENT_NOT = "no_evident_not_"
PREFIX_HAVE_EVIDENT = "have_evident_"
PREFIX_TYPES = ("raw", "no_evident_not", "have_evident")

NEEDED_COLS = {
    "Review ID", "Topic", "Sentiment", "Head",
    "Selected Content", "Literal", "Literal Type", "Valid", "Errors",
}


# ----------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------

_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        print("  [loading SentenceTransformer all-MiniLM-L6-v2 ...]")
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def normalize(text) -> str:
    if pd.isna(text):
        return ""
    return re.sub(r"\s+", " ", str(text).strip().lower())


def norm_id(value) -> str:
    s = str(value).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s


def literal_prefix_type(literal: str) -> str:
    """Which prefix bucket a body/cont literal belongs to (Prefix-Type Restriction)."""
    if literal.startswith(PREFIX_NO_EVIDENT_NOT):
        return "no_evident_not"
    if literal.startswith(PREFIX_HAVE_EVIDENT):
        return "have_evident"
    return "raw"


def normalize_literal_text(literal: str) -> str:
    """Token Normalization step: underscores -> spaces (whole literal phrase)."""
    return literal.replace("_", " ").strip().lower()


def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_n = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    b_n = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return a_n @ b_n.T


def is_error_row(valid_val, errors_val) -> bool:
    has_error_text = pd.notna(errors_val) and str(errors_val).strip() not in ("", "nan")
    not_valid = str(valid_val).strip().lower() != "true"
    return has_error_text or not_valid


# ----------------------------------------------------------------------
# Ground truth loading
# ----------------------------------------------------------------------

def load_ground_truth() -> list:
    if not GT_CSV.exists():
        sys.exit(f"[ERROR] Ground truth CSV not found:\n  {GT_CSV}")

    df = pd.read_csv(GT_CSV)
    df = df.rename(columns={"ID": "Review ID", "Pos/Neg": "Sentiment"})
    df = df[
        df["Topic"].notna() &
        (df["Topic"].str.strip() != "") &
        (df["Topic"].str.strip() != "Off") &
        df["Selected Content"].notna() &
        df["Sentiment"].notna()
    ].copy()
    df["Topic"] = df["Topic"].str.strip()
    df["Sentiment"] = df["Sentiment"].str.strip()
    df = df.reset_index(drop=True)

    body_cols = [f"Body {i}" for i in range(1, N_BODY_COLS + 1) if f"Body {i}" in df.columns]
    cont_cols = [f"Cont. Body {i}" for i in range(1, N_BODY_COLS + 1) if f"Cont. Body {i}" in df.columns]

    records = []
    for _, row in df.iterrows():
        body = [str(row[c]).strip() for c in body_cols if pd.notna(row[c]) and str(row[c]).strip()]
        cont = [str(row[c]).strip() for c in cont_cols if pd.notna(row[c]) and str(row[c]).strip()]
        records.append({
            "review_id": norm_id(row["Review ID"]),
            "topic": row["Topic"],
            "sentiment": row["Sentiment"],
            "head": str(row["Head"]).strip() if pd.notna(row.get("Head", "")) else "",
            "content": str(row["Selected Content"]),
            "norm_content": normalize(row["Selected Content"]),
            "body": body,
            "cont": cont,
        })
    return records


GT_RECORDS = load_ground_truth()

GT_BY_REVIEW: dict = defaultdict(list)
GT_BY_REVIEW_HEAD: dict = defaultdict(list)
for _i, _r in enumerate(GT_RECORDS):
    GT_BY_REVIEW[_r["review_id"]].append(_i)
    if _r["head"]:
        GT_BY_REVIEW_HEAD[(_r["review_id"], _r["head"].lower())].append(_i)
GT_BY_REVIEW = dict(GT_BY_REVIEW)
GT_BY_REVIEW_HEAD = dict(GT_BY_REVIEW_HEAD)

print(f"Ground truth: {len(GT_RECORDS)} rows, "
      f"{len({r['review_id'] for r in GT_RECORDS})} unique Review IDs\n")


# ----------------------------------------------------------------------
# Triple-Gate matching
# ----------------------------------------------------------------------

def gate_check_pass1(rid: str, head_actual: str, norm_content: str) -> dict:
    """Cheap, no-embedding pass. Resolves gate1/gate2 always, and gate3
    immediately if an exact normalized-content match exists. Otherwise
    leaves gate3 pending (candidates recorded) for the fuzzy batch pass."""
    out = {
        "gate1": False, "gate2": False, "gate3": False,
        "gt_idx": None, "content_sim": None, "match_kind": None,
        "pending_candidates": None,
    }
    cand1 = GT_BY_REVIEW.get(rid)
    if not cand1:
        return out
    out["gate1"] = True

    cand2 = GT_BY_REVIEW_HEAD.get((rid, head_actual.lower()), [])
    if not cand2:
        return out
    out["gate2"] = True

    for idx in cand2:
        if GT_RECORDS[idx]["norm_content"] == norm_content:
            out["gate3"] = True
            out["gt_idx"] = idx
            out["content_sim"] = 1.0
            out["match_kind"] = "exact"
            return out

    out["pending_candidates"] = cand2
    return out


def resolve_pending_gate3(instances: list, model) -> None:
    """Batch-embed LLM content + all candidate GT contents for instances
    still needing a fuzzy Gate-3 decision, then fill in gate3/gt_idx."""
    pending = [i for i, inst in enumerate(instances) if inst["gate"]["pending_candidates"]]
    if not pending:
        return

    texts = []
    text_index: dict = {}

    def _idx_for(t: str) -> int:
        if t not in text_index:
            text_index[t] = len(texts)
            texts.append(t)
        return text_index[t]

    jobs = []  # (inst_i, llm_text_idx, [(cand_gt_idx, cand_text_idx), ...])
    for i in pending:
        inst = instances[i]
        llm_idx = _idx_for(inst["content"])
        cand_list = []
        for gidx in inst["gate"]["pending_candidates"]:
            cand_list.append((gidx, _idx_for(GT_RECORDS[gidx]["content"])))
        jobs.append((i, llm_idx, cand_list))

    embs = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

    for i, llm_idx, cand_list in jobs:
        q = embs[llm_idx:llm_idx + 1]
        cand_embs = np.array([embs[t] for _, t in cand_list])
        sims = cosine_matrix(q, cand_embs)[0]
        best = int(np.argmax(sims))
        best_gidx, _ = cand_list[best]
        best_sim = float(sims[best])
        gate = instances[i]["gate"]
        if best_sim >= CONTENT_SIM_THRESHOLD:
            gate["gate3"] = True
            gate["gt_idx"] = best_gidx
            gate["content_sim"] = best_sim
            gate["match_kind"] = "fuzzy"
        else:
            gate["gate3"] = False
            gate["content_sim"] = best_sim
            gate["match_kind"] = "fuzzy_below_threshold"


# ----------------------------------------------------------------------
# Body / cont token matching (prefix-restricted, greedy 1:1, threshold 0.50)
# ----------------------------------------------------------------------

def bucket_literals(literals: list) -> dict:
    buckets = {p: [] for p in PREFIX_TYPES}
    for lit in literals:
        buckets[literal_prefix_type(lit)].append(lit)
    return buckets


def match_instance_tokens(pred_lits: list, gt_lits: list, emb_of: dict) -> dict:
    """Greedy one-to-one matching within each prefix bucket.
    Returns per-bucket + total {tp, pb, rb, pairs:[(pred,gt,sim)]}."""
    pred_buckets = bucket_literals(pred_lits)
    gt_buckets = bucket_literals(gt_lits)

    out = {}
    total = {"tp": 0, "pb": 0, "rb": 0, "pairs": []}
    for ptype in PREFIX_TYPES:
        p_list = pred_buckets[ptype]
        g_list = gt_buckets[ptype]
        pb, rb = len(p_list), len(g_list)

        pairs_kept = []
        if p_list and g_list:
            P = np.array([emb_of[normalize_literal_text(x)] for x in p_list])
            G = np.array([emb_of[normalize_literal_text(x)] for x in g_list])
            sim = cosine_matrix(P, G)
            candidates = [
                (i, j, float(sim[i, j]))
                for i in range(pb) for j in range(rb)
                if sim[i, j] >= TOKEN_SIM_THRESHOLD
            ]
            candidates.sort(key=lambda t: -t[2])
            used_p, used_g = set(), set()
            for i, j, s in candidates:
                if i in used_p or j in used_g:
                    continue
                used_p.add(i); used_g.add(j)
                pairs_kept.append((p_list[i], g_list[j], s))

        tp = len(pairs_kept)
        out[ptype] = {"tp": tp, "pb": pb, "rb": rb, "pairs": pairs_kept}
        total["tp"] += tp; total["pb"] += pb; total["rb"] += rb
        total["pairs"].extend(pairs_kept)

    out["total"] = total
    return out


def prf1(tp: int, pb: int, rb: int) -> tuple:
    precision = tp / pb if pb else 0.0
    recall = tp / rb if rb else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


class Accumulator:
    def __init__(self):
        self.by_bucket = {p: {"tp": 0, "pb": 0, "rb": 0} for p in PREFIX_TYPES}
        self.total = {"tp": 0, "pb": 0, "rb": 0}

    def add(self, match_result: dict):
        for ptype in PREFIX_TYPES:
            b = match_result[ptype]
            acc = self.by_bucket[ptype]
            acc["tp"] += b["tp"]; acc["pb"] += b["pb"]; acc["rb"] += b["rb"]
        t = match_result["total"]
        self.total["tp"] += t["tp"]; self.total["pb"] += t["pb"]; self.total["rb"] += t["rb"]

    def merge(self, other: "Accumulator"):
        for ptype in PREFIX_TYPES:
            a, b = self.by_bucket[ptype], other.by_bucket[ptype]
            a["tp"] += b["tp"]; a["pb"] += b["pb"]; a["rb"] += b["rb"]
        self.total["tp"] += other.total["tp"]
        self.total["pb"] += other.total["pb"]
        self.total["rb"] += other.total["rb"]


# ----------------------------------------------------------------------
# Per-file evaluation
# ----------------------------------------------------------------------

def _stem_parts(llm_csv: Path, base_dir: Path) -> tuple:
    rel = llm_csv.relative_to(base_dir)
    subdir = Path(*rel.parts[:-1]) if len(rel.parts) > 1 else Path(".")
    stem = re.sub(r"_n\d+$", "", llm_csv.stem)
    return subdir, stem


def _write(llm_csv: Path, workings: list, results: list) -> None:
    subdir, stem = _stem_parts(llm_csv, LLM_DIR)
    out_dir = EVAL_DIR / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    r_path = out_dir / f"{stem}_results.txt"
    w_path = out_dir / f"{stem}_workings.txt"
    r_path.write_text("\n".join(results), encoding="utf-8")
    w_path.write_text("\n".join(workings), encoding="utf-8")
    print(f"  Saved: {r_path.relative_to(TASK_DIR)}")
    print(f"  Saved: {w_path.relative_to(TASK_DIR)}")


def evaluate(llm_csv: Path, limit: int = None) -> dict:
    t_start = time.perf_counter()
    model = _get_model()

    raw = pd.read_csv(llm_csv, dtype={"Review ID": str})
    if "Review ID" not in raw.columns and "ID" in raw.columns:
        raw = raw.rename(columns={"ID": "Review ID"})
    if not NEEDED_COLS.issubset(raw.columns):
        print(f"  [SKIP] missing columns {NEEDED_COLS - set(raw.columns)} - {llm_csv.name}")
        return None

    workings: list = []
    results: list = []
    W = workings.append
    R = results.append

    W("=" * 72)
    W(f"WORKINGS (v2) - {llm_csv.name}")
    W("=" * 72)
    W(f"Token similarity threshold (body/cont literals) : {TOKEN_SIM_THRESHOLD}")
    W(f"Gate-3 content similarity threshold (fuzzy fallback) : {CONTENT_SIM_THRESHOLD}")

    R("=" * 72)
    R(f"EVALUATION RESULTS (v2) - {llm_csv.name}")
    R("=" * 72)

    # ------------------------------------------------------------------
    # 1) Row-level error report (rows, not IDs) — informational only,
    #    nothing is filtered out of the dataset below.
    # ------------------------------------------------------------------
    total_rows = len(raw)
    error_mask = raw.apply(lambda r: is_error_row(r.get("Valid"), r.get("Errors")), axis=1)
    n_error_rows = int(error_mask.sum())
    n_remaining_rows = total_rows - n_error_rows
    pct_remaining = n_remaining_rows / total_rows if total_rows else 0.0

    error_msgs = Counter(
        str(e).strip() for e in raw.loc[error_mask, "Errors"]
        if pd.notna(e) and str(e).strip() not in ("", "nan")
    )

    R(f"\n{'-' * 72}")
    R("ROW-LEVEL DATA INTEGRITY (errors are reported, never filtered out)")
    R(f"{'-' * 72}")
    R(f"  Total CSV rows        : {total_rows}")
    R(f"  Error rows (Valid=False / Errors set) : {n_error_rows}")
    R(f"  Error rate             : {n_error_rows}/{total_rows} = {(n_error_rows / total_rows if total_rows else 0):.4f}")
    R(f"  Remaining (clean) rows: {n_remaining_rows}/{total_rows} = {pct_remaining:.4f}")
    if error_msgs:
        R("  Error message breakdown:")
        for msg, cnt in error_msgs.most_common(10):
            R(f"    {cnt:>5}  {msg}")

    W(f"\nTotal CSV rows: {total_rows}, error rows: {n_error_rows}, "
      f"remaining: {n_remaining_rows}/{total_rows} = {pct_remaining:.4f}")

    # ------------------------------------------------------------------
    # 2) Build per-instance records (Review ID, Topic, Sentiment, Content)
    # ------------------------------------------------------------------
    raw["Review ID"] = raw["Review ID"].apply(norm_id)
    raw["Topic"] = raw["Topic"].astype(str).str.strip()
    raw["Sentiment"] = raw["Sentiment"].astype(str).str.strip()
    raw["Selected Content"] = raw["Selected Content"].astype(str)
    raw["Head"] = raw["Head"].astype(str).str.strip()

    group_keys = ["Review ID", "Topic", "Sentiment", "Selected Content"]
    groups = list(raw.groupby(group_keys, sort=False))
    if limit is not None:
        groups = groups[:limit]

    instances = []
    for (rid, topic, sentiment, content), df in groups:
        lits = df[["Literal", "Literal Type"]].copy()
        lits = lits[lits["Literal"].astype(str).str.strip() != NO_LITERAL_PLACEHOLDER]
        body_lits = [str(x).strip() for x in lits.loc[lits["Literal Type"] == "body", "Literal"] if str(x).strip()]
        cont_lits = [str(x).strip() for x in lits.loc[lits["Literal Type"] == "cont", "Literal"] if str(x).strip()]
        is_valid = str(df["Valid"].iloc[0]).strip().lower() == "true"

        instances.append({
            "rid": rid, "topic": topic, "sentiment": sentiment, "content": content,
            "norm_content": normalize(content),
            "head_actual": str(df["Head"].iloc[0]).strip(),
            "valid": is_valid,
            "body_lits": body_lits, "cont_lits": cont_lits,
            "gate": None,
        })

    # ------------------------------------------------------------------
    # 3) Triple-Gate check for every instance
    # ------------------------------------------------------------------
    for inst in instances:
        inst["gate"] = gate_check_pass1(inst["rid"], inst["head_actual"], inst["norm_content"])
    resolve_pending_gate3(instances, model)

    n_total = len(instances)
    n_g1 = sum(1 for i in instances if i["gate"]["gate1"])
    n_g2 = sum(1 for i in instances if i["gate"]["gate2"])
    n_g3 = sum(1 for i in instances if i["gate"]["gate3"])

    R(f"\n{'-' * 72}")
    R("HEAD: TRIPLE-GATE MATCHING (prerequisite, not similarity-scored)")
    R(f"{'-' * 72}")
    R(f"  Instances (Review ID x Topic x Sentiment x Selected Content groups): {n_total}")
    R(f"  Gate 1 (Review ID match)             : {n_g1}/{n_total} = {(n_g1 / n_total if n_total else 0):.4f}")
    R(f"  Gate 2 (+ Head/claim+sentiment match) : {n_g2}/{n_total} = {(n_g2 / n_total if n_total else 0):.4f}")
    R(f"  Gate 3 (+ same-content match)         : {n_g3}/{n_total} = {(n_g3 / n_total if n_total else 0):.4f}")
    R(f"  => Triple-Gate pass rate (accuracy)   : {n_g3}/{n_total} = {(n_g3 / n_total if n_total else 0):.4f}")
    match_kinds = Counter(i["gate"]["match_kind"] for i in instances if i["gate"]["gate3"])
    R(f"  Gate-3 match kind breakdown : {dict(match_kinds)}")

    W(f"\n{'-' * 72}\nTRIPLE-GATE DETAIL PER INSTANCE\n{'-' * 72}")
    for inst in instances:
        g = inst["gate"]
        W(f"\nID={inst['rid']}  Topic={inst['topic']}  Sentiment={inst['sentiment']}  Valid={inst['valid']}")
        W(f"  Content   : {inst['content'][:140]}")
        W(f"  LLM Head  : {inst['head_actual']}")
        W(f"  Gate1(ID) : {'PASS' if g['gate1'] else 'FAIL'}")
        W(f"  Gate2(Head/claim+sentiment) : {'PASS' if g['gate2'] else 'FAIL'}")
        if g["content_sim"] is not None:
            W(f"  Gate3(content) : {'PASS' if g['gate3'] else 'FAIL'}  "
              f"kind={g['match_kind']} sim={g['content_sim']:.4f}")
        else:
            W(f"  Gate3(content) : FAIL  (no candidate reached this gate)")

    # ------------------------------------------------------------------
    # 4) Body/cont literal matching for every Gate-3-aligned instance
    # ------------------------------------------------------------------
    aligned = [i for i in instances if i["gate"]["gate3"]]

    vocab = set()
    for inst in aligned:
        for lit in inst["body_lits"] + inst["cont_lits"]:
            vocab.add(normalize_literal_text(lit))
        gt_rec = GT_RECORDS[inst["gate"]["gt_idx"]]
        for lit in gt_rec["body"] + gt_rec["cont"]:
            vocab.add(normalize_literal_text(lit))
    vocab = sorted(vocab)

    emb_of = {}
    if vocab:
        embs = model.encode(vocab, convert_to_numpy=True, show_progress_bar=False)
        emb_of = dict(zip(vocab, embs))

    acc_incl = Accumulator()   # includes error (invalid) instances
    acc_excl = Accumulator()   # excludes error (invalid) instances

    W(f"\n{'-' * 72}\nBODY/CONT TOKEN MATCHING (prefix-restricted, greedy 1:1, "
      f"threshold {TOKEN_SIM_THRESHOLD})\n{'-' * 72}")

    for inst in aligned:
        gt_rec = GT_RECORDS[inst["gate"]["gt_idx"]]
        pred_lits = inst["body_lits"] + inst["cont_lits"]
        gt_lits = gt_rec["body"] + gt_rec["cont"]
        m = match_instance_tokens(pred_lits, gt_lits, emb_of)

        acc_incl.add(m)
        if inst["valid"]:
            acc_excl.add(m)

        t = m["total"]
        p_, r_, f_ = prf1(t["tp"], t["pb"], t["rb"])
        W(f"\nID={inst['rid']}  Topic={inst['topic']}  Sentiment={inst['sentiment']}  Valid={inst['valid']}")
        W(f"  Pred literals : {pred_lits}")
        W(f"  GT   literals : {gt_lits}")
        for ptype in PREFIX_TYPES:
            b = m[ptype]
            if b["pb"] or b["rb"]:
                W(f"  [{ptype}] pred={b['pb']} gt={b['rb']} matched={b['tp']} "
                  f"pairs={[(p, g, round(s, 3)) for p, g, s in b['pairs']]}")
        W(f"  => instance TP={t['tp']} FP={t['pb'] - t['tp']} FN={t['rb'] - t['tp']}  "
          f"P={p_:.4f} R={r_:.4f} F1={f_:.4f}")

    n_not_aligned = n_total - len(aligned)
    n_aligned_valid = sum(1 for i in aligned if i["valid"])
    n_aligned_error = len(aligned) - n_aligned_valid

    R(f"\n{'-' * 72}")
    R("BODY (LITERAL) TOKEN MATCHING")
    R(f"{'-' * 72}")
    R("  Prefix-Type Restriction: raw <-> raw, no_evident_not_ <-> no_evident_not_, "
      "have_evident_ <-> have_evident_ only.")
    R("  TP = greedy 1:1 matched pairs (sim >= threshold, same prefix bucket).")
    R("  FP = Pb - TP (unmatched predicted literals).  FN = Rb - TP (unmatched GT literals).")
    R("  TN = not applicable / not computed for this task.")
    R(f"  Instances passing Triple-Gate (evaluable) : {len(aligned)}/{n_total} "
      f"(not aligned / excluded from body scoring: {n_not_aligned})")
    R(f"    of which Valid (no parse error) : {n_aligned_valid}")
    R(f"    of which Error (parse error, 0 predicted literals) : {n_aligned_error}")

    def report_scenario(label: str, acc: Accumulator, n_instances: int):
        R(f"\n  --- {label} (n={n_instances} aligned instances) ---")
        t = acc.total
        p_, r_, f_ = prf1(t["tp"], t["pb"], t["rb"])
        fp, fn = t["pb"] - t["tp"], t["rb"] - t["tp"]
        R(f"    TP={t['tp']} FP={fp} FN={fn}")
        R(f"    Precision : {p_:.4f}  ({t['tp']}/{t['pb']})")
        R(f"    Recall    : {r_:.4f}  ({t['tp']}/{t['rb']})")
        R(f"    F1        : {f_:.4f}")
        R(f"    Micro TP={t['tp']} FP={fp} FN={fn}")
        R(f"    Micro Precision : {p_:.4f}  ({t['tp']}/{t['pb']})")
        R(f"    Micro Recall    : {r_:.4f}  ({t['tp']}/{t['rb']})")
        R(f"    Micro F1        : {f_:.4f}")
        R(f"\n    Breakdown by prefix type:")
        R(f"    {'Prefix type':<16} {'TP':>6} {'FP':>6} {'FN':>6} {'Prec':>8} {'Rec':>8} {'F1':>8}")
        for ptype in PREFIX_TYPES:
            b = acc.by_bucket[ptype]
            pp, rr, ff = prf1(b["tp"], b["pb"], b["rb"])
            R(f"    {ptype:<16} {b['tp']:>6} {b['pb'] - b['tp']:>6} {b['rb'] - b['tp']:>6} "
              f"{pp:>8.4f} {rr:>8.4f} {ff:>8.4f}")

    report_scenario("INCLUDING error instances (Gate-3-aligned, all Valid states)",
                     acc_incl, len(aligned))
    report_scenario("EXCLUDING error instances (Gate-3-aligned, Valid=True only)",
                     acc_excl, n_aligned_valid)

    elapsed = time.perf_counter() - t_start
    R(f"\nRuntime: {elapsed:.2f}s")

    _write(llm_csv, workings, results)
    print("\n".join(results))

    return {
        "file": str(llm_csv.relative_to(TASK_DIR)),
        "total_rows": total_rows,
        "error_rows": n_error_rows,
        "n_total": n_total,
        "n_gate1": n_g1, "n_gate2": n_g2, "n_gate3": n_g3,
        "n_aligned_valid": n_aligned_valid, "n_aligned_error": n_aligned_error,
        "acc_incl": acc_incl, "acc_excl": acc_excl,
    }


def write_overall_summary(file_summaries: list) -> None:
    """Aggregate every processed file (every model / version / run) into a
    single grand-total report: one true overall Precision/Recall/F1, plus a
    per-file leaderboard so all versions and runs are visible at a glance."""
    if not file_summaries:
        return

    results: list = []
    R = results.append

    total_rows = sum(f["total_rows"] for f in file_summaries)
    error_rows = sum(f["error_rows"] for f in file_summaries)
    n_total = sum(f["n_total"] for f in file_summaries)
    n_g1 = sum(f["n_gate1"] for f in file_summaries)
    n_g2 = sum(f["n_gate2"] for f in file_summaries)
    n_g3 = sum(f["n_gate3"] for f in file_summaries)
    n_aligned_valid = sum(f["n_aligned_valid"] for f in file_summaries)
    n_aligned_error = sum(f["n_aligned_error"] for f in file_summaries)

    acc_incl_global = Accumulator()
    acc_excl_global = Accumulator()
    for f in file_summaries:
        acc_incl_global.merge(f["acc_incl"])
        acc_excl_global.merge(f["acc_excl"])

    R("=" * 72)
    R(f"OVERALL SUMMARY - {len(file_summaries)} file(s) "
      "(all models / versions / runs combined)")
    R("=" * 72)

    R(f"\n{'-' * 72}")
    R("ROW-LEVEL DATA INTEGRITY (combined across every file)")
    R(f"{'-' * 72}")
    R(f"  Total CSV rows        : {total_rows}")
    R(f"  Error rows            : {error_rows}")
    R(f"  Error rate             : {error_rows}/{total_rows} = "
      f"{(error_rows / total_rows if total_rows else 0):.4f}")
    R(f"  Remaining (clean) rows: {total_rows - error_rows}/{total_rows} = "
      f"{(total_rows - error_rows) / total_rows if total_rows else 0:.4f}")

    R(f"\n{'-' * 72}")
    R("HEAD: TRIPLE-GATE MATCHING (combined across every file)")
    R(f"{'-' * 72}")
    R(f"  Instances total : {n_total}")
    R(f"  Gate 1 (Review ID match)              : {n_g1}/{n_total} = {(n_g1 / n_total if n_total else 0):.4f}")
    R(f"  Gate 2 (+ Head/claim+sentiment match)  : {n_g2}/{n_total} = {(n_g2 / n_total if n_total else 0):.4f}")
    R(f"  Gate 3 (+ same-content match)          : {n_g3}/{n_total} = {(n_g3 / n_total if n_total else 0):.4f}")
    R(f"  => Triple-Gate pass rate (accuracy)    : {n_g3}/{n_total} = {(n_g3 / n_total if n_total else 0):.4f}")

    R(f"\n{'-' * 72}")
    R("BODY (LITERAL) TOKEN MATCHING - OVERALL (combined across every file)")
    R(f"{'-' * 72}")
    R(f"  Gate-3-aligned instances : {n_aligned_valid + n_aligned_error}/{n_total}  "
      f"(Valid: {n_aligned_valid}, Error: {n_aligned_error})")

    def report_scenario(label: str, acc: Accumulator):
        R(f"\n  --- {label} ---")
        t = acc.total
        p_, r_, f_ = prf1(t["tp"], t["pb"], t["rb"])
        fp, fn = t["pb"] - t["tp"], t["rb"] - t["tp"]
        R(f"    TP={t['tp']} FP={fp} FN={fn}")
        R(f"    Precision : {p_:.4f}  ({t['tp']}/{t['pb']})")
        R(f"    Recall    : {r_:.4f}  ({t['tp']}/{t['rb']})")
        R(f"    F1        : {f_:.4f}")
        R(f"    Micro TP={t['tp']} FP={fp} FN={fn}")
        R(f"    Micro Precision : {p_:.4f}  ({t['tp']}/{t['pb']})")
        R(f"    Micro Recall    : {r_:.4f}  ({t['tp']}/{t['rb']})")
        R(f"    Micro F1        : {f_:.4f}")
        R(f"\n    Breakdown by prefix type:")
        R(f"    {'Prefix type':<16} {'TP':>6} {'FP':>6} {'FN':>6} {'Prec':>8} {'Rec':>8} {'F1':>8}")
        for ptype in PREFIX_TYPES:
            b = acc.by_bucket[ptype]
            pp, rr, ff = prf1(b["tp"], b["pb"], b["rb"])
            R(f"    {ptype:<16} {b['tp']:>6} {b['pb'] - b['tp']:>6} {b['rb'] - b['tp']:>6} "
              f"{pp:>8.4f} {rr:>8.4f} {ff:>8.4f}")

    report_scenario("INCLUDING error instances (every Gate-3-aligned instance)", acc_incl_global)
    report_scenario("EXCLUDING error instances (Valid=True only)", acc_excl_global)

    R(f"\n{'-' * 72}")
    R("PER-FILE LEADERBOARD (Gate pass rate, error rate, body F1 excluding errors)")
    R(f"{'-' * 72}")
    R(f"  {'File':<90} {'GateOK':>10} {'ErrRow%':>10} {'F1(excl-err)':>13}")
    for f in sorted(file_summaries, key=lambda x: x["file"]):
        gate_pct = f["n_gate3"] / f["n_total"] if f["n_total"] else 0.0
        err_pct = f["error_rows"] / f["total_rows"] if f["total_rows"] else 0.0
        t = f["acc_excl"].total
        _, _, f1_excl = prf1(t["tp"], t["pb"], t["rb"])
        R(f"  {f['file']:<90} {gate_pct:>10.4%} {err_pct:>10.4%} {f1_excl:>13.4f}")

    out_path = EVAL_DIR / "_OVERALL_results.txt"
    out_path.write_text("\n".join(results), encoding="utf-8")
    print(f"\n  Saved: {out_path.relative_to(TASK_DIR)}")
    print("\n".join(results))


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Task 2 evaluator v2 (Triple-Gate head alignment + prefix-restricted body token matching)"
    )
    parser.add_argument("--limit", type=int, default=None,
                         help="Evaluate only the first N instances per file (quick smoke tests)")
    parser.add_argument("--glob", default="*.csv",
                         help="Glob pattern (relative to outputs/task2) selecting which files to evaluate")
    args = parser.parse_args()

    llm_csvs = sorted(
        p for p in LLM_DIR.rglob(args.glob)
        if "old_output" not in p.parts and p.suffix == ".csv"
    )
    if not llm_csvs:
        sys.exit(f"[ERROR] No CSV files found under {LLM_DIR}")

    print(f"Found {len(llm_csvs)} Task 2 output file(s):\n")
    for p in llm_csvs:
        print(f"  {p.relative_to(TASK_DIR)}")
    print()

    total_start = time.perf_counter()
    file_summaries = []
    for llm_csv in llm_csvs:
        print(f"\n{'#' * 72}")
        print(f"Processing: {llm_csv.relative_to(TASK_DIR)}")
        print(f"{'#' * 72}")
        summary = evaluate(llm_csv, limit=args.limit)
        if summary is not None:
            file_summaries.append(summary)

    print(f"\n\n{'#' * 72}")
    print("Building overall summary across all files")
    print(f"{'#' * 72}")
    write_overall_summary(file_summaries)

    total_elapsed = time.perf_counter() - total_start
    print(f"\n\nAll done. Results in: {EVAL_DIR.relative_to(TASK_DIR)}/")
    print(f"Total runtime: {total_elapsed:.2f}s")
