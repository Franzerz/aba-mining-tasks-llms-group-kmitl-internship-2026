#### RATCHANON ####

import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Paths
TASK_DIR = Path(__file__).resolve().parent.parent
GT_CSV = TASK_DIR / "Dataset" / (
    "Original ABA Dataset for Version 2 "
    "[Oct 23, 2025], Senior Project, MUICT - Sheet2_.csv"
)
LLM_DIR = TASK_DIR / "outputs" / "task2"
EVAL_DIR = TASK_DIR / "outputs" / "eval" / "task2"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

SIM_THRESHOLD = 0.5
N_BODY_COLS = 15
NO_LITERAL_PLACEHOLDER = "(no literals)"

# Rule-compliance vocab tables (rule numbers refer to the 19-rule spec)
INTENSIFIERS = {"very", "extremely", "really", "quite", "rather", "so", "such"}
ARTICLES = {"a", "an", "the"}
SCAFFOLD_WORDS = {"no", "not", "evident", "have", "could", "be", "more"}
FACILITY_SYNONYMS = {"apartment", "apartments", "accommodation", "accommodations",
                      "place", "building", "atmosphere", "property"}
STAFF_SYNONYMS = {"people", "receptionist", "housekeeper", "owner", "host",
                   "gentleman", "lady"}
LOCATION_INSIDE_NOISE = {"wall", "walls", "ac"}

MANUAL_ONLY_RULES = (
    "Rules requiring semantic/human judgement and NOT auto-checked: "
    "R2 (safe plural removal), R3 (-ed ending necessity), R10 (typo correction), "
    "R11 (sentiment-matched extraction), R12/R13 (double-meaning splitting), "
    "R14 (implied-noun phrasing), R15 (indirect-meaning rewrites), "
    "R18 (active voice), R19 (multi-meaning extraction)."
)

RULE_SUBTASKS_NOTE = (
    "Task 2 evaluator processes every CSV under outputs/task2/ "
    "(currently only the 'gt' source is produced by run_task2.py)."
)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

_sent_model = None


def _get_sent_model():
    global _sent_model
    if _sent_model is None:
        from sentence_transformers import SentenceTransformer
        print("  [Loading sentence model all-MiniLM-L6-v2 …]")
        _sent_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _sent_model


def normalize(text) -> str:
    if pd.isna(text):
        return ""
    return re.sub(r"\s+", " ", str(text).strip().lower())


def norm_id(value) -> str:
    s = str(value).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s


def make_head(topic: str, sentiment: str) -> str:
    """Naive code-rule head (mirrors src/task2.py make_head)."""
    polarity = "good" if sentiment.strip().lower() == "positive" else "bad"
    t = topic.strip().lower().replace(" ", "_").replace("-", "_")
    return f"{polarity}_{t}"


def cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity between two equally-shaped embedding matrices."""
    num = (a * b).sum(axis=1)
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-9
    return num / denom


# ----------------------------------------------------------------------
# Rule-compliance checker (static, per literal)
# ----------------------------------------------------------------------

def check_literal_rules(literal: str, topic: str, original_text: str) -> list:
    """Return a list of violated-rule descriptions for one body literal.
    Empty list = literal passes every mechanically-checkable rule."""
    violations = []

    if literal != literal.strip():
        violations.append("format: leading/trailing whitespace")
    if " " in literal:
        violations.append("R7: contains spaces (should use underscores)")
    if re.search(r"[.!?,]", literal):
        violations.append("R4: contains punctuation")
    if literal != literal.lower():
        violations.append("R6: not lowercase")

    tokens = [t for t in literal.lower().split("_") if t]
    if not tokens:
        violations.append("empty literal")
        return violations

    bad_intensifiers = INTENSIFIERS & set(tokens)
    if bad_intensifiers:
        violations.append(f"R1: contains intensifier(s) {sorted(bad_intensifiers)}")

    bad_articles = ARTICLES & set(tokens)
    if bad_articles:
        violations.append(f"R5: contains article(s) {sorted(bad_articles)}")

    topic_key = topic.strip()
    if topic_key == "Facility":
        bad = FACILITY_SYNONYMS & set(tokens)
        if bad:
            violations.append(f"R16: Facility literal should use '_hotel' suffix, found {sorted(bad)}")
    elif topic_key == "Staff":
        bad = STAFF_SYNONYMS & set(tokens)
        if bad:
            violations.append(f"R16: Staff literal should use '_staff' suffix, found {sorted(bad)}")
    elif topic_key == "Location":
        bad = LOCATION_INSIDE_NOISE & set(tokens)
        if bad:
            violations.append(f"R16: indoor-noise word(s) {sorted(bad)} belong to Room topic, not Location")
    elif topic_key == "Room" and "room" not in tokens and "no" not in tokens and "not" not in tokens:
        violations.append("R16(soft): Room-topic literal missing '_room' suffix")

    orig_tokens = set(re.findall(r"[a-zA-Z']+", original_text.lower()))
    orig_singulars = {t[:-1] for t in orig_tokens if t.endswith("s") and len(t) > 3}
    orig_all = orig_tokens | orig_singulars

    allowed_extra = set(SCAFFOLD_WORDS)
    if topic_key == "Facility":
        allowed_extra.add("hotel")
    elif topic_key == "Staff":
        allowed_extra.add("staff")
    elif topic_key == "Room":
        allowed_extra.add("room")
    elif topic_key == "Location":
        allowed_extra.add("location")

    unknown = []
    for t in tokens:
        t_sing = t[:-1] if t.endswith("s") and len(t) > 3 else t
        if t in allowed_extra or t_sing in allowed_extra:
            continue
        if t in orig_all or t_sing in orig_all:
            continue
        unknown.append(t)
    if unknown:
        violations.append(f"R8: word(s) not found in original text {unknown} (possible synonym/hallucination)")

    if len(tokens) > 6:
        violations.append(f"R9: phrase is long ({len(tokens)} words) — consider shortening")

    topic_first_word = topic_key.lower().replace("-", "_").split("_")[0]
    if len(tokens) == 1 and tokens[0] == topic_first_word:
        violations.append("R17: literal duplicates the topic/head — consider Contrary instead")

    return violations


# ----------------------------------------------------------------------
# Ground truth loading
# ----------------------------------------------------------------------

if not GT_CSV.exists():
    sys.exit(f"[ERROR] Ground truth not found:\n  {GT_CSV}")

_gt_raw = pd.read_csv(GT_CSV)
_gt_raw = _gt_raw.rename(columns={"ID": "Review ID", "Pos/Neg": "Sentiment"})
_gt_raw = _gt_raw[
    _gt_raw["Topic"].notna() &
    (_gt_raw["Topic"].str.strip() != "") &
    (_gt_raw["Topic"].str.strip() != "Off") &
    _gt_raw["Selected Content"].notna() &
    _gt_raw["Sentiment"].notna()
].copy()
_gt_raw["Topic"] = _gt_raw["Topic"].str.strip()
_gt_raw["Sentiment"] = _gt_raw["Sentiment"].str.strip()
_gt_raw = _gt_raw.reset_index(drop=True)

_BODY_COLS = [f"Body {i}" for i in range(1, N_BODY_COLS + 1) if f"Body {i}" in _gt_raw.columns]
_CONT_COLS = [f"Cont. Body {i}" for i in range(1, N_BODY_COLS + 1) if f"Cont. Body {i}" in _gt_raw.columns]

gt_records = []
for _, row in _gt_raw.iterrows():
    body_lits = [str(row[c]).strip() for c in _BODY_COLS if pd.notna(row[c]) and str(row[c]).strip()]
    cont_lits = [str(row[c]).strip() for c in _CONT_COLS if pd.notna(row[c]) and str(row[c]).strip()]
    concat_field = row.get("Concat", "")
    concat = str(concat_field).strip() if pd.notna(concat_field) and str(concat_field).strip() else \
        " , ".join(body_lits + cont_lits)
    gt_records.append({
        "review_id": norm_id(row["Review ID"]),
        "topic": row["Topic"],
        "sentiment": row["Sentiment"],
        "selected_content": str(row["Selected Content"]),
        "norm_content": normalize(row["Selected Content"]),
        "head": str(row["Head"]).strip() if pd.notna(row.get("Head", "")) else "",
        "body": body_lits,
        "cont": cont_lits,
        "concat": concat,
    })

gt_exact_index: dict = {}
gt_by_review: dict = defaultdict(list)
gt_head_counter: dict = defaultdict(Counter)

for i, rec in enumerate(gt_records):
    key = (rec["review_id"], rec["topic"], rec["sentiment"], rec["norm_content"])
    gt_exact_index.setdefault(key, []).append(i)
    gt_by_review[rec["review_id"]].append(i)
    if rec["head"]:
        gt_head_counter[(rec["topic"], rec["sentiment"])][rec["head"]] += 1

gt_expected_head = {k: c.most_common(1)[0][0] for k, c in gt_head_counter.items()}

# (Review ID, Topic) -> union of body / cont literal sets across all spans for
# that topic. This is the coarser grouping used for the micro precision/recall
# metric (mirrors the original CLAR-2025-style evaluator's grouping).
LITERAL_MATCH_THRESHOLD = 0.8
gt_literal_index: dict = defaultdict(lambda: {"body": set(), "cont": set()})
for rec in gt_records:
    key = (rec["review_id"], rec["topic"])
    gt_literal_index[key]["body"].update(rec["body"])
    gt_literal_index[key]["cont"].update(rec["cont"])
gt_literal_index = dict(gt_literal_index)

print(f"Ground truth: {len(gt_records)} rows, "
      f"{len({r['review_id'] for r in gt_records})} unique Review IDs")
print(f"Expected heads learned for {len(gt_expected_head)} (Topic, Sentiment) pairs\n")


def expected_head_for(topic: str, sentiment: str) -> str:
    return gt_expected_head.get((topic, sentiment)) or make_head(topic, sentiment)


def find_gt_match(review_id: str, topic: str, sentiment: str, norm_content: str,
                   selected_content: str, model) -> tuple:
    """Return (gt_record_or_None, match_type, score)."""
    key = (review_id, topic, sentiment, norm_content)
    idxs = gt_exact_index.get(key)
    if idxs:
        return gt_records[idxs[0]], "exact", 1.0

    candidates = [i for i in gt_by_review.get(review_id, []) if gt_records[i]["topic"] == topic]
    if not candidates:
        candidates = gt_by_review.get(review_id, [])
    if not candidates:
        return None, "none", 0.0

    cand_texts = [gt_records[i]["selected_content"] for i in candidates]
    embs = model.encode([selected_content] + cand_texts, convert_to_numpy=True, show_progress_bar=False)
    query = np.repeat(embs[0:1], len(candidates), axis=0)
    sims = cosine_rows(query, embs[1:])
    best_j = int(np.argmax(sims))
    return gt_records[candidates[best_j]], "fuzzy", float(sims[best_j])


# ----------------------------------------------------------------------
# Micro precision / recall (per-literal matching, (Review ID, Topic) grouped)
# ----------------------------------------------------------------------

def compute_micro_precision_recall(raw: pd.DataFrame, model) -> dict:
    preds: dict = defaultdict(lambda: {"body": set(), "cont": set()})
    for _, row in raw.iterrows():
        lit = str(row["Literal"]).strip()
        if lit in (NO_LITERAL_PLACEHOLDER, "", "nan"):
            continue
        ltype = str(row["Literal Type"]).strip()
        if ltype not in ("body", "cont"):
            continue
        preds[(row["Review ID"], row["Topic"])][ltype].add(lit)
    preds = dict(preds)

    vocab = set()
    for v in preds.values():
        vocab |= v["body"] | v["cont"]
    for v in gt_literal_index.values():
        vocab |= v["body"] | v["cont"]
    vocab = sorted(vocab)

    if not vocab:
        return {}

    embs = model.encode(vocab, convert_to_numpy=True, show_progress_bar=False)
    embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9)
    emb_of = dict(zip(vocab, embs))

    def sims_matrix(a_list: list, b_list: list) -> np.ndarray:
        if not a_list or not b_list:
            return np.zeros((len(a_list), len(b_list)))
        A = np.array([emb_of[a] for a in a_list])
        B = np.array([emb_of[b] for b in b_list])
        return A @ B.T

    out: dict = {}
    for ltype in ("body", "cont", "both"):
        tp_pred = tp_gt = total_pred = total_gt = 0
        topic_counts: dict = defaultdict(lambda: {"tp_pred": 0, "total_pred": 0, "tp_gt": 0, "total_gt": 0})

        all_keys = set(preds) | set(gt_literal_index)
        for key in all_keys:
            topic = key[1]
            pred_lits = preds.get(key, {"body": set(), "cont": set()})
            gt_lits = gt_literal_index.get(key, {"body": set(), "cont": set()})

            if ltype == "body":
                p_set, g_set = pred_lits["body"], gt_lits["body"]
            elif ltype == "cont":
                p_set, g_set = pred_lits["cont"], gt_lits["cont"]
            else:
                p_set, g_set = pred_lits["body"] | pred_lits["cont"], gt_lits["body"] | gt_lits["cont"]

            p_list, g_list = sorted(p_set), sorted(g_set)
            matched_pred = matched_gt = 0
            if p_list and g_list:
                m = sims_matrix(p_list, g_list)
                matched_pred = int((m.max(axis=1) >= LITERAL_MATCH_THRESHOLD).sum())
                matched_gt = int((m.max(axis=0) >= LITERAL_MATCH_THRESHOLD).sum())

            tp_pred += matched_pred; total_pred += len(p_list)
            tp_gt += matched_gt; total_gt += len(g_list)
            tc = topic_counts[topic]
            tc["tp_pred"] += matched_pred; tc["total_pred"] += len(p_list)
            tc["tp_gt"] += matched_gt; tc["total_gt"] += len(g_list)

        precision = tp_pred / total_pred if total_pred else 0.0
        recall = tp_gt / total_gt if total_gt else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        per_topic = {}
        for topic, tc in topic_counts.items():
            p_ = tc["tp_pred"] / tc["total_pred"] if tc["total_pred"] else 0.0
            r_ = tc["tp_gt"] / tc["total_gt"] if tc["total_gt"] else 0.0
            f_ = 2 * p_ * r_ / (p_ + r_) if (p_ + r_) else 0.0
            per_topic[topic] = {"precision": p_, "recall": r_, "f1": f_,
                                 "n_pred": tc["total_pred"], "n_gt": tc["total_gt"]}

        out[ltype] = {
            "precision": precision, "recall": recall, "f1": f1,
            "tp_pred": tp_pred, "total_pred": total_pred,
            "tp_gt": tp_gt, "total_gt": total_gt,
            "per_topic": per_topic,
        }

    return out


# ----------------------------------------------------------------------
# Per-file evaluation
# ----------------------------------------------------------------------

def _stem_parts(llm_csv: Path, base_dir: Path) -> tuple:
    rel = llm_csv.relative_to(base_dir)
    subdir = Path(*rel.parts[:-1]) if len(rel.parts) > 1 else Path(".")
    stem = llm_csv.stem
    stem = re.sub(r"^task\d+_", "", stem)
    stem = re.sub(r"_generator", "", stem)
    stem = re.sub(r"_n\d+$", "", stem)
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


def evaluate(llm_csv: Path, limit: int = None) -> None:
    t_start = time.perf_counter()
    model = _get_sent_model()

    raw = pd.read_csv(llm_csv, dtype={"Review ID": str})
    if "Review ID" not in raw.columns and "ID" in raw.columns:
        raw = raw.rename(columns={"ID": "Review ID"})

    needed = {"Review ID", "Topic", "Sentiment", "Head", "Selected Content", "Literal", "Literal Type"}
    if not needed.issubset(raw.columns):
        print(f"  [SKIP] missing columns {needed - set(raw.columns)} – {llm_csv.name}")
        return

    raw["Review ID"] = raw["Review ID"].apply(norm_id)
    raw["Topic"] = raw["Topic"].astype(str).str.strip()
    raw["Sentiment"] = raw["Sentiment"].astype(str).str.strip()
    raw["Selected Content"] = raw["Selected Content"].astype(str)

    group_keys = ["Review ID", "Topic", "Sentiment", "Selected Content"]
    groups = list(raw.groupby(group_keys, sort=False))
    if limit is not None:
        groups = groups[:limit]

    workings: list = []
    results: list = []
    W = workings.append
    R = results.append

    W("=" * 72)
    W(f"WORKINGS — {llm_csv.name}")
    W("=" * 72)
    W(f"\nInstances (Review ID, Topic, Sentiment, Selected Content groups): {len(groups)}")
    W(f"Similarity threshold (body match)                                : {SIM_THRESHOLD}")
    W(f"\n{MANUAL_ONLY_RULES}\n")

    R("=" * 72)
    R(f"EVALUATION RESULTS — {llm_csv.name}")
    R("=" * 72)

    n_head_eval = n_head_correct = 0
    n_head_rule_mismatch = 0  # actual head != naive code rule (generation bug signal)
    match_type_counts = Counter()
    body_scores = []
    n_body_pass = 0
    n_body_evaluated = 0

    rule_violation_counts: dict = Counter()
    n_literals_checked = 0
    n_literals_clean = 0

    llm_concats_batch = []
    gt_concats_batch = []
    group_meta = []  # parallel: dict per group needing batch score

    instance_records = []

    for (rid, topic, sentiment, content), df in groups:
        head_actual = str(df["Head"].iloc[0]).strip()
        norm_content = normalize(content)

        lits = df[["Literal", "Literal Type"]].copy()
        lits = lits[lits["Literal"].astype(str).str.strip() != NO_LITERAL_PLACEHOLDER]
        body_lits = [str(x).strip() for x in lits.loc[lits["Literal Type"] == "body", "Literal"] if str(x).strip()]
        cont_lits = [str(x).strip() for x in lits.loc[lits["Literal Type"] == "cont", "Literal"] if str(x).strip()]

        gt_rec, match_type, match_score = find_gt_match(rid, topic, sentiment, norm_content, content, model)
        match_type_counts[match_type] += 1

        llm_concat = " , ".join(body_lits + cont_lits)
        gt_concat = gt_rec["concat"] if gt_rec else ""

        instance_records.append({
            "rid": rid, "topic": topic, "sentiment": sentiment, "content": content,
            "head_actual": head_actual, "body_lits": body_lits, "cont_lits": cont_lits,
            "gt_rec": gt_rec, "match_type": match_type, "match_score": match_score,
            "llm_concat": llm_concat, "gt_concat": gt_concat,
        })

        if llm_concat and gt_concat:
            llm_concats_batch.append(llm_concat)
            gt_concats_batch.append(gt_concat)
            group_meta.append(len(instance_records) - 1)

    # Batch-encode all concat pairs for this file in one shot
    body_sim_by_idx: dict = {}
    if llm_concats_batch:
        all_texts = llm_concats_batch + gt_concats_batch
        embs = model.encode(all_texts, convert_to_numpy=True, show_progress_bar=False)
        n = len(llm_concats_batch)
        sims = cosine_rows(embs[:n], embs[n:])
        for meta_idx, sim in zip(group_meta, sims):
            body_sim_by_idx[meta_idx] = float(sim)

    for i, inst in enumerate(instance_records):
        rid, topic, sentiment, content = inst["rid"], inst["topic"], inst["sentiment"], inst["content"]
        head_actual = inst["head_actual"]
        body_lits, cont_lits = inst["body_lits"], inst["cont_lits"]
        gt_rec, match_type, match_score = inst["gt_rec"], inst["match_type"], inst["match_score"]

        W(f"\n{'─' * 60}")
        W(f"ID={rid}  Topic={topic}  Sentiment={sentiment}")
        W(f"  Selected Content : {content[:160]}")

        # ---- Head check ----
        rule_head = make_head(topic, sentiment)
        expected_head = expected_head_for(topic, sentiment)
        head_ok = (head_actual == expected_head)
        n_head_eval += 1
        n_head_correct += int(head_ok)
        if head_actual != rule_head:
            n_head_rule_mismatch += 1

        W(f"  LLM Head         : {head_actual}")
        W(f"  Expected (GT) Head: {expected_head}")
        W(f"  Code-rule Head    : {rule_head}")
        W(f"  → Head {'CORRECT' if head_ok else 'WRONG'}"
          + ("  [rule/GT convention differ]" if rule_head != expected_head else ""))

        # ---- Body match (manual-check print) ----
        W(f"\n  LLM body literals : {body_lits}")
        W(f"  LLM cont  literals: {cont_lits}")
        W(f"  GT match type     : {match_type}"
          + (f"  (content sim={match_score:.4f})" if match_type == "fuzzy" else ""))

        if gt_rec is None:
            W("  [No GT row found for this instance → body NOT evaluated]")
        else:
            W(f"  GT  body literals : {gt_rec['body']}")
            W(f"  GT  cont  literals: {gt_rec['cont']}")
            W(f"  LLM concat        : {inst['llm_concat']}")
            W(f"  GT  concat        : {inst['gt_concat']}")
            sim = body_sim_by_idx.get(i)
            if sim is not None:
                n_body_evaluated += 1
                body_scores.append(sim)
                passed = sim >= SIM_THRESHOLD
                n_body_pass += int(passed)
                W(f"  Cosine similarity : {sim:.4f}  → {'PASS' if passed else 'FAIL'} "
                  f"(threshold {SIM_THRESHOLD})  [MANUAL CHECK: confirm this is the right GT body]")
            else:
                W("  [Empty literal set on one side → similarity not computed]")

        # ---- Rule compliance per body literal ----
        if body_lits:
            W("  Rule compliance (body literals):")
            for lit in body_lits:
                violations = check_literal_rules(lit, topic, content)
                n_literals_checked += 1
                if violations:
                    for v in violations:
                        rule_violation_counts[v.split(":")[0]] += 1
                    W(f"    {lit:<40} → VIOLATES: {'; '.join(violations)}")
                else:
                    n_literals_clean += 1
                    W(f"    {lit:<40} → OK")

    elapsed = time.perf_counter() - t_start

    # ---- Results summary ----
    R(f"\nInstances evaluated: {len(instance_records)}")
    R(f"\n{'─' * 72}")
    R("HEAD CHECK")
    R(f"{'─' * 72}")
    head_acc = n_head_correct / n_head_eval if n_head_eval else 0.0
    R(f"  Correct  : {n_head_correct} / {n_head_eval}  (accuracy = {head_acc:.4f})")
    R(f"  Code-rule vs GT-convention mismatches (potential generation bug): {n_head_rule_mismatch}")

    R(f"\n{'─' * 72}")
    R("BODY MATCH (sentence-embedding cosine similarity vs GT 'Concat')")
    R(f"{'─' * 72}")
    R(f"  GT match type breakdown : {dict(match_type_counts)}")
    if body_scores:
        body_arr = np.array(body_scores)
        R(f"  Evaluated  : {n_body_evaluated}")
        R(f"  Pass rate (>= {SIM_THRESHOLD}) : {n_body_pass}/{n_body_evaluated} = {n_body_pass/n_body_evaluated:.4f}")
        R(f"  Mean similarity        : {body_arr.mean():.4f}")
        R(f"  Median similarity      : {np.median(body_arr):.4f}")
        R(f"  Min / Max               : {body_arr.min():.4f} / {body_arr.max():.4f}")
    else:
        R("  [No instances with both LLM and GT literals to compare]")

    R(f"\n{'─' * 72}")
    R("RULE COMPLIANCE (body literals, mechanically-checkable rules only)")
    R(f"{'─' * 72}")
    clean_rate = n_literals_clean / n_literals_checked if n_literals_checked else 0.0
    R(f"  Literals checked : {n_literals_checked}")
    R(f"  Clean (no violations) : {n_literals_clean}  ({clean_rate:.4f})")
    R(f"  Violations by rule code:")
    for rule, cnt in sorted(rule_violation_counts.items(), key=lambda kv: -kv[1]):
        R(f"    {rule:<14} {cnt}")
    R(f"\n  {MANUAL_ONLY_RULES}")

    # ---- Micro precision / recall (per-literal matching) ----
    raw_for_micro = pd.concat([df for _, df in groups], ignore_index=True) if groups else raw.iloc[0:0]
    micro = compute_micro_precision_recall(raw_for_micro, model)

    R(f"\n{'─' * 72}")
    R("MICRO PRECISION / RECALL (per-literal sentence-embedding matching, "
      f"threshold {LITERAL_MATCH_THRESHOLD}, grouped by Review ID+Topic)")
    R(f"{'─' * 72}")
    R("  Counts are aggregated across ALL instances before computing a single "
      "global ratio (micro), not averaged per-instance (macro).")
    for ltype in ("body", "cont", "both"):
        m = micro.get(ltype)
        if not m:
            continue
        R(f"\n  Label type: {ltype.upper()}")
        R(f"    Precision : {m['precision']:.4f}  ({m['tp_pred']}/{m['total_pred']} literals matched)")
        R(f"    Recall    : {m['recall']:.4f}  ({m['tp_gt']}/{m['total_gt']} GT literals covered)")
        R(f"    F1        : {m['f1']:.4f}")
        R(f"    {'Topic':<16} {'Prec':>8} {'Rec':>8} {'F1':>8} {'n_pred':>8} {'n_gt':>6}")
        for topic in sorted(m["per_topic"]):
            t = m["per_topic"][topic]
            R(f"    {topic:<16} {t['precision']:>8.4f} {t['recall']:>8.4f} {t['f1']:>8.4f} "
              f"{t['n_pred']:>8} {t['n_gt']:>6}")

    R(f"\nRuntime: {elapsed:.2f}s")

    _write(llm_csv, workings, results)
    print("\n".join(results))


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Task 2 evaluator (head + body literal quality)")
    parser.add_argument("--limit", type=int, default=None,
                         help="Evaluate only the first N instances per file (for quick smoke tests)")
    parser.add_argument("--glob", default="*.csv",
                         help="Glob pattern (relative to outputs/task2) selecting which files to evaluate")
    args = parser.parse_args()

    llm_csvs = sorted(
        p for p in LLM_DIR.rglob(args.glob)
        if "old_output" not in p.parts and p.suffix == ".csv"
    )
    if not llm_csvs:
        sys.exit(f"[ERROR] No CSV files found under {LLM_DIR}")

    print(f"{RULE_SUBTASKS_NOTE}")
    print(f"Found {len(llm_csvs)} Task 2 output file(s):\n")
    for p in llm_csvs:
        print(f"  {p.relative_to(TASK_DIR)}")
    print()

    total_start = time.perf_counter()
    for llm_csv in llm_csvs:
        print(f"\n{'#' * 72}")
        print(f"Processing: {llm_csv.relative_to(TASK_DIR)}")
        print(f"{'#' * 72}")
        evaluate(llm_csv, limit=args.limit)

    total_elapsed = time.perf_counter() - total_start
    print(f"\n\nAll done. Results in: {EVAL_DIR.relative_to(TASK_DIR)}/")
    print(f"Total runtime: {total_elapsed:.2f}s")
