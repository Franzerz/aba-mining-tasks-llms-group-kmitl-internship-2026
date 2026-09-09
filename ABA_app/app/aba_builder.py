#### PATCHARAKORN ####

"""
Builds a generalized ABA framework (C*: A -> P(L), the advisor's "New" formalization) directly
from Task 3's gold data, then reduces it to standard ABA (C: A -> L) via the Translation Rule
(aba_translator.py) so py_arg can actually evaluate it.

WHICH ATOM IS THE ASSUMPTION
----------------------------
This is the part an earlier version of this file got backwards, so it is worth stating plainly.
The atom that is DEFEASIBLE -- the thing an argument rests on and that evidence can knock out --
is `no_evident_not_X`, read as "there is no evidence that not-X". The plain mined literal `X` is
an observed FACT from a review, not an assumption.

    proposition (fact)   easy_check-in
    assumption           no_evident_not_easy_check-in
    contrary of it       customer_wait_10_minute_for_check-in   (evidence that it wasn't easy)

`no_evident_not_X` cannot be the *contrary* of `X` -- it is a statement that supports X, not one
that contradicts it. Task 3 asks "is phrase B contrary to phrase A?" where phrase A is the
`Original A` column, i.e. the assumption; a "Yes" means B is evidence against it. The previous
team's database encodes exactly the same split (check_in.assumption = `no_evident_not_*`,
check_in.proposition = the plain literal), so both the logic and their data agree.

CONSTRUCTION, from a gold row (Original A, A, B, Vote)
-----------------------------------------------------
    A          positive-sentiment literal, e.g. easy_check-in            -> FACT
    Original A the assumption for A, always "no_evident_not_" + A        -> ASSUMPTION
    B          negative-sentiment literal                                -> FACT
    Vote       human annotation: is B contrary to the assumption?

    assumptions  = { no_evident_not_A }  u  { have_evident_B }
    facts        = every A and every B          (empty-body rules; they were observed)
    C*(no_evident_not_A) = { B : Vote(A, B) == "Yes" }      <- the multi-valued contrary
                           or { have_evident_not_A } when nothing attacks it
    C*(have_evident_B)   = { no_evident_B }                 <- default; nothing derives it
    rules: good_{aspect} <- A, no_evident_not_A
           bad_{aspect}  <- B, have_evident_B

The compound rule (fact AND assumption together supporting the claim) is what makes both atoms
point at the head claim, matching how the previous team drew it.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from py_arg.aba_classes.rule import Rule

from .aba_translator import NewAbaFramework, TranslationResult, translate_to_standard_abaf
from .config import ASPECT_GOLD_FILES, TASK3_GOLD_DIR, TOPIC_TO_ASPECT

PHRASE_A_CONTRARY_CANDIDATES = ["Original Assumption", "Original A"]
PHRASE_A_BODY_CANDIDATES = ["Assumption", "A"]
PHRASE_B_CANDIDATES = ["Proposition", "B"]
VOTE_CANDIDATES = ["Vote"]


@dataclass(frozen=True)
class AbaBuildResult:
    aspect: str
    new_framework: NewAbaFramework      # C*: A -> P(L), pre-translation
    translation: TranslationResult      # reduced ABAF (py_arg-ready) + clone bookkeeping
    atom_roles: dict[str, str]          # atom -> head | assumption_pos/neg | proposition_pos/neg | contrary
    pairs: list[dict]                   # raw (id, a, b, vote) rows, for the sidebar / audit trail
    counts: dict[str, int]              # literal -> how many mined spans produced it (Task 1 gold)
    heads: list[str]                    # the claim atoms, for the head-claim selector

    @property
    def framework(self):
        """The reduced, standard ABAF -- what py_arg actually evaluates."""
        return self.translation.framework


def _detect_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    mapping = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in mapping:
            return mapping[cand.lower()]
    return None


def build_aba_framework(aspect: str) -> AbaBuildResult:
    if aspect not in ASPECT_GOLD_FILES:
        raise ValueError(f"No gold data for aspect '{aspect}'. Available: {list(ASPECT_GOLD_FILES)}")

    filename, sheet = ASPECT_GOLD_FILES[aspect]
    csv_path = TASK3_GOLD_DIR / filename
    # Four aspects were exported to single-sheet CSVs; facility's gold is still a multi-sheet
    # workbook, so it names which sheet holds the Contrary(P)Body(N) category.
    if sheet is None:
        df = pd.read_csv(csv_path, dtype=str)
    else:
        df = pd.read_excel(csv_path, sheet_name=sheet, dtype=str)

    a_assumption_col = _detect_column(df, PHRASE_A_CONTRARY_CANDIDATES)
    a_body_col = _detect_column(df, PHRASE_A_BODY_CANDIDATES)
    b_col = _detect_column(df, PHRASE_B_CANDIDATES)
    vote_col = _detect_column(df, VOTE_CANDIDATES)
    if not all([a_assumption_col, a_body_col, b_col, vote_col]):
        raise ValueError(f"Missing expected columns in {csv_path.name}. Found: {list(df.columns)}")

    df = df[[a_assumption_col, a_body_col, b_col, vote_col]].copy()
    df.columns = ["AsmA", "A", "B", "Vote"]
    df = df.dropna(subset=["AsmA", "A", "B"])
    for c in ["AsmA", "A", "B"]:
        df[c] = df[c].astype(str).str.strip()
    df["Vote"] = df["Vote"].astype(str).str.strip().str.capitalize()
    df = df[(df["A"] != "") & (df["B"] != "")]

    good_head = f"good_{aspect}"
    bad_head = f"bad_{aspect}"

    a_values = sorted(set(df["A"]))          # positive facts
    b_values = sorted(set(df["B"]))          # negative facts
    # Each positive literal's assumption, taken from the gold file rather than reconstructed.
    assumption_of_a = {row["A"]: row["AsmA"] for _, row in df[["A", "AsmA"]].drop_duplicates().iterrows()}
    assumption_of_b = {b: f"have_evident_{b}" for b in b_values}

    # C*: assumption -> set of contraries.
    multi_contraries: dict[str, set[str]] = {}
    for _, row in df[df["Vote"] == "Yes"].iterrows():
        multi_contraries.setdefault(assumption_of_a[row["A"]], set()).add(row["B"])
    # An assumption nothing attacks still needs a contrary for the framework to be well formed.
    # "have evidence that not-X" is the natural one; nothing derives it, so it is never defeated.
    for a, asm in assumption_of_a.items():
        multi_contraries.setdefault(asm, {f"have_evident_not_{a}"})
    for b, asm in assumption_of_b.items():
        multi_contraries.setdefault(asm, {f"no_evident_{b}"})

    assumptions: set[str] = set(assumption_of_a.values()) | set(assumption_of_b.values())

    rules: set[Rule] = set()
    atom_roles: dict[str, str] = {good_head: "head", bad_head: "head"}

    # Facts: the literals were actually observed in reviews, so they are derivable outright.
    # Without this the contraries would never be derivable and py_arg would find no attacks.
    for a in a_values:
        rules.add(Rule(f"f_{a}", set(), a))
        rules.add(Rule(f"r_good_{a}", {a, assumption_of_a[a]}, good_head))
        atom_roles[a] = "proposition_pos"
        atom_roles[assumption_of_a[a]] = "assumption_pos"
    for b in b_values:
        rules.add(Rule(f"f_{b}", set(), b))
        rules.add(Rule(f"r_bad_{b}", {b, assumption_of_b[b]}, bad_head))
        atom_roles[b] = "proposition_neg"
        atom_roles[assumption_of_b[b]] = "assumption_neg"

    language: set[str] = set(a_values) | set(b_values) | assumptions | {good_head, bad_head}
    for contrary_set in multi_contraries.values():
        language |= contrary_set
    # Anything still unlabelled is a synthetic default contrary.
    for contrary_set in multi_contraries.values():
        for c in contrary_set:
            atom_roles.setdefault(c, "contrary")

    new_framework = NewAbaFramework(
        language=language, assumptions=assumptions, rules=rules, multi_contraries=multi_contraries,
    )
    translation = translate_to_standard_abaf(new_framework)

    # Clones are atoms the translator invented; give each its original's role so the reduced
    # view can colour it without re-deriving anything the translator already knows.
    for original, clones in translation.clones_by_original.items():
        role = atom_roles.get(original, "atom")
        for clone in clones:
            atom_roles.setdefault(clone, role)

    pairs: list[dict] = [
        {"id": i + 1, "a": row["A"], "assumption": row["AsmA"], "b": row["B"],
         "vote": row["Vote"] if row["Vote"] in {"Yes", "No"} else None}
        for i, row in df.reset_index(drop=True).iterrows()
    ]

    return AbaBuildResult(
        aspect=aspect, new_framework=new_framework, translation=translation,
        atom_roles=atom_roles, pairs=pairs,
        counts=_literal_counts(aspect, a_values, b_values),
        heads=[good_head, bad_head],
    )


def _literal_counts(aspect: str, a_values: list[str], b_values: list[str]) -> dict[str, int]:
    """How many mined spans produced each literal, from the Task 1 gold file.

    Task 3's literals came from the same extraction as Task 1's, so they line up exactly
    (verified: 42/42 positive and 6/6 negative for check-in). This is the same "evidence count"
    the previous team put in a badge on each node.
    """
    from .review_data import review_rows  # local import: avoids a cycle at module load

    topic = {v: k for k, v in TOPIC_TO_ASPECT.items()}.get(aspect)
    if not topic:
        return {}
    wanted = set(a_values) | set(b_values)
    counts: dict[str, int] = {}
    try:
        for sentiment in ("Positive", "Negative"):
            for row in review_rows(topic, sentiment):
                if row["supporting"] in wanted:
                    counts[row["supporting"]] = int(row["count"])
    except Exception:
        return {}   # counts are a nicety; never break the graph over them
    return counts
