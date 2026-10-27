#### PATCHARAKORN ####

"""
Builds a generalized ABA framework (C*: A -> P(L), the advisor's "New" formalization) from a
topic's contrary data, then reduces it to standard ABA (C: A -> L) via the Translation Rule
(aba_translator.py) so py_arg can actually evaluate it.

The contrary data comes from contrary_sources.py: human Task 3 votes where they exist, otherwise
one local model's predictions over the full pair set, and never a mix within one side. Which is
which travels with the result, so the UI can always say whether a contrary was voted or guessed.

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

CONSTRUCTION, from a Contrary(P)Body(N) row (Original A, A, B, Vote)
-------------------------------------------------------------------
    A          positive-sentiment literal, e.g. easy_check-in            -> FACT
    Original A the assumption for A, always "no_evident_not_" + A        -> ASSUMPTION
    B          negative-sentiment literal                                -> FACT
    Vote       is B contrary to the assumption? (human vote or LLM prediction)
and, once a topic has them, Contrary(N)Body(P) rows (have_evident_B, B, A, Vote): is the
positive literal A contrary to have_evident_B?

    assumptions  = { no_evident_not_A }  u  { have_evident_B }
    facts        = every A and every B          (empty-body rules; they were observed)
    C*(no_evident_not_A) = { B : Vote(A, B) == "Yes" }      <- the multi-valued contrary
                           or { have_evident_not_A } (default) when nothing attacks it
    C*(have_evident_B)   = { A : Contrary(N)Body(P) says "Yes" }
                           or { no_evident_B } (default) when nothing attacks it
    rules: good_{aspect} <- A, no_evident_not_A
           bad_{aspect}  <- B, have_evident_B

The compound rule (fact AND assumption together supporting the claim) is what makes both atoms
point at the head claim, as in CLAR 2025 Fig. 2 / Fig. 7 and the previous team's drawing.

DEFAULT CONTRARIES
------------------
CLAR 2025 (Sect. 2) defines the contrary function as TOTAL and NON-EMPTY (A -> 2^L minus the
empty set), and py_arg refuses a framework otherwise, so an assumption nothing attacks still needs
a contrary. It gets the default one — the literal negation of what it assumes:
    no_evident_not_A  ->  have_evident_not_A
    have_evident_B    ->  no_evident_B
No rule derives a default contrary, so it never attacks anything (these are the diamonds in the
graph, as reviewed with the advisor).

One exception: when a topic mined both `X` and `not_X` (staff: kind_staff / not_kind_staff;
facility: easy_parking / not_easy_parking), the default name `no_evident_not_X` IS the real
assumption behind `X`. Using that atom would make it an attack no annotator voted for, so in
exactly those cases the default gets a `_default` suffix.

DEVIATION FROM THE PAPER (raise with the advisor)
-------------------------------------------------
CLAR 2025 writes ONE rule per review span, with every support and assumption of that span in the
body. Here each literal gets its own rule. The drawn graph is identical either way (every literal
and its assumption point at the head), but the Translation Rule replicates a rule once per
combination of clones in its body, and per-span rules make that explode: measured on staff,
~11.4 billion reduced rules from the voted positive side alone, against 2,368. Per-literal rules are what keeps the reduction
computable. The semantic difference: an attack knocks out one point, not the whole review.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from py_arg.aba_classes.rule import Rule

from .aba_translator import NewAbaFramework, TranslationResult, translate_to_standard_abaf
from .datasets import DEFAULT_DATASET
from .contrary_sources import (
    NEG_PREFIX,
    POS_PREFIX,
    ContrarySource,
    available_aspects,
    no_pair_topics,
    sources_for,
    span_literals,
)

# The sidebar lists a sample, not every pair: the LLM-only topics have up to ~100k per side.
PAIR_SAMPLE_SIZE = 200

# Which contrary sets a framework is built with. "pn" = Contrary(P)Body(N), what attacks the
# praise side; "np" = Contrary(N)Body(P), what attacks the complaint side. With both at once the
# two sets of attacks run in opposite directions across the same literals, which is hard to read,
# so the UI shows one at a time by default. An assumption whose side is left out keeps a
# placeholder contrary, exactly as if that set had not been mined — the evaluation always matches
# the picture.
CONTRARY_SETS = {"pn": ("pn",), "np": ("np",), "both": ("pn", "np")}
DEFAULT_CONTRARY_SET = "pn"

def placeholder_contrary(assumption: str, taken: set[str]) -> str:
    """The default contrary of an assumption nothing in the data attacks (see module docstring):
    have_evident_not_A for no_evident_not_A, no_evident_B for have_evident_B — suffixed with
    `_default` if that name is already a mined atom."""
    if assumption.startswith(POS_PREFIX):
        name = f"have_evident_not_{assumption[len(POS_PREFIX):]}"
    elif assumption.startswith(NEG_PREFIX):
        name = f"no_evident_{assumption[len(NEG_PREFIX):]}"
    else:
        name = f"not_{assumption}"
    if name in taken:
        name = f"{name}_default"
    if name in taken:
        raise ValueError(f"Default contrary '{name}' collides with a mined atom")
    return name


@dataclass(frozen=True)
class AbaBuildResult:
    aspect: str
    new_framework: NewAbaFramework      # C*: A -> P(L), pre-translation
    translation: TranslationResult      # reduced ABAF (py_arg-ready) + clone bookkeeping
    atom_roles: dict[str, str]          # atom -> head | assumption_pos/neg | proposition_pos/neg | placeholder
    pairs: list[dict]                   # a sample of the (assumption, attacker, vote) rows behind C*
    pair_stats: list[dict]              # per source: kind, label, total / yes / no
    contrary_source: dict[str, str | None]  # "pn" / "np" -> "human" | "llm" | None
    counts: dict[str, int]              # literal -> how many mined spans produced it (Task 1 gold)
    heads: list[str]                    # the claim atoms, for the head-claim selector

    @property
    def framework(self):
        """The reduced, standard ABAF -- what py_arg actually evaluates."""
        return self.translation.framework


def _sample_pairs(sources: list[ContrarySource]) -> list[dict]:
    """The Yes rows first (they are the ones that became contraries), then the rest."""
    out: list[dict] = []
    for vote in ("Yes", "No", None):
        for src in sources:
            votes = src.rows["vote"]
            rows = src.rows[votes.isna()] if vote is None else src.rows[votes == vote]
            for row in rows.itertuples(index=False):
                if len(out) >= PAIR_SAMPLE_SIZE:
                    return out
                out.append({"id": len(out) + 1, "category": src.category, "source": src.kind,
                            "assumption": row.assumption, "a": row.literal, "b": row.attacker,
                            "vote": vote})
    return out


def _head_atom(polarity: str, aspect: str, taken: set[str]) -> str:
    """The claim atom for one side of a topic, guaranteed not to collide with a mined literal.

    A literal can be spelled exactly like the claim it backs: Green Harbor's price votes contain
    the literal `good_price` and its location votes `good_location`. The two are not the same
    proposition -- one is "a guest said the price was good", the other is the claim the framework
    derives from such evidence -- and collapsing them into one atom breaks the framework twice
    over: the claim becomes a fact (so it holds no matter what attacks it), and its support rule
    turns into the self-loop `good_price <- good_price, ...`, which is a rule cycle.

    The mined literal keeps its own name, because that is data; the claim is this app's own atom,
    so it is the one that moves. Same `_default`-style convention as placeholder_contrary.
    """
    head = f"{polarity}_{aspect}"
    while head in taken:
        head = f"{head}_claim"
    return head


def build_aba_framework(aspect: str, contraries: str = DEFAULT_CONTRARY_SET,
                        dataset: str = DEFAULT_DATASET) -> AbaBuildResult:
    if contraries not in CONTRARY_SETS:
        raise ValueError(f"Unknown contrary set '{contraries}'. Options: {list(CONTRARY_SETS)}")
    if aspect not in available_aspects(dataset):
        raise ValueError(f"No complete contrary data for '{aspect}' yet. "
                         f"Available: {available_aspects(dataset)}")
    sides = CONTRARY_SETS[contraries]
    sources = sources_for(dataset, aspect)
    pn, np_ = sources["pn"], sources["np"]
    np_ = np_ if np_ is not None and np_.complete and "np" in sides else None
    use_pn = "pn" in sides

    multi_contraries: dict[str, set[str]] = {}   # C*: assumption -> set of contraries
    if pn is not None:
        rows = pn.rows
        a_values = sorted(set(rows["literal"]))     # positive facts
        b_values = sorted(set(rows["attacker"]))    # negative facts
        # Each positive literal's assumption, taken from the pair set rather than reconstructed.
        assumption_of_a = dict(rows[["literal", "assumption"]].drop_duplicates().itertuples(index=False))
        if use_pn:
            for row in rows[rows["vote"] == "Yes"].itertuples(index=False):
                multi_contraries.setdefault(assumption_of_a[row.literal], set()).add(row.attacker)
    elif aspect in no_pair_topics(dataset):
        # Mined with one sentiment only: no pair exists, so the literals come from the spans and
        # every assumption keeps a placeholder contrary.
        lits = span_literals(dataset, aspect)
        a_values, b_values = sorted(lits["Positive"]), sorted(lits["Negative"])
        assumption_of_a = {a: f"{POS_PREFIX}{a}" for a in a_values}
    else:
        raise ValueError(f"No contrary data for '{aspect}'.")
    assumption_of_b = {b: f"{NEG_PREFIX}{b}" for b in b_values}
    if np_ is not None:
        # Only pairs over this framework's own literals: both sides were built from the same
        # literal sets, so anything else would be a stale pair set, not a new attack.
        known_b, known_a = set(b_values), set(a_values)
        for row in np_.rows[np_.rows["vote"] == "Yes"].itertuples(index=False):
            if row.literal in known_b and row.attacker in known_a:
                multi_contraries.setdefault(assumption_of_b[row.literal], set()).add(row.attacker)
    assumptions: set[str] = set(assumption_of_a.values()) | set(assumption_of_b.values())

    # Named only now, once every atom the data brings is known, so a claim can never be spelled
    # like one of them (see _head_atom).
    reserved = set(a_values) | set(b_values) | assumptions
    good_head = _head_atom("good", aspect, reserved)
    bad_head = _head_atom("bad", aspect, reserved)

    # An assumption nothing attacks still needs a contrary for the framework to be well formed.
    mined_atoms = set(a_values) | set(b_values) | assumptions | {good_head, bad_head}
    for asm in sorted(assumptions):
        if asm in multi_contraries:
            continue
        multi_contraries[asm] = {placeholder_contrary(asm, mined_atoms)}

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
    # Anything still unlabelled is a placeholder contrary.
    for contrary_set in multi_contraries.values():
        for c in contrary_set:
            atom_roles.setdefault(c, "placeholder")

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

    used = [src for src in (pn if use_pn else None, np_) if src is not None]
    pn_kind = pn.kind if pn is not None and use_pn else None
    return AbaBuildResult(
        aspect=aspect, new_framework=new_framework, translation=translation,
        atom_roles=atom_roles,
        pairs=_sample_pairs(used),
        pair_stats=[src.stats() for src in used],
        contrary_source={"pn": pn_kind, "np": np_.kind if np_ is not None else None},
        counts=_literal_counts(dataset, aspect, a_values, b_values),
        heads=[good_head, bad_head],
    )


def _literal_counts(dataset: str, aspect: str, a_values: list[str],
                    b_values: list[str]) -> dict[str, int]:
    """How many mined spans produced each literal, from the Task 1 gold file.

    Task 3's literals came from the same extraction as Task 1's, so they line up exactly
    (verified: 42/42 positive and 6/6 negative for check-in). This is the same "evidence count"
    the previous team put in a badge on each node.
    """
    from .review_data import review_rows, topic_label  # local import: avoids a cycle at module load

    topic = topic_label(dataset, aspect)
    if not topic:
        return {}
    wanted = set(a_values) | set(b_values)
    counts: dict[str, int] = {}
    try:
        for sentiment in ("Positive", "Negative"):
            for row in review_rows(dataset, topic, sentiment):
                if row["supporting"] in wanted:
                    counts[row["supporting"]] = int(row["count"])
    except Exception:
        return {}   # counts are a nicety; never break the graph over them
    return counts
