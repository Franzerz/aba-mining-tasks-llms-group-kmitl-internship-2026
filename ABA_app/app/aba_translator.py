#### PATCHARAKORN ####

"""
Implements the ABA "Translation Rule" from the advisor whiteboard session (see
Task3_Implementation_Report.md / meeting notes): reduces a generalized ABA framework with a
multi-valued contrary function C*: A -> P(L) down to standard (flat) ABA with a single-valued
contrary function C: A -> L, which is what py_arg's ABAF actually accepts.

Why this exists: standard ABA only lets each assumption have ONE contrary atom. Our real data
doesn't fit that -- Task 3 finds MULTIPLE different Propositions that each independently attack
the same Assumption, and those are genuinely different atoms (not one placeholder). The
"Old" formalization (C: A -> L) can't represent that honestly; the "New" one (C*: A -> P(L))
can, but py_arg can't consume it directly. This module is the bridge.

The rule (matching the whiteboard's worked example B-bar = {C1, C2}):
  For each assumption B with |C*(B)| = k contraries {C1, ..., Ck}, split B into k CLONES,
  named "B#C1", "B#C2", ..., each a distinct assumption with exactly ONE contrary:
      C(B#Ci) = Ci
  Every rule that had B in its body is replicated once per clone (logically: "B held" becomes
  "some clone of B held" -- each clone represents one specific line of attack against B).
  An assumption with only one contrary (k=1) still gets "split" into one clone -- i.e. this is
  a strict generalization of the old single-contrary case, not a special-cased branch.

BRIDGING RULES (an addition beyond the whiteboard's literal example -- flag this to the advisor):
  The whiteboard example only covers B appearing in a rule BODY; its C1/C2 were plain atoms.
  If an assumption B is ALSO some other assumption's contrary, splitting B into clones means
  nothing derives the bare atom "B" any more, so that contrary can never be concluded and the
  attack silently disappears. For such a B we also emit, per clone:
      B <- B#Ci
  i.e. "B holds if it holds under any one of its lines of defense". B is no longer an
  assumption after splitting (only its clones are), so the framework stays flat.
  They are emitted ONLY for assumptions that really are someone's contrary. With the current
  builder no contrary is an assumption (contraries are observed facts, as in CLAR 2025), so none
  are emitted -- adding them anyway only inflated the AA view with arguments that conclude an
  original assumption and take part in no attack.
"""

from __future__ import annotations

from dataclasses import dataclass

from py_arg.aba_classes.aba_framework import ABAF
from py_arg.aba_classes.rule import Rule


@dataclass(frozen=True)
class NewAbaFramework:
    """Generalized ABA framework with a multi-valued contrary function C*: A -> P(L) (the
    "New" formalization on the whiteboard), before reduction to the standard form py_arg needs.
    """
    language: set[str]
    assumptions: set[str]
    rules: set[Rule]
    multi_contraries: dict[str, set[str]]  # C*: assumption -> set of contrary atoms (non-empty)

    def __post_init__(self) -> None:
        missing = [a for a in self.assumptions if not self.multi_contraries.get(a)]
        if missing:
            raise ValueError(f"Assumption(s) with no contrary at all in C*: {missing}")


def clone_name(assumption: str, contrary: str) -> str:
    """Deterministic name for the clone of `assumption` paired with one specific `contrary`."""
    return f"{assumption}#{contrary}"


@dataclass(frozen=True)
class TranslationResult:
    framework: ABAF
    # original assumption -> list of its clone names, in the reduced framework. Lets the UI
    # show "this one honest multi-contrary assumption became these N standard assumptions"
    # instead of hiding the split.
    clones_by_original: dict[str, list[str]]


def translate_to_standard_abaf(new_fw: NewAbaFramework) -> TranslationResult:
    """Applies the Translation Rule to reduce a NewAbaFramework (C*: A -> P(L)) to a standard
    py_arg ABAF (C: A -> L)."""
    clones_by_original: dict[str, list[str]] = {}
    reduced_assumptions: set[str] = set()
    reduced_contraries: dict[str, str] = {}
    reduced_language: set[str] = set(new_fw.language)

    for assumption in sorted(new_fw.assumptions):
        contraries = sorted(new_fw.multi_contraries[assumption])
        clones = [clone_name(assumption, c) for c in contraries]
        clones_by_original[assumption] = clones
        for clone, contrary in zip(clones, contraries):
            reduced_assumptions.add(clone)
            reduced_contraries[clone] = contrary
            reduced_language.add(clone)

    # Every rule mentioning an original (multi-contrary) assumption in its body is replicated
    # once per clone -- "the assumption held" becomes "some clone of it held".
    reduced_rules: set[Rule] = set()
    for rule in new_fw.rules:
        multi_valued_in_body = [a for a in rule.body if a in clones_by_original]
        if not multi_valued_in_body:
            reduced_rules.add(rule)
            continue
        _expand_rule(rule, multi_valued_in_body, clones_by_original, reduced_rules)

    # Bridging rules: keep a split assumption derivable from any of its clones, but only where
    # some contrary actually points at it -- see the module docstring.
    contrary_atoms = set(reduced_contraries.values())
    for original, clones in clones_by_original.items():
        if original not in contrary_atoms:
            continue
        for clone in clones:
            reduced_rules.add(Rule(f"r_clone_{clone}", {clone}, original))

    framework = ABAF(
        assumptions=reduced_assumptions,
        rules=reduced_rules,
        language=reduced_language,
        contraries=reduced_contraries,
    )
    return TranslationResult(framework=framework, clones_by_original=clones_by_original)


def _expand_rule(
    rule: Rule,
    multi_valued_atoms: list[str],
    clones_by_original: dict[str, list[str]],
    out: set[Rule],
    _prefix: str = "",
) -> None:
    """Recursively expands one rule into the cross-product of clone choices for every
    multi-contrary assumption in its body (handles bodies with more than one such atom,
    even though the current aba_builder only ever produces at most one per rule)."""
    if not multi_valued_atoms:
        return
    atom, *rest = multi_valued_atoms
    for clone in clones_by_original[atom]:
        new_body = {clone if b == atom else b for b in rule.body}
        if rest:
            _expand_rule(
                Rule(f"{rule.id}_{clone}", new_body, rule.head), rest, clones_by_original, out
            )
        else:
            out.add(Rule(f"{rule.id}_{clone}", new_body, rule.head))
