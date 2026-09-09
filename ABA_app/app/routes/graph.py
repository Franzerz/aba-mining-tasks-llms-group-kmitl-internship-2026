#### PATCHARAKORN ####

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ..aba_builder import build_aba_framework
from ..config import ASPECT_GOLD_FILES, TOPIC_TO_ASPECT
from ..pyarg_service import DEFAULT_SEMANTICS, FAST_SEMANTICS, SEMANTICS, build_aa_view, compute_extensions

router = APIRouter(prefix="/api")


@router.get("/aspects")
def list_aspects() -> dict:
    """`topic_to_aspect` lets the Review page tell which of its topics can open a Graph
    Studio — only the five with Task 3 gold data have a framework."""
    return {"aspects": sorted(ASPECT_GOLD_FILES), "topic_to_aspect": TOPIC_TO_ASPECT}


@router.get("/graph/{aspect}")
def get_graph(aspect: str, semantics: str = DEFAULT_SEMANTICS) -> dict:
    semantics = semantics or DEFAULT_SEMANTICS  # e.g. a bookmarked "?semantics=" with no value
    """Returns ONE canonical structure covering all three views on the advisor's pipeline:
      - new_aba: the honest, pre-translation framework with a multi-valued contrary function
                 C*: A -> P(L) — one assumption can show several distinct contrary atoms.
      - aba:     the reduced, standard framework (C: A -> L) produced by applying the
                 Translation Rule (clone-splitting) to new_aba — this is what py_arg consumes.
      - aa:      the real Dung instantiation via framework.generate_af() on that SAME reduced
                 framework (constructed arguments + derived defeats).
      - extensions: computed by py_arg directly on that same reduced framework object.

    `aba`, `aa`, and `extensions` are all derived from the exact one ABAF instance produced by
    the translator, so the rendered "reduced" picture and the evaluated framework can never
    disagree (the original bug this app was built to avoid). `new_aba` is kept alongside it,
    with `clones_by_original` recording exactly how each multi-contrary assumption was split,
    so the translation step itself is inspectable instead of being collapsed away.
    """
    try:
        result = build_aba_framework(aspect)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    new_fw = result.new_framework
    framework = result.framework
    clones_by_original = result.translation.clones_by_original

    new_aba_rules = [
        {"id": rule.id, "body": sorted(rule.body), "head": rule.head}
        for rule in new_fw.rules
    ]
    aba_rules = [
        {"id": rule.id, "body": sorted(rule.body), "head": rule.head}
        for rule in framework.rules
    ]

    try:
        extensions = compute_extensions(framework, semantics)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except TimeoutError as e:
        raise HTTPException(status_code=504, detail=str(e))

    return {
        "aspect": aspect,
        "semantics": semantics,
        "available_semantics": list(SEMANTICS),
        "fast_semantics": sorted(FAST_SEMANTICS),
        "new_aba": {
            "assumptions": sorted(new_fw.assumptions),
            "language": sorted(new_fw.language),
            "multi_contraries": {a: sorted(cs) for a, cs in new_fw.multi_contraries.items()},
            "rules": new_aba_rules,
            "atom_roles": result.atom_roles,
        },
        "aba": {
            "assumptions": sorted(framework.assumptions),
            "language": sorted(framework.language),
            "contraries": framework.contraries,
            "rules": aba_rules,
            "atom_roles": result.atom_roles,
            "clones_by_original": clones_by_original,
        },
        "aa": build_aa_view(framework),
        "extensions": extensions,
        "pairs": result.pairs,
        "counts": result.counts,
        "heads": result.heads,
    }
