#### PATCHARAKORN ####

from __future__ import annotations

import threading
from collections import OrderedDict, defaultdict

from fastapi import APIRouter, HTTPException

from ..aba_builder import CONTRARY_SETS, DEFAULT_CONTRARY_SET, build_aba_framework
from ..aba_translator import clone_name
from ..contrary_sources import aspect_status, available_aspects, source_signature, sources_for
from ..datasets import DATASETS, DEFAULT_DATASET, listing
from ..pyarg_service import DEFAULT_SEMANTICS, FAST_PATH_MIN_ASSUMPTIONS, FAST_SEMANTICS, SEMANTICS, evaluate
from ..review_data import dataset_summary, topic_ratios, topic_to_aspect

router = APIRouter(prefix="/api")

# The LLM-only topics take seconds to build and serialise, so a response is kept until the data
# behind it changes (source_signature) — a finished run is computed once, a running one again
# only when it has written new predictions.
_CACHE: OrderedDict[tuple[str, str, str, str], tuple[tuple, dict]] = OrderedDict()
_CACHE_SIZE = 80          # every hotel × topic × contrary set, grounded, with room to spare
_CACHE_LOCK = threading.Lock()
_BUILD_LOCKS: defaultdict[tuple[str, str, str, str], threading.Lock] = defaultdict(threading.Lock)

# The reduced and AA views of an LLM-only topic run to tens of megabytes and are never drawn
# whole (the page asks for a focus first), so for frameworks this big only the focused
# neighbourhood of those two views is sent. The New ABA view always goes out whole.
PARTIAL_MIN_ASSUMPTIONS = FAST_PATH_MIN_ASSUMPTIONS


@router.get("/hotels")
def list_hotels() -> dict:
    """The hotels the site can show, for the picker in the top nav."""
    return {"hotels": listing(), "default": DEFAULT_DATASET}


@router.get("/aspects")
def list_aspects(hotel: str = DEFAULT_DATASET) -> dict:
    """`topic_to_aspect` lets the Review page tell which of its topics can open a Graph
    Studio. `status` says, per topic, whether each side's contraries are human-voted or
    LLM-predicted, and how far an unfinished LLM run has got."""
    return {"aspects": available_aspects(hotel), "topic_to_aspect": topic_to_aspect(hotel),
            "status": aspect_status(hotel), "hotel": hotel}


@router.get("/graph/{aspect}")
def get_graph(aspect: str, semantics: str = DEFAULT_SEMANTICS, focus: str = "",
              contraries: str = DEFAULT_CONTRARY_SET, hotel: str = DEFAULT_DATASET) -> dict:
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

    For a framework of PARTIAL_MIN_ASSUMPTIONS or more, `partial` is true: `aba` and `aa` hold
    only the neighbourhood of `focus` (null without one) and `sizes` gives the full counts.

    `contraries` picks the contrary sets the framework is built with: "pn" (Contrary(P)Body(N),
    the default), "np" (Contrary(N)Body(P)) or "both".
    """
    semantics = semantics or DEFAULT_SEMANTICS  # e.g. a bookmarked "?semantics=" with no value
    contraries = contraries or DEFAULT_CONTRARY_SET
    if contraries not in CONTRARY_SETS:
        raise HTTPException(status_code=400, detail=f"contraries must be one of {list(CONTRARY_SETS)}")
    full = _cached_full_graph(hotel, aspect, semantics, contraries)
    if len(full["aba"]["contraries"]) < PARTIAL_MIN_ASSUMPTIONS:
        return full

    aba, aa = _neighbourhood(full, focus) if focus else (None, None)
    return {
        **full, "aba": aba, "aa": aa, "partial": True, "focus": focus or None,
        "sizes": {"clones": len(full["aba"]["contraries"]),
                  "aa_arguments": len(full["aa"]["arguments"]),
                  "aa_defeats": len(full["aa"]["defeats"])},
    }


def warm_caches() -> None:
    """Loads every contrary source and builds every topic's default graph, so the first visitor
    doesn't pay for it (~5 s for the sources, a few more for the large topics). Run in a
    background thread at startup; a failure only means that page is built on demand instead."""
    for hotel in DATASETS:
        try:
            aspect_status(hotel)
            topic_ratios(hotel)          # Home / Review load these first
            dataset_summary(hotel)
            for aspect in available_aspects(hotel):
                _cached_full_graph(hotel, aspect, DEFAULT_SEMANTICS, DEFAULT_CONTRARY_SET)
        except Exception as exc:  # noqa: BLE001 — warming is best effort
            print(f"[warm_caches] {hotel} stopped early: {exc}")


def _cached_full_graph(dataset: str, aspect: str, semantics: str, contraries: str) -> dict:
    signature = source_signature(dataset, aspect)
    key = (dataset, aspect, semantics, contraries)
    with _CACHE_LOCK:
        hit = _CACHE.get(key)
        if hit and hit[0] == signature:
            _CACHE.move_to_end(key)
            return hit[1]
        build_lock = _BUILD_LOCKS[key]
    with build_lock:   # a second request for the same graph waits for the first build
        with _CACHE_LOCK:
            hit = _CACHE.get(key)
            if hit and hit[0] == signature:
                return hit[1]
        response = _build_graph(dataset, aspect, semantics, contraries)
    with _CACHE_LOCK:
        # Stored under the same key it is looked up by. It used to be written under
        # (aspect, semantics) while being read as the full key, so nothing ever hit and every
        # request rebuilt the graph.
        _CACHE[key] = (signature, response)
        _CACHE.move_to_end(key)
        while len(_CACHE) > _CACHE_SIZE:
            _CACHE.popitem(last=False)
    return response


def _neighbourhood(full: dict, focus: str) -> tuple[dict, dict]:
    """A superset of what the frontend's filterToFocus keeps for `focus` in the reduced and AA
    views: the literal, its assumption, the assumptions it attacks, all their clones, the
    clones' attackers and the claims. The browser still makes the exact cut, so the picture is
    the same as filtering the full data would give."""
    contraries_star = full["new_aba"]["multi_contraries"]
    clones_by_original = full["aba"]["clones_by_original"]
    contraries = full["aba"]["contraries"]

    # Of an assumption the focus attacks, only the clone it attacks is reachable from the focus;
    # of the focus's own assumption, every clone is (they are its co-premises).
    # A literal mined with both sentiments (location: close_to_airport) owns two assumptions.
    attacked = {a for a, cs in contraries_star.items() if focus in cs}
    picked = {a: [clone_name(a, focus)] for a in attacked}
    for own in (f"no_evident_not_{focus}", f"have_evident_{focus}"):
        if own in contraries_star:
            picked[own] = clones_by_original.get(own, [])
    core = set(picked)
    clones = {c for cs in picked.values() for c in cs}

    # Every literal in view keeps its support edges to the claims — including an attacker that
    # was ALSO mined with the other sentiment (location: close_to_airport), whose edge lands on
    # the same claim as the focus. Those rules' other premises are clones nobody keeps, so they
    # are not sent as atoms.
    literals = {focus, *(contraries[c] for c in clones)}
    rules = [r for r in full["aba"]["rules"]
             if clones.intersection(r["body"]) or literals.intersection(r["body"])]
    atoms = {focus, *full["heads"], *core, *clones, *literals}
    atoms |= {atom for r in rules if clones.intersection(r["body"]) for atom in (r["head"], *r["body"])}
    all_roles = full["aba"]["atom_roles"]
    aba = {
        "assumptions": sorted(clones),
        "language": sorted(atoms),
        "contraries": {c: contraries[c] for c in sorted(clones)},
        "rules": rules,
        "atom_roles": {a: all_roles[a] for a in atoms if a in all_roles},
        "clones_by_original": {a: picked[a] for a in sorted(core)},
    }

    arguments = [arg for arg in full["aa"]["arguments"]
                 if arg["conclusion"] in atoms or clones.intersection(arg["premise"])]
    ids = {arg["id"] for arg in arguments}
    defeats = [d for d in full["aa"]["defeats"] if d["source"] in ids and d["target"] in ids]
    return aba, {"arguments": arguments, "defeats": defeats}


def _contrary_sets_available(dataset: str, aspect: str) -> dict[str, bool]:
    sources = sources_for(dataset, aspect)
    has_pn = sources["pn"] is not None
    has_np = sources["np"] is not None and sources["np"].complete
    return {"pn": has_pn, "np": has_np, "both": has_pn and has_np}


def _build_graph(dataset: str, aspect: str, semantics: str,
                 contraries: str = DEFAULT_CONTRARY_SET) -> dict:
    try:
        result = build_aba_framework(aspect, contraries, dataset)
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
        aa_view, extensions, engine = evaluate(framework, semantics)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except TimeoutError as e:
        raise HTTPException(status_code=504, detail=str(e))

    return {
        "aspect": aspect,
        "hotel": dataset,
        "semantics": semantics,
        "contraries": contraries,
        # which sets this topic actually has data for (a set without data only adds placeholders)
        "contrary_sets": _contrary_sets_available(dataset, aspect),
        "available_semantics": list(SEMANTICS),
        "fast_semantics": sorted(FAST_SEMANTICS),
        "new_aba": {
            "assumptions": sorted(new_fw.assumptions),
            "language": sorted(new_fw.language),
            "multi_contraries": {a: sorted(cs) for a, cs in new_fw.multi_contraries.items()},
            "rules": new_aba_rules,
            # clones only exist in the reduced view; leaving them out here keeps this part small
            "atom_roles": {a: result.atom_roles[a] for a in new_fw.language if a in result.atom_roles},
        },
        "aba": {
            "assumptions": sorted(framework.assumptions),
            "language": sorted(framework.language),
            "contraries": framework.contraries,
            "rules": aba_rules,
            "atom_roles": result.atom_roles,
            "clones_by_original": clones_by_original,
        },
        "aa": aa_view,
        "extensions": extensions,
        "engine": engine,
        "pairs": result.pairs,
        "pair_stats": result.pair_stats,
        "contrary_source": result.contrary_source,
        "counts": result.counts,
        "heads": result.heads,
    }
