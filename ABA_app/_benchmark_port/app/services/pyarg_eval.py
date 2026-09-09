#### PATCHARAKORN ####

"""In-process port of mu_team's backend/scripts/pyarg_runner.py.

Their Node backend shelled out to this script per request; since this backend is already
Python we call py_arg directly. The logic below is theirs, essentially verbatim — same
semantics dispatch, same Credulous/Skeptical accepted-assumption rule, same forward-closure
query check, same response keys — so evaluation results are directly comparable.

Two deliberate deviations, both about running in a web process rather than a subprocess:
  * a wall-clock timeout, because a hung exponential semantics would otherwise pin a server
    thread forever (their subprocess could at least be killed);
  * `ABAF` construction errors are returned as {"error": ...} exactly as their script did.
"""

from __future__ import annotations

import multiprocessing
from typing import Any, Iterable

from py_arg.aba_classes.aba_framework import ABAF
from py_arg.aba_classes.rule import Rule
from py_arg.aba_classes.semantics import (
    get_admissible_extensions,
    get_complete_extensions,
    get_conflict_free_extensions,
    get_grounded_extensions as _grounded_module,
    get_naive_extensions,
    get_preferred_extensions,
    get_semi_stable_extensions,
    get_stable_extensions,
)

from .. import settings


def get_grounded_extensions_compat(abaf: ABAF):
    """Their compatibility shim, kept as-is.

    The installed py_arg really does define grounded ABA semantics under the name
    `get_preferred_extensions` inside get_grounded_extensions.py — an upstream typo, which is
    why this probes several names rather than calling one directly.
    """
    for name in ("get_grounded_extensions", "get_grounded_extension", "get_extension",
                 "get_preferred_extensions"):
        fn = getattr(_grounded_module, name, None)
        if callable(fn):
            return fn(abaf)

    nested_mod = getattr(_grounded_module, "get_grounded_extensions_af", None)
    nested_fn = getattr(nested_mod, "get_grounded_extension", None) if nested_mod else None
    if callable(nested_fn):
        af = abaf.generate_af()
        af_ext = nested_fn(af)
        abaf_ext = {arg.conclusion for arg in af_ext
                    if getattr(arg, "conclusion", None) in abaf.assumptions}
        return {frozenset(abaf_ext)}
    raise AttributeError("py_arg grounded semantics API not found")


SEMANTICS_DISPATCH = {
    "Stable": lambda f: get_stable_extensions.get_stable_extensions(f),
    "Preferred": lambda f: get_preferred_extensions.get_preferred_extensions(f),
    "Conflict-Free": lambda f: get_conflict_free_extensions.get_conflict_free_extensions(f),
    "Naive": lambda f: get_naive_extensions.get_naive_extensions(f),
    "Admissible": lambda f: get_admissible_extensions.get_admissible_extensions(f),
    "Complete": lambda f: get_complete_extensions.get_complete_extensions(f),
    "SemiStable": lambda f: get_semi_stable_extensions.get_semi_stable_extensions(f),
    "Grounded": get_grounded_extensions_compat,
}


def get_abaf_extensions(abaf: ABAF, semantics_specification: str):
    fn = SEMANTICS_DISPATCH.get(semantics_specification)
    if not fn:
        raise ValueError(f"Unsupported semantics_specification: {semantics_specification}")
    return fn(abaf)


def get_accepted_assumptions(extensions, strategy_specification: str) -> frozenset:
    ext_list = list(extensions or [])
    if not ext_list:
        return frozenset()
    if strategy_specification == "Skeptical":
        common = set(ext_list[0])
        for ext in ext_list[1:]:
            common.intersection_update(ext)
        return frozenset(common)
    if strategy_specification == "Credulous":
        merged: set = set()
        for ext in ext_list:
            merged.update(ext)
        return frozenset(merged)
    raise ValueError("strategy_specification must be 'Skeptical' or 'Credulous'")


def forward_closure(base: set[str], rules: Iterable[dict]) -> set[str]:
    derived = set(base)
    changed = True
    while changed:
        changed = False
        for r in rules:
            premises = set(r.get("premises", []))
            conclusion = str(r.get("conclusion", ""))
            if conclusion and premises.issubset(derived) and conclusion not in derived:
                derived.add(conclusion)
                changed = True
    return derived


def validate(payload: dict) -> None:
    language = set(payload.get("language", []))
    assumptions = set(payload.get("assumptions", []))
    contraries = payload.get("contraries", {}) or {}
    rules = payload.get("rules", []) or []

    missing = sorted(a for a in assumptions if a not in contraries)
    if missing:
        raise ValueError("Missing contrary mapping for assumptions: " + ", ".join(missing))

    invalid: list[str] = []
    for idx, r in enumerate(rules, start=1):
        premises = set(r.get("premises", []))
        conclusion = str(r.get("conclusion", ""))
        unknown = sorted(x for x in premises | {conclusion} if x and x not in language)
        if unknown:
            invalid.append(f"rule[{idx}] unknown atoms: {', '.join(unknown)}")
    if invalid:
        raise ValueError("; ".join(invalid))

    if not str(payload.get("semantics_specification", "Preferred")).strip():
        raise ValueError("semantics_specification must not be empty")
    strategy = str(payload.get("strategy_specification", "Credulous")).strip()
    if strategy not in {"Credulous", "Skeptical"}:
        raise ValueError("strategy_specification must be 'Credulous' or 'Skeptical'")


def _evaluate(payload: dict) -> dict:
    validate(payload)

    language = set(payload.get("language", []))
    assumptions = set(payload.get("assumptions", []))
    contraries = dict(payload.get("contraries", {}))
    rule_defs = payload.get("rules", []) or []

    rules = set()
    for idx, r in enumerate(rule_defs, start=1):
        rules.add(Rule(str(r.get("name") or f"Rule{idx}"),
                       set(r.get("premises", [])), str(r.get("conclusion", ""))))

    framework = ABAF(assumptions, rules, language, contraries)
    semantics_specification = str(payload.get("semantics_specification", "Preferred")).strip() or "Preferred"
    strategy_specification = str(payload.get("strategy_specification", "Credulous")).strip() or "Credulous"

    exts = get_abaf_extensions(framework, semantics_specification)
    accepted_raw = get_accepted_assumptions(exts, strategy_specification)
    accepted_assumptions = sorted(str(x) for x in accepted_raw if str(x) in assumptions)

    query_value = str(payload.get("query", "") or "").strip()
    include_debug = bool(payload.get("include_debug_fields", False))
    raw_exts: list[list[str]] = []
    normalized_exts: list[list[str]] = []
    derived_per_ext: list[list[str]] = []
    query_in_ext: list[bool] = []

    for ext in exts:
        ext_set = set(ext)
        normalized_exts.append(sorted(str(x) for x in ext_set if str(x) in assumptions))
        if include_debug:
            derived = forward_closure(ext_set, rule_defs)
            raw_exts.append(sorted(str(x) for x in ext_set))
            derived_per_ext.append(sorted(derived))
        if query_value:
            derived = forward_closure(ext_set, rule_defs)
            query_in_ext.append(query_value in derived)

    response = {
        "semantics_specification": semantics_specification,
        "strategy_specification": strategy_specification,
        "extensions": normalized_exts,
        "accepted_assumptions": accepted_assumptions,
        "count": len(normalized_exts),
        "query": query_value or None,
        "credulous": (any(query_in_ext) if query_value else None),
        "skeptical": ((all(query_in_ext) if normalized_exts else False) if query_value else None),
    }
    if include_debug:
        response["raw_extensions"] = raw_exts
        response["derived"] = derived_per_ext
    return response


def _worker(payload: dict, queue: multiprocessing.Queue) -> None:
    """Module-level so it is picklable under Windows' spawn start method."""
    try:
        queue.put(("ok", _evaluate(payload)))
    except Exception as exc:  # noqa: BLE001
        queue.put(("error", str(exc)))


def evaluate(payload: dict) -> dict:
    """Same contract as their runner: errors come back as {"error": "..."} rather than raising.

    Runs in a child PROCESS, not a thread. These semantics are exponential and a runaway
    computation cannot be interrupted from inside Python — a thread would keep burning a core
    forever and, worse, `ThreadPoolExecutor.__exit__` blocks on it, so the request never
    returns. mu_team had the same problem and solved it the same way: their Node backend
    SIGKILLed the Python subprocess. Here the child is terminated on timeout.
    """
    try:
        ctx = multiprocessing.get_context("spawn")
        queue: multiprocessing.Queue = ctx.Queue()
        proc = ctx.Process(target=_worker, args=(payload or {}, queue), daemon=True)
        proc.start()
        proc.join(settings.PYARG_TIMEOUT_SECONDS)

        if proc.is_alive():
            proc.terminate()
            proc.join(5)
            semantics = payload.get("semantics_specification")
            hint = ("Try layer_mode=layer1, or a smaller k."
                    if semantics == "Grounded"
                    else "Try Grounded, or layer_mode=layer1.")
            return {"error": (
                f"PyArg evaluation exceeded {settings.PYARG_TIMEOUT_SECONDS}s. py_arg expands "
                f"every derivation into an argument before computing '{semantics}', which does "
                f"not scale to a framework this size ({len(payload.get('assumptions') or [])} "
                f"assumptions, {len(payload.get('rules') or [])} rules). {hint}"
            )}

        if queue.empty():
            return {"error": f"PyArg evaluation failed (exit code {proc.exitcode})"}
        status, value = queue.get()
        return value if status == "ok" else {"error": value}
    except Exception as exc:  # noqa: BLE001 — mirrors their catch-all
        return {"error": str(exc)}


def sanitize_result(result: dict) -> dict:
    """Port of sanitizePyArgResult: clip the extension list and drop debug fields."""
    if not isinstance(result, dict):
        return result
    extensions = result.get("extensions") if isinstance(result.get("extensions"), list) else []
    max_extensions = settings.PYARG_MAX_EXTENSIONS
    clipped = extensions[:max_extensions] if max_extensions > 0 else extensions

    sanitized = {**result, "extensions": clipped, "count": len(extensions)}
    if not settings.PYARG_INCLUDE_DEBUG_FIELDS:
        sanitized.pop("raw_extensions", None)
        sanitized.pop("derived", None)
    return sanitized
