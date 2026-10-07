#### PATCHARAKORN ####

"""
Thin, correctly-named wrapper around py_arg's ABA semantics functions.

Upstream naming bug (verified directly against the installed package,
.venv/Lib/site-packages/py_arg/aba_classes/semantics/get_grounded_extensions.py): the module
is named get_grounded_extensions.py but the function INSIDE it is literally named
`get_preferred_extensions` (a typo in py_arg itself, not in this project or in mu_team's
code — this is exactly why mu_team's pyarg_runner.py had to guess across several possible
function names for grounded semantics). We import it aliased and expose it under its correct
name here so nothing downstream has to know about this upstream quirk.

Every semantics function in py_arg internally calls `framework.generate_af()` to get the real
Dung AA instantiation and delegates to the AA-level algorithm — so `generate_af()` is already
the single source of truth py_arg itself uses. We call it once, explicitly, and reuse the same
result for the "AA view" sent to the frontend — never a second, separately-derived AA graph.
"""

from __future__ import annotations

import concurrent.futures

from py_arg.aba_classes.aba_framework import ABAF

from . import fast_af
from py_arg.aba_classes.semantics import (
    get_admissible_extensions as _admissible,
    get_complete_extensions as _complete,
    get_conflict_free_extensions as _conflict_free,
    get_grounded_extensions as _grounded_mislabeled,  # see module docstring
    get_naive_extensions as _naive,
    get_preferred_extensions as _preferred,
    get_semi_stable_extensions as _semi_stable,
    get_stable_extensions as _stable,
)

SEMANTICS = {
    "grounded": lambda f: _grounded_mislabeled.get_preferred_extensions(f),  # real name, wrong file
    "conflict_free": lambda f: _conflict_free.get_conflict_free_extensions(f),
    "naive": lambda f: _naive.get_naive_extensions(f),
    "admissible": lambda f: _admissible.get_admissible_extensions(f),
    "complete": lambda f: _complete.get_complete_extensions(f),
    "preferred": lambda f: _preferred.get_preferred_extensions(f),
    "stable": lambda f: _stable.get_stable_extensions(f),
    "semi_stable": lambda f: _semi_stable.get_semi_stable_extensions(f),
}

# Verified directly by timing each one against a real 48-assumption/142-argument framework
# (check-in): "grounded" uses a proper least-fixed-point algorithm (0.01s, always terminates,
# always exactly one extension) — genuinely fast regardless of framework size. Every other
# semantics here is exponential in the number of ARGUMENTS (not assumptions) by construction:
# admissible/conflict_free enumerate the full assumption powerset (2^|assumptions|), and
# preferred/stable/complete/naive/semi_stable use an unpruned binary-branching search over
# every AA-level argument (2^|arguments|, and |arguments| is usually >> |assumptions| once
# generate_af() expands every possible derivation). "stable" alone took >60s and was killed on
# the 142-argument check-in framework — this is a real limitation of py_arg's bundled
# algorithms, not something fixable by calling them differently. Default to "grounded"; treat
# the rest as "try it, but it may legitimately never finish for a framework this size."
FAST_SEMANTICS = {"grounded"}
DEFAULT_SEMANTICS = "grounded"
EXTENSION_TIMEOUT_SECONDS = 15

# Above this many assumptions (clones) py_arg's own generate_af / grounded loops take minutes to
# hours (see fast_af.py), so the verified-identical indexed version is used instead. Every
# human-voted topic stays below it and is evaluated by py_arg itself. The exhaustive semantics
# are refused outright up there: they could never finish, and a timed-out worker thread keeps
# burning CPU in the background because Python cannot kill it.
FAST_PATH_MIN_ASSUMPTIONS = 2500


def uses_fast_path(framework: ABAF) -> bool:
    return len(framework.assumptions) >= FAST_PATH_MIN_ASSUMPTIONS


def evaluate(framework: ABAF, semantics: str) -> tuple[dict, list[list[str]], str]:
    """(AA view, extensions, engine) from ONE argument graph, whichever engine builds it."""
    if semantics not in SEMANTICS:
        raise ValueError(f"Unknown semantics '{semantics}'. Options: {list(SEMANTICS)}")
    if not uses_fast_path(framework):
        return build_aa_view(framework), compute_extensions(framework, semantics), "py_arg"

    if semantics not in FAST_SEMANTICS:
        raise ValueError(
            f"'{semantics}' is not available for a framework this size "
            f"({len(framework.assumptions):,} assumptions): py_arg enumerates it exhaustively. "
            f"Use 'grounded'."
        )
    af = fast_af.generate(framework)
    view = {
        "arguments": [{"id": name, "premise": sorted(premise), "conclusion": conclusion}
                      for name, (premise, conclusion) in af.arguments.items()],
        "defeats": [{"source": src, "target": dst} for src, dst in af.defeats],
    }
    extension = sorted(fast_af.grounded_assumptions(framework, af))
    return view, [extension], "indexed (verified = py_arg)"


def compute_extensions(framework: ABAF, semantics: str) -> list[list[str]]:
    if semantics not in SEMANTICS:
        raise ValueError(f"Unknown semantics '{semantics}'. Options: {list(SEMANTICS)}")

    if semantics in FAST_SEMANTICS:
        extensions = SEMANTICS[semantics](framework)
        return [sorted(ext) for ext in extensions]

    # Everything else can be genuinely exponential (see note above) — never let a slow
    # computation hang the request forever; time it out and report clearly instead.
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        future = ex.submit(SEMANTICS[semantics], framework)
        try:
            extensions = future.result(timeout=EXTENSION_TIMEOUT_SECONDS)
        except concurrent.futures.TimeoutError:
            raise TimeoutError(
                f"'{semantics}' did not finish within {EXTENSION_TIMEOUT_SECONDS}s. This framework has "
                f"{len(framework.assumptions)} assumptions — py_arg's '{semantics}' algorithm is "
                f"exponential and may not be feasible at this size. Try 'grounded' (always fast), "
                f"or a smaller aspect/subset."
            )
    return [sorted(ext) for ext in extensions]


def build_aa_view(framework: ABAF) -> dict:
    """The genuine ABA->AA (Dung) instantiation — real constructed arguments and real
    derived defeats, straight from py_arg's own `generate_af()`. Nothing re-derived, nothing
    hidden: every argument/defeat py_arg computes here is exactly what gets rendered."""
    af = framework.generate_af()
    arguments = [
        {"id": arg.name, "premise": sorted(arg.premise), "conclusion": arg.conclusion}
        for arg in af.arguments
    ]
    defeats = [
        {"source": defeat.from_argument.name, "target": defeat.to_argument.name}
        for defeat in af.defeats
    ]
    return {"arguments": arguments, "defeats": defeats}
