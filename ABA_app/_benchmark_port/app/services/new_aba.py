#### PATCHARAKORN ####

"""The contribution on top of mu_team's pipeline: the advisor's generalized contrary function
C*: A -> P(L), and the Translation Rule that reduces it back to standard ABA.

Everything up to this point is a faithful reproduction of mu_team's construction (see
benchmark/compare_with_mu_team.py — 16/16 identical). This module takes THE SAME framework
selection they build and derives two extra views from it, so a reader can see exactly what the
generalization changes and nothing else.

What it fixes
-------------
mu_team's `buildPyArgPayloadFromFrameworkSelection` funnels every attacker of an assumption
through ONE synthetic atom:

    contraries[target] = "__ctr__<target>"
    rule: attacker_1 -> __ctr__<target>
    rule: attacker_2 -> __ctr__<target>      # a different, unrelated attacker

Standard ABA allows only one contrary per assumption, so several genuinely distinct attackers
become indistinguishable, and the `__ctr__` atoms are never drawn in their graph. Assumptions
with no attacker at all also get a synthetic contrary purely so ABAF() will accept the
framework — recorded as a warning, invisible in the picture.

C* represents the real situation directly: C*(A) is the SET of atoms that attack A. The
Translation Rule then splits each assumption into one clone per contrary,

    C*(B) = {C1, C2}   ->   assumptions B#C1, B#C2   with  C(B#Ci) = Ci

replicating every rule that used B once per clone, plus a bridging rule `B <- B#Ci` so the
original atom stays derivable (without it nothing concludes a contrary and py_arg finds zero
attacks — see aba_translator.py).
"""

from __future__ import annotations

from py_arg.aba_classes.rule import Rule

from ..aba_translator import NewAbaFramework, translate_to_standard_abaf
from .framework import FrameworkContext

SYNTHETIC_PREFIX = "__ctr__"


def build_new_aba_view(ctx: FrameworkContext, selection: dict) -> dict:
    """Derives C*: A -> P(L) from the same selection mu_team's payload builder consumes."""
    atom_keys = set(selection.get("atomKeys") or [])

    assumptions: list[str] = []
    language: list[str] = []
    for atom_key in atom_keys:
        meta = ctx.atomMeta.get(atom_key)
        if not meta:
            continue
        if meta["label"] not in language:
            language.append(meta["label"])
        if meta["type"] == "assumption" and meta["label"] not in assumptions:
            assumptions.append(meta["label"])

    # C*: every attacker becomes a contrary in its own right, instead of collapsing into one
    # __ctr__ placeholder.
    multi_contraries: dict[str, list[str]] = {a: [] for a in assumptions}
    for pair in selection.get("attackPairs") or []:
        attacker = ctx.atomMeta.get(pair.get("attackerKey"))
        target = ctx.atomMeta.get(pair.get("targetKey"))
        if not attacker or not target:
            continue
        if target["label"] not in multi_contraries:
            continue
        if attacker["label"] not in multi_contraries[target["label"]]:
            multi_contraries[target["label"]].append(attacker["label"])

    # An assumption nothing attacks still needs a contrary for the framework to be well formed.
    # Keeping mu_team's exact synthetic name makes the two payloads directly diffable.
    unattacked: list[str] = []
    for assumption in assumptions:
        if not multi_contraries[assumption]:
            synthetic = f"{SYNTHETIC_PREFIX}{assumption}"
            multi_contraries[assumption] = [synthetic]
            unattacked.append(assumption)
            if synthetic not in language:
                language.append(synthetic)

    rules: list[dict] = []
    for rule in selection.get("compoundSupportRules") or []:
        premises = [ctx.atomMeta[k]["label"] for k in rule.get("premiseKeys") or [] if k in ctx.atomMeta]
        conclusion_meta = ctx.atomMeta.get(rule.get("conclusionKey"))
        if not conclusion_meta or len(premises) < 2:
            continue
        rules.append({"premises": premises, "conclusion": conclusion_meta["label"]})

    conclusions_with_compound = {
        r.get("conclusionKey") for r in (selection.get("compoundSupportRules") or []) if r.get("conclusionKey")
    }
    for rule in selection.get("supportRules") or []:
        premise_meta = ctx.atomMeta.get(rule.get("premiseKey"))
        conclusion_meta = ctx.atomMeta.get(rule.get("conclusionKey"))
        if not premise_meta or not conclusion_meta:
            continue
        if rule.get("conclusionKey") in conclusions_with_compound:
            continue
        rules.append({"premises": [premise_meta["label"]], "conclusion": conclusion_meta["label"]})

    for contraries in multi_contraries.values():
        for contrary in contraries:
            if contrary not in language:
                language.append(contrary)

    return {
        "language": language,
        "assumptions": assumptions,
        "multiContraries": multi_contraries,
        "rules": rules,
        # How much the single-contrary form actually loses on this framework.
        "stats": {
            "assumptions": len(assumptions),
            "multiContraryAssumptions": sum(1 for v in multi_contraries.values() if len(v) > 1),
            "maxContraries": max((len(v) for v in multi_contraries.values()), default=0),
            "totalContraries": sum(len(v) for v in multi_contraries.values()),
            "unattackedAssumptions": len(unattacked),
        },
    }


def build_translation_view(new_aba: dict) -> dict:
    """Applies the Translation Rule and returns the reduced, py_arg-ready framework."""
    rules = {
        Rule(f"R{i + 1}", set(r["premises"]), r["conclusion"])
        for i, r in enumerate(new_aba["rules"])
    }
    framework = NewAbaFramework(
        language=set(new_aba["language"]),
        assumptions=set(new_aba["assumptions"]),
        rules=rules,
        multi_contraries={a: set(cs) for a, cs in new_aba["multiContraries"].items()},
    )
    result = translate_to_standard_abaf(framework)
    reduced = result.framework

    payload = {
        "language": sorted(reduced.language),
        "assumptions": sorted(reduced.assumptions),
        "contraries": dict(reduced.contraries),
        "rules": [
            {"name": rule.id, "premises": sorted(rule.body), "conclusion": rule.head}
            for rule in sorted(reduced.rules, key=lambda r: (r.head, sorted(r.body)))
        ],
    }
    return {
        "payload": payload,
        "clonesByOriginal": result.clones_by_original,
        "stats": {
            "originalAssumptions": len(new_aba["assumptions"]),
            "cloneAssumptions": len(reduced.assumptions),
            "rules": len(reduced.rules),
            "isFlat": bool(reduced.is_flat),
        },
    }
