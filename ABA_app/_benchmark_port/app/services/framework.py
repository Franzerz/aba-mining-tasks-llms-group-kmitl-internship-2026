#### PATCHARAKORN ####

"""Canonical framework context — direct port of the atom/rule/attack registry in mu_team's
backend/services/abaGraphService.js (lines ~327-660).

This is the structure their whole graph is built on: typed atoms (claim / assumption /
proposition) keyed by (cluster claim, type, label), single- and compound-support rules, and
attack pairs, each tagged with a "level" that the layer modes filter on.

Ported literally, including the parts I would have written differently, because the point is
a like-for-like benchmark: any difference in the output graph should come from the new
translation work, not from me quietly improving their construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# mu_team's display ordering for atoms within a level.
_TYPE_RANK = {"claim": 0, "proposition": 1, "assumption": 2}


def locale_compare_key(value: str) -> tuple:
    """Sort key reproducing JavaScript's `String.prototype.localeCompare`, which their display
    ordering tie-breaks on.

    Python's default string comparison is by code point, and that genuinely differs here:
    `"greet_staff" < "greeting_with_smile"` in Python (because "_" < "i"), but ICU orders
    them the other way round. Two rules, derived empirically from Node and then verified by
    sorting all 796 labels in the database both ways and diffing (exact match):
      1. punctuation is fully ignorable — "a_b" and "ab" compare equal;
      2. on an otherwise equal comparison, lowercase sorts before uppercase.
    """
    stripped = [c for c in str(value) if c.isalnum()]
    return ("".join(stripped).lower(), tuple(1 if c.isupper() else 0 for c in stripped))


@dataclass
class FrameworkContext:
    topic: str
    supporting: str
    selectedClaim: str | None = None
    opposingClaim: str | None = None
    defenseLayerLabel: str | None = None
    atomMeta: dict[str, dict] = field(default_factory=dict)
    supportRules: list[dict] = field(default_factory=list)
    compoundSupportRules: list[dict] = field(default_factory=list)
    attackPairs: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def make_synthetic_defense_layer_label(selected_claim: str | None) -> str:
    base = str(selected_claim or "").strip()
    if not base:
        return "synthetic_defense_claim"
    return f"{base}_2" if base.endswith("_defense") else f"{base}_defense"


def normalize_framework_atom_type(raw_type: Any) -> str:
    if raw_type == "claim":
        return "claim"
    return "assumption" if raw_type == "assumption" else "proposition"


def _cluster_claim(ctx: FrameworkContext, spec: dict, atom_type: str) -> str | None:
    if atom_type == "claim":
        return str(spec.get("label") or "").strip()
    return str(spec.get("clusterClaim") or ctx.selectedClaim or "").strip() or None


def make_framework_atom_key(cluster_claim: Any, atom_type: Any, label: Any) -> str:
    return (f"atom::{str(cluster_claim or '').strip()}"
            f"::{normalize_framework_atom_type(atom_type)}::{str(label or '').strip()}")


def resolve_framework_atom_key(ctx: FrameworkContext, spec: dict) -> str | None:
    label = str(spec.get("label") or "").strip()
    if not label:
        return None
    atom_type = normalize_framework_atom_type(spec.get("type"))
    cluster_claim = _cluster_claim(ctx, spec, atom_type)
    if not cluster_claim:
        return None
    key = make_framework_atom_key(cluster_claim, atom_type, label)
    return key if key in ctx.atomMeta else None


def register_framework_atom(ctx: FrameworkContext, spec: dict) -> str | None:
    label = str(spec.get("label") or "").strip()
    if not label:
        return None
    atom_type = normalize_framework_atom_type(spec.get("type"))
    cluster_claim = _cluster_claim(ctx, spec, atom_type)
    if not cluster_claim:
        return None

    atom_key = make_framework_atom_key(cluster_claim, atom_type, label)
    existing = ctx.atomMeta.get(atom_key)
    count_value = _num_or_none(spec.get("count"))

    level_spec = spec.get("level")
    nxt = dict(existing) if existing else {
        "key": atom_key,
        "label": label,
        "type": atom_type,
        "clusterClaim": cluster_claim,
        "count": None,
        "isFocus": False,
        "level": float(level_spec) if _is_finite(level_spec) else None,
    }

    if atom_type == "claim":
        nxt["clusterClaim"] = label
    elif not nxt.get("clusterClaim") and spec.get("clusterClaim"):
        nxt["clusterClaim"] = str(spec["clusterClaim"]).strip() or None

    if count_value is not None:
        prev = float(nxt.get("count") or 0)
        nxt["count"] = max(prev, count_value) or None
    if spec.get("isFocus"):
        nxt["isFocus"] = True
    if _is_finite(level_spec):
        level = float(level_spec)
        nxt["level"] = level if nxt.get("level") is None else min(float(nxt["level"]), level)

    ctx.atomMeta[atom_key] = nxt
    return atom_key


def register_support_rule(ctx: FrameworkContext, premise_spec: dict, conclusion_spec: dict,
                          level: Any) -> None:
    premise_key = resolve_framework_atom_key(ctx, premise_spec)
    conclusion_key = resolve_framework_atom_key(ctx, conclusion_spec)
    if not premise_key or not conclusion_key or premise_key == conclusion_key:
        return
    key = f"support::{premise_key}::{conclusion_key}"
    if any(r["key"] == key for r in ctx.supportRules):
        return
    ctx.supportRules.append({
        "key": key, "premiseKey": premise_key, "conclusionKey": conclusion_key,
        "level": float(level) if _is_finite(level) else None,
    })


def register_compound_support_rule(ctx: FrameworkContext, premise_specs: list[dict],
                                   conclusion_spec: dict, level: Any) -> None:
    premise_keys: list[str] = []
    seen: set[str] = set()
    for spec in premise_specs or []:
        key = resolve_framework_atom_key(ctx, spec)
        if not key or key in seen:
            continue
        seen.add(key)
        premise_keys.append(key)
    conclusion_key = resolve_framework_atom_key(ctx, conclusion_spec)
    if not conclusion_key or len(premise_keys) < 2:
        return
    key = "compound_support::" + "::".join(premise_keys) + f"::{conclusion_key}"
    if any(r["key"] == key for r in ctx.compoundSupportRules):
        return
    ctx.compoundSupportRules.append({
        "key": key, "premiseKeys": premise_keys, "conclusionKey": conclusion_key,
        "level": float(level) if _is_finite(level) else None,
    })


def register_attack_pair(ctx: FrameworkContext, attacker_spec: dict, target_spec: dict,
                         level: Any) -> None:
    attacker_key = resolve_framework_atom_key(ctx, attacker_spec)
    target_key = resolve_framework_atom_key(ctx, target_spec)
    if not attacker_key or not target_key:
        return
    key = f"attack::{attacker_key}::{target_key}"
    if any(p["key"] == key for p in ctx.attackPairs):
        return
    ctx.attackPairs.append({
        "key": key, "attackerKey": attacker_key, "targetKey": target_key,
        "level": float(level) if _is_finite(level) else None,
    })


# --- payload / graph construction -----------------------------------------------------------

def build_pyarg_payload(ctx: FrameworkContext, selection: dict) -> dict:
    """Port of buildPyArgPayloadFromFrameworkSelection.

    Note the two behaviours this deliberately preserves, both of which the New ABA view exists
    to improve on:
      * every attack is funnelled through ONE synthetic atom `__ctr__{target}` per target, so
        several distinct attackers collapse into a single contrary;
      * an assumption with no attacker still gets a synthetic contrary so ABAF() won't reject
        the framework — recorded in `warnings`, but invisible in their drawn graph.
    """
    atom_keys = set(selection.get("atomKeys") or [])
    language: list[str] = []
    assumptions: list[str] = []
    contraries: dict[str, str] = {}
    helper_contraries: dict[str, str] = {}
    rules: list[dict] = []
    warnings = list(ctx.warnings or [])

    def add_language(label: str) -> None:
        if label not in language:
            language.append(label)

    for atom_key in atom_keys:
        meta = ctx.atomMeta.get(atom_key)
        if not meta:
            continue
        add_language(meta["label"])
        if meta["type"] == "assumption" and meta["label"] not in assumptions:
            assumptions.append(meta["label"])

    conclusions_with_compound = {
        r.get("conclusionKey") for r in (selection.get("compoundSupportRules") or []) if r.get("conclusionKey")
    }

    for rule in selection.get("compoundSupportRules") or []:
        premise_metas = [ctx.atomMeta.get(k) for k in (rule.get("premiseKeys") or [])]
        premise_metas = [m for m in premise_metas if m]
        conclusion_meta = ctx.atomMeta.get(rule.get("conclusionKey"))
        if not conclusion_meta or len(premise_metas) < 2:
            continue
        for m in premise_metas:
            add_language(m["label"])
        add_language(conclusion_meta["label"])
        rules.append({"name": f"Rule{len(rules) + 1}",
                      "premises": [m["label"] for m in premise_metas],
                      "conclusion": conclusion_meta["label"]})

    for rule in selection.get("supportRules") or []:
        premise_meta = ctx.atomMeta.get(rule.get("premiseKey"))
        conclusion_meta = ctx.atomMeta.get(rule.get("conclusionKey"))
        if not premise_meta or not conclusion_meta:
            continue
        if rule.get("conclusionKey") in conclusions_with_compound:
            continue
        add_language(premise_meta["label"])
        add_language(conclusion_meta["label"])
        rules.append({"name": f"Rule{len(rules) + 1}",
                      "premises": [premise_meta["label"]],
                      "conclusion": conclusion_meta["label"]})

    assumption_set = set(assumptions)
    for pair in selection.get("attackPairs") or []:
        attacker_meta = ctx.atomMeta.get(pair.get("attackerKey"))
        target_meta = ctx.atomMeta.get(pair.get("targetKey"))
        if not attacker_meta or not target_meta:
            continue
        if target_meta["label"] not in assumption_set:
            continue
        add_language(attacker_meta["label"])
        contrary_atom = contraries.get(target_meta["label"]) or f"__ctr__{target_meta['label']}"
        contraries[target_meta["label"]] = contrary_atom
        helper_contraries[target_meta["label"]] = contrary_atom
        add_language(contrary_atom)
        rules.append({"name": f"Rule{len(rules) + 1}",
                      "premises": [attacker_meta["label"]],
                      "conclusion": contrary_atom})

    for assumption in assumptions:
        if not contraries.get(assumption):
            synthetic = f"__ctr__{assumption}"
            contraries[assumption] = synthetic
            helper_contraries[assumption] = synthetic
            add_language(synthetic)
            warnings.append(
                f"assumption '{assumption}' had no attacker; added synthetic contrary '{synthetic}'"
            )

    return {
        "payload": {
            "language": language,
            "assumptions": assumptions,
            "contraries": contraries,
            "helperContraries": helper_contraries,
            "rules": rules,
            "query": ctx.selectedClaim or None,
        },
        "warnings": warnings,
    }


def make_framework_node_id(meta: dict) -> str:
    prefix = "C" if meta["type"] == "claim" else ("A" if meta["type"] == "assumption" else "P")
    return f"framework::{meta.get('clusterClaim') or meta['label']}::{prefix}::{meta['label']}"


def _display_sort_key(meta: dict):
    return (
        _TYPE_RANK.get(meta.get("type"), 9),
        0 if meta.get("isFocus") else 1,
        -float(meta.get("count") or 0),
        locale_compare_key(meta.get("label") or ""),
    )


def build_display_rows(ctx: FrameworkContext, selection: dict) -> list[list[str]]:
    atoms = [ctx.atomMeta[k] for k in (selection.get("atomKeys") or []) if k in ctx.atomMeta]
    rows_by_level: dict[float, list[dict]] = {}
    for meta in atoms:
        level = float(meta["level"]) if _is_finite(meta.get("level")) else 99.0
        rows_by_level.setdefault(level, []).append(meta)

    out = []
    for level in sorted(rows_by_level):
        metas = sorted(rows_by_level[level], key=_display_sort_key)
        row = [make_framework_node_id(m) for m in metas]
        if row:
            out.append(row)
    return out


def build_graph(ctx: FrameworkContext, selection: dict) -> dict:
    atom_keys = selection.get("atomKeys") or set()
    cluster_labels = [
        claim for claim in [ctx.selectedClaim, ctx.opposingClaim, ctx.defenseLayerLabel]
        if claim and make_framework_atom_key(claim, "claim", claim) in atom_keys
    ]
    clusters = [{"id": f"framework::{claim}", "label": claim} for claim in cluster_labels]

    nodes: list[dict] = []
    edges: list[dict] = []
    node_id_by_key: dict[str, str] = {}

    for atom_key in atom_keys:
        meta = ctx.atomMeta.get(atom_key)
        if not meta:
            continue
        cluster_claim = (meta["label"] if meta["type"] == "claim"
                         else (meta.get("clusterClaim") or ctx.selectedClaim or meta["label"]))
        node_id = make_framework_node_id(meta)
        node_id_by_key[atom_key] = node_id
        nodes.append({"data": {
            "id": node_id,
            "label": meta["label"],
            "type": meta["type"],
            "clusterId": f"framework::{cluster_claim}",
            "clusterSentiment": "bad" if str(cluster_claim or "").startswith("bad_") else "good",
            "count": (float(meta["count"]) or None) if meta.get("count") is not None else None,
            "isFocus": bool(meta.get("isFocus")),
            "level": meta.get("level"),
        }})

    seen_edges: set[str] = set()

    def push_edge(source_key: str, target_key: str, edge_type: str) -> None:
        source = node_id_by_key.get(source_key)
        target = node_id_by_key.get(target_key)
        if not source or not target:
            return
        key = f"{edge_type}::{source}::{target}"
        if key in seen_edges:
            return
        seen_edges.add(key)
        edges.append({"data": {"id": f"e_{len(edges) + 1}", "source": source,
                               "target": target, "type": edge_type}})

    for rule in selection.get("supportRules") or []:
        push_edge(rule["premiseKey"], rule["conclusionKey"], "support")
    for pair in selection.get("attackPairs") or []:
        push_edge(pair["attackerKey"], pair["targetKey"], "attack")

    return {"clusters": clusters, "nodes": nodes, "edges": edges,
            "displayRows": build_display_rows(ctx, selection)}


def get_layer_mode_max_level(layer_mode: str) -> int:
    return 4 if layer_mode == "layer1" else 7


def select_framework_view(ctx: FrameworkContext, layer_mode: str = "layer2") -> dict:
    layer_mode = "layer1" if str(layer_mode or "layer2").strip().lower() == "layer1" else "layer2"
    max_level = get_layer_mode_max_level(layer_mode)

    # JS reads `Number.isFinite(level) ? level <= max : true` — an atom with no level survives
    # every layer mode. Spelled out rather than squeezed into a comprehension.
    atom_keys: set[str] = set()
    for key, meta in ctx.atomMeta.items():
        level = meta.get("level")
        if _is_finite(level):
            if float(level) <= max_level:
                atom_keys.add(key)
        else:
            atom_keys.add(key)

    def level_ok(item: dict) -> bool:
        level = item.get("level")
        return float(level) <= max_level if _is_finite(level) else True

    support_rules = [
        r for r in (ctx.supportRules or [])
        if level_ok(r) and r["premiseKey"] in atom_keys and r["conclusionKey"] in atom_keys
    ]
    compound_rules = [
        r for r in (ctx.compoundSupportRules or [])
        if level_ok(r) and all(k in atom_keys for k in r.get("premiseKeys") or [])
        and r["conclusionKey"] in atom_keys
    ]
    attack_pairs = [
        p for p in (ctx.attackPairs or [])
        if level_ok(p) and p["attackerKey"] in atom_keys and p["targetKey"] in atom_keys
    ]

    return {"layerMode": layer_mode, "maxLevel": max_level, "atomKeys": atom_keys,
            "supportRules": support_rules, "compoundSupportRules": compound_rules,
            "attackPairs": attack_pairs}


# --- small helpers ---------------------------------------------------------------------------

def _is_finite(value: Any) -> bool:
    if value is None or isinstance(value, bool):
        return False
    try:
        f = float(value)
    except (TypeError, ValueError):
        return False
    return f == f and f not in (float("inf"), float("-inf"))


def _num_or_none(value: Any) -> float | None:
    if not _is_finite(value):
        return None
    return float(value) or None


def unique_by_label_and_count(rows: list[dict], label_key: str) -> list[dict]:
    out: dict[str, dict] = {}
    for row in rows or []:
        label = str(row.get(label_key) or "").strip()
        if not label:
            continue
        count = float(row.get("cnt") or row.get("count") or 0) or 0
        prev = out.get(label)
        if not prev or count > (prev["count"] or 0):
            out[label] = {"label": label, "count": count or None}
    return list(out.values())


def find_attack_pairs_from_graph(nodes: list[dict], edges: list[dict],
                                 attacker_labels: list[str], target_labels: list[str]) -> list[dict]:
    node_by_id = {n["id"]: n for n in nodes or []}
    attacker_set = {str(x or "").strip() for x in (attacker_labels or []) if str(x or "").strip()}
    target_set = {str(x or "").strip() for x in (target_labels or []) if str(x or "").strip()}
    seen: set[str] = set()
    out: list[dict] = []
    for edge in edges or []:
        if edge.get("type") != "attack":
            continue
        src = node_by_id.get(edge.get("source"))
        tgt = node_by_id.get(edge.get("target"))
        attacker = str((src or {}).get("label") or "").strip()
        target = str((tgt or {}).get("label") or "").strip()
        if attacker not in attacker_set or target not in target_set:
            continue
        key = f"{attacker}::{target}"
        if key in seen:
            continue
        seen.add(key)
        out.append({"attacker": attacker, "target": target})
    return out


def select_top_claim_by_score(claim_scores: dict[str, float]) -> str | None:
    claim = None
    top = -1.0
    for candidate, score in claim_scores.items():
        if score > top:
            top = score
            claim = candidate
    return claim
