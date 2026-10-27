#### PATCHARAKORN ####

"""Port of mu_team's buildCanonicalFrameworkFromDb / getAbaGraph
(backend/services/abaGraphService.js, lines ~915-1900).

Their graph is NOT one framework per topic. It is a small, focused framework built around one
selected supporting literal:

    level 0  selected claim                     (e.g. good_check-in)
    level 1  the focus literal + its assumptions
    level 2  propositions of the opposing claim that attack those assumptions
    level 3  the opposing claim
    level 4  assumptions of the opposing claim
    level 5  "defense layer" propositions that attack the opposing assumptions
    level 6  a synthetic defense claim
    level 7  defense-layer assumptions

`layer_mode` then truncates that at level 4 (layer1) or 7 (layer2).

Ported literally — same queries, same ordering, same tie-breaks, same fallbacks — so the two
apps produce the same graph for the same request. Deviating here would invalidate the benchmark.
"""

from __future__ import annotations

from typing import Any

from .. import db
from ..normalizers import get_head_claim, normalize_sentiment_or_all, normalize_topic
from .new_aba import build_new_aba_view, build_translation_view
from .framework import (
    FrameworkContext,
    build_graph,
    build_pyarg_payload,
    find_attack_pairs_from_graph,
    make_synthetic_defense_layer_label,
    register_attack_pair,
    register_compound_support_rule,
    register_framework_atom,
    register_support_rule,
    select_framework_view,
    locale_compare_key,
    select_top_claim_by_score,
    unique_by_label_and_count,
)


class HttpError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


def parse_aba_graph_request(query: dict) -> dict:
    topic_raw = str(query.get("topic") or "").strip()
    supporting = str(query.get("supporting") or "").strip()
    sentiment = normalize_sentiment_or_all(query.get("sentiment") or "All")
    topic = normalize_topic(topic_raw)

    try:
        k_raw = float(query.get("k"))
        k = min(int(k_raw), 50) if k_raw > 0 else 8
    except (TypeError, ValueError):
        k = 8

    attack_mode = "cross" if str(query.get("attack_mode") or "all").strip().lower() == "cross" else "all"
    try:
        attack_depth = 2 if int(query.get("attack_depth")) == 2 else 1
    except (TypeError, ValueError):
        attack_depth = 1
    layer_mode = "layer1" if str(query.get("layer_mode") or "layer2").strip().lower() == "layer1" else "layer2"
    focus_only = str(query.get("focus_only") if query.get("focus_only") is not None else "1").strip().lower() in {"1", "true", "yes"}
    show_all_contrary = str(query.get("show_all_contrary") or "0").strip().lower() in {"1", "true", "yes"}
    semantics = str(query.get("semantics") or "Preferred").strip() or "Preferred"
    strategy = "Skeptical" if str(query.get("strategy") or "Credulous").strip() == "Skeptical" else "Credulous"

    return {"topic": topic, "supporting": supporting, "sentiment": sentiment, "k": k,
            "attackMode": attack_mode, "attackDepth": attack_depth, "layerMode": layer_mode,
            "focusOnly": focus_only, "showAllContrary": show_all_contrary,
            "semantics": semantics, "strategy": strategy}


def _resolve_supporting_context(topic_table: str, supporting: str,
                                allowed_claims: set[str]) -> dict | None:
    assumption_match = db.query(
        f"SELECT claim, cnt FROM `{topic_table}` WHERE assumption = %s", [supporting])
    proposition_match = db.query(
        f"SELECT claim, cnt FROM `{topic_table}` WHERE proposition = %s", [supporting])

    prop_in_scope = next((r for r in proposition_match if r["claim"] in allowed_claims), None)
    assum_in_scope = next((r for r in assumption_match if r["claim"] in allowed_claims), None)

    if prop_in_scope:
        return {"supportOrigin": "proposition", "selectedClaim": prop_in_scope["claim"],
                "supportCount": float(prop_in_scope["cnt"] or 0) or None}
    if assum_in_scope:
        return {"supportOrigin": "assumption", "selectedClaim": assum_in_scope["claim"],
                "supportCount": float(assum_in_scope["cnt"] or 0) or None}
    if proposition_match:
        return {"supportOrigin": "proposition", "selectedClaim": proposition_match[0]["claim"],
                "supportCount": float(proposition_match[0]["cnt"] or 0) or None}
    if assumption_match:
        return {"supportOrigin": "assumption", "selectedClaim": assumption_match[0]["claim"],
                "supportCount": float(assumption_match[0]["cnt"] or 0) or None}
    return None


def _in_clause(values: list[str]) -> str:
    return ", ".join(["%s"] * len(values))


def build_canonical_framework_from_db(request: dict) -> dict:
    topic = request["topic"]
    supporting = request["supporting"]
    sentiment = request["sentiment"]
    K = request["k"]
    attack_mode = request["attackMode"]
    attack_depth = request["attackDepth"]
    focus_only = request["focusOnly"]
    show_all_contrary = request["showAllContrary"]

    if not topic or not supporting:
        raise HttpError(400, "topic and supporting are required")
    if not sentiment:
        raise HttpError(400, "sentiment must be Positive, Negative, or All")

    topic_context = db.resolve_topic_context(topic, True)
    if not topic_context["supported"]:
        raise HttpError(400, f"Unsupported topic: {topic}")
    if not topic_context["tablesExist"]:
        raise HttpError(404, f"Missing topic tables for {topic}")
    topic_table, contrary_table = topic_context["topicTable"], topic_context["contraryTable"]

    head_rows = db.fetch_head_claims_by_topic(topic, sentiment)
    allowed_claims = {c for c in (get_head_claim(r) for r in head_rows) if c}
    if not allowed_claims:
        raise HttpError(404, "No claims found for this topic/sentiment")

    supporting_context = _resolve_supporting_context(topic_table, supporting, allowed_claims)
    if not supporting_context:
        raise HttpError(404, "Supporting atom not found in assumption/proposition")
    support_origin = supporting_context["supportOrigin"]
    selected_claim = supporting_context["selectedClaim"]
    support_count = supporting_context["supportCount"]
    if selected_claim not in allowed_claims:
        raise HttpError(404, "Supporting does not belong to selected topic/sentiment claim set")

    selected_claim_propositions_all = db.query(
        f"""SELECT t.proposition, MAX(t.cnt) AS cnt
            FROM `{topic_table}` t
            WHERE t.claim = %s
              AND EXISTS (SELECT 1 FROM `{contrary_table}` c
                          WHERE c.isContrary = 1 AND c.proposition = t.proposition)
            GROUP BY t.proposition
            ORDER BY cnt DESC, t.proposition ASC""", [selected_claim])
    selected_claim_assumptions_all = db.fetch_top_assumptions_by_claim(topic_table, selected_claim)
    selected_claim_support_pairs_all = db.query(
        f"""SELECT t.proposition, t.assumption, MAX(t.cnt) AS cnt
            FROM `{topic_table}` t WHERE t.claim = %s
            GROUP BY t.proposition, t.assumption
            ORDER BY cnt DESC, t.proposition ASC, t.assumption ASC""", [selected_claim])
    selected_claim_attack_pairs = db.query(
        f"""SELECT DISTINCT c.proposition, c.assumption
            FROM `{contrary_table}` c
            JOIN `{topic_table}` p ON p.proposition = c.proposition
            JOIN `{topic_table}` a ON a.assumption = c.assumption
            WHERE c.isContrary = 1 AND p.claim = %s AND a.claim = %s""",
        [selected_claim, selected_claim])

    # ---- draw-graph scaffolding (their nodeMap/edgeMap, used to find real attack pairs) -----
    node_map: dict[str, dict] = {}
    edge_map: dict[str, dict] = {}
    cluster_claim_by_id: dict[str, str] = {}

    def add_node(node_id, label, node_type, cluster_id, is_focus=False, count=None) -> None:
        if not node_id:
            return
        prev = node_map.get(node_id)
        if prev:
            prev["isFocus"] = bool(prev["isFocus"] or is_focus)
            if count is not None:
                prev["count"] = max(float(prev["count"] or 0), float(count or 0)) or None
            return
        node_map[node_id] = {"id": node_id, "label": label, "type": node_type,
                             "clusterId": cluster_id or None, "isFocus": bool(is_focus),
                             "count": (float(count) or None) if count is not None else None}

    def add_edge(source, target, edge_type) -> None:
        if not source or not target:
            return
        key = f"{edge_type}::{source}::{target}"
        if key in edge_map:
            return
        edge_map[key] = {"id": f"e_{len(edge_map) + 1}", "source": source,
                         "target": target, "type": edge_type}

    def parse_node_id(node_id: str) -> dict | None:
        for role in ("A", "P", "C", "R"):
            marker = f"::{role}::"
            i = node_id.find(marker)
            if i >= 0:
                return {"clusterId": node_id[:i], "role": role, "raw": node_id[i + 5:]}
        return None

    selected_cluster_id = f"arg::{topic}::{supporting}::{selected_claim}"
    cluster_claim_by_id[selected_cluster_id] = selected_claim
    selected_claim_node_id = f"{selected_cluster_id}::C::{selected_claim}"
    add_node(selected_claim_node_id, selected_claim, "claim", selected_cluster_id, False)

    if support_origin == "proposition":
        focus_id = f"{selected_cluster_id}::P::{supporting}"
        add_node(focus_id, supporting, "proposition", selected_cluster_id, True, support_count)
    else:
        focus_id = f"{selected_cluster_id}::A::{supporting}"
        add_node(focus_id, supporting, "assumption", selected_cluster_id, True, support_count)
    add_edge(focus_id, selected_claim_node_id, "support")

    # ---- level-1 assumptions of the selected claim -----------------------------------------
    assump_rows: list[dict] = []
    preferred_assumption_row = None
    if support_origin == "proposition":
        claim_lower = str(selected_claim or "").lower()
        expected_prefix = ("no_evident_not_" if claim_lower.startswith("good_")
                           else ("have_evident_" if claim_lower.startswith("bad_") else None))
        if expected_prefix:
            pref_rows = db.query(
                f"SELECT assumption, cnt FROM `{topic_table}` WHERE claim = %s AND assumption = %s LIMIT 1",
                [selected_claim, f"{expected_prefix}{supporting}"])
            preferred_assumption_row = pref_rows[0] if pref_rows else None

    if focus_only and support_origin == "assumption":
        rows = db.query(
            f"SELECT assumption, cnt FROM `{topic_table}` WHERE claim = %s AND assumption = %s LIMIT 1",
            [selected_claim, supporting])
        assump_rows = rows if rows else [{"assumption": supporting, "cnt": support_count}]
    elif focus_only and support_origin == "proposition":
        if preferred_assumption_row:
            assump_rows = [preferred_assumption_row]
        else:
            assump_rows = db.query(
                f"""SELECT a.assumption, MAX(a.cnt) AS cnt,
                       SUM(CASE WHEN p2.claim IS NOT NULL AND p2.claim <> %s THEN 1 ELSE 0 END) AS cross_claim_hits
                    FROM `{contrary_table}` c
                    JOIN `{topic_table}` a ON a.assumption = c.assumption
                    LEFT JOIN `{contrary_table}` c2 ON c2.assumption = a.assumption AND c2.isContrary = 1
                    LEFT JOIN `{topic_table}` p2 ON p2.proposition = c2.proposition
                    WHERE c.isContrary = 1 AND c.proposition = %s AND a.claim = %s
                    GROUP BY a.assumption
                    ORDER BY cross_claim_hits DESC, cnt DESC, a.assumption ASC
                    LIMIT 1""",
                [selected_claim, supporting, selected_claim])
            if not assump_rows:
                assump_rows = db.query(
                    f"""SELECT a.assumption, MAX(a.cnt) AS cnt,
                           CASE WHEN a.assumption LIKE CONCAT('%%', %s, '%%') THEN 1 ELSE 0 END AS match_supporting,
                           SUM(CASE WHEN p.claim IS NOT NULL AND p.claim <> %s THEN 1 ELSE 0 END) AS cross_claim_hits
                        FROM `{topic_table}` a
                        LEFT JOIN `{contrary_table}` c ON c.assumption = a.assumption AND c.isContrary = 1
                        LEFT JOIN `{topic_table}` p ON p.proposition = c.proposition
                        WHERE a.claim = %s
                        GROUP BY a.assumption
                        ORDER BY match_supporting DESC, cross_claim_hits DESC, cnt DESC, a.assumption ASC
                        LIMIT 1""",
                    [supporting, selected_claim, selected_claim])
    else:
        assump_rows = db.fetch_top_assumptions_by_claim(topic_table, selected_claim, K)

    selected_claim_assumption_rows_for_graph = list(assump_rows)
    focal_assumption_raw = (assump_rows[0]["assumption"]
                            if focus_only and support_origin == "proposition" and assump_rows else None)

    # ---- pick the opposing claim -------------------------------------------------------------
    claim_scores: dict[str, float] = {}
    if support_origin == "proposition":
        if focal_assumption_raw:
            rows = db.query(
                f"""SELECT p.claim AS claim, p.cnt AS cnt FROM `{contrary_table}` c
                    JOIN `{topic_table}` p ON p.proposition = c.proposition
                    WHERE c.isContrary = 1 AND c.assumption = %s""", [focal_assumption_raw])
        else:
            rows = db.query(
                f"""SELECT a.claim AS claim, a.cnt AS cnt FROM `{contrary_table}` c
                    JOIN `{topic_table}` a ON a.assumption = c.assumption
                    WHERE c.isContrary = 1 AND c.proposition = %s""", [supporting])
    else:
        rows = db.query(
            f"""SELECT p.claim AS claim, p.cnt AS cnt FROM `{contrary_table}` c
                JOIN `{topic_table}` p ON p.proposition = c.proposition
                WHERE c.isContrary = 1 AND c.assumption = %s""", [supporting])
    db.add_claim_scores(rows, claim_scores, selected_claim, allowed_claims)

    opposing_claim = select_top_claim_by_score(claim_scores)
    if not opposing_claim:
        for row in db.query("SELECT * FROM head WHERE LOWER(Topic)=%s LIMIT 50", [topic]):
            c = get_head_claim(row)
            if c and c != selected_claim:
                opposing_claim = c
                break

    contrary_candidates_count = 0
    assump_rows_b_for_graph: list[dict] = []
    selected_claim_defense_attack_pairs: list[dict] = []
    opposing_claim_support_pairs_all: list[dict] = []

    if opposing_claim:
        opposing_cluster_id = f"arg::{topic}::{supporting}::{opposing_claim}"
        cluster_claim_by_id[opposing_cluster_id] = opposing_claim
        opposing_claim_node_id = f"{opposing_cluster_id}::C::{opposing_claim}"
        add_node(opposing_claim_node_id, opposing_claim, "claim", opposing_cluster_id, False)

        opposing_claim_support_pairs_all = db.query(
            f"""SELECT t.proposition, t.assumption, MAX(t.cnt) AS cnt
                FROM `{topic_table}` t WHERE t.claim = %s
                GROUP BY t.proposition, t.assumption
                ORDER BY cnt DESC, t.proposition ASC, t.assumption ASC""", [opposing_claim])

        if support_origin == "proposition" and focal_assumption_raw:
            count_rows = db.query(
                f"""SELECT COUNT(DISTINCT p.proposition) AS total FROM `{contrary_table}` c
                    JOIN `{topic_table}` p ON p.proposition = c.proposition
                    WHERE c.isContrary = 1 AND c.assumption = %s AND p.claim = %s""",
                [focal_assumption_raw, opposing_claim])
            contrary_candidates_count = int((count_rows[0]["total"] if count_rows else 0) or 0)
            limit_sql = "" if show_all_contrary else "LIMIT %s"
            params: list[Any] = [focal_assumption_raw, opposing_claim]
            if not show_all_contrary:
                params.append(K)
            props_b = db.query(
                f"""SELECT p.proposition, MAX(p.cnt) AS cnt FROM `{contrary_table}` c
                    JOIN `{topic_table}` p ON p.proposition = c.proposition
                    WHERE c.isContrary = 1 AND c.assumption = %s AND p.claim = %s
                    GROUP BY p.proposition ORDER BY cnt DESC, p.proposition ASC {limit_sql}""",
                params)
        else:
            props_b = db.fetch_top_propositions_by_claim(topic_table, opposing_claim, K)
            contrary_candidates_count = len(props_b)

        for r in props_b:
            pid = f"{opposing_cluster_id}::P::{r['proposition']}"
            add_node(pid, r["proposition"], "proposition", opposing_cluster_id, False, r["cnt"])
            add_edge(pid, opposing_claim_node_id, "support")

        assump_rows_b = db.fetch_top_assumptions_by_claim(topic_table, opposing_claim, K)
        props_b_raw = [r["proposition"] for r in props_b if r.get("proposition")]
        assump_rows_b_for_graph = assump_rows_b

        if props_b_raw:
            if focus_only:
                claim_b_lower = str(opposing_claim or "").lower()
                expected_prefix_b = ("have_evident_" if claim_b_lower.startswith("bad_")
                                     else ("no_evident_not_" if claim_b_lower.startswith("good_") else None))
                expected_assumptions = [f"{expected_prefix_b}{p}" for p in props_b_raw] if expected_prefix_b else []
                expected_rows = db.query(
                    f"""SELECT assumption, MAX(cnt) AS cnt FROM `{topic_table}`
                        WHERE claim = %s AND assumption IN ({_in_clause(expected_assumptions)})
                        GROUP BY assumption""",
                    [opposing_claim, *expected_assumptions]) if expected_assumptions else []
                expected_set = {str(r["assumption"] or "") for r in expected_rows}
                expected_cnt = {str(r["assumption"] or ""): float(r["cnt"] or 0) for r in expected_rows}

                pair_rows = db.query(
                    f"""SELECT c.proposition, a.assumption AS assumption, MAX(a.cnt) AS cnt
                        FROM `{contrary_table}` c
                        JOIN `{topic_table}` a ON a.assumption = c.assumption
                        WHERE c.isContrary = 1 AND c.proposition IN ({_in_clause(props_b_raw)})
                          AND a.claim = %s
                        GROUP BY c.proposition, a.assumption""",
                    [*props_b_raw, opposing_claim])

                best_by_proposition: dict[str, dict] = {}
                for row in pair_rows:
                    p = str(row.get("proposition") or "")
                    a = str(row.get("assumption") or "")
                    c = float(row.get("cnt") or 0)
                    prev = best_by_proposition.get(p)
                    if not prev or c > prev["cnt"] or (
                            c == prev["cnt"]
                            and locale_compare_key(a) < locale_compare_key(prev["assumption"])):
                        best_by_proposition[p] = {"assumption": a, "cnt": c}

                chosen: dict[str, dict] = {}
                for p in props_b_raw:
                    expected = f"{expected_prefix_b}{p}" if expected_prefix_b else None
                    if expected and expected in expected_set:
                        chosen[expected] = {"assumption": expected, "cnt": expected_cnt.get(expected, 0)}
                        continue
                    best = best_by_proposition.get(p)
                    if best and best["assumption"]:
                        chosen[best["assumption"]] = {"assumption": best["assumption"], "cnt": best.get("cnt") or 0}

                chosen_rows = sorted(chosen.values(),
                                     key=lambda r: (-float(r.get("cnt") or 0), locale_compare_key(r.get("assumption") or "")))
                if chosen_rows:
                    assump_rows_b_for_graph = chosen_rows
                else:
                    rows2 = db.fetch_assumptions_attacking_propositions(
                        topic_table, contrary_table, opposing_claim, props_b_raw)
                    if rows2:
                        assump_rows_b_for_graph = rows2
            else:
                rows2 = db.fetch_assumptions_attacking_propositions(
                    topic_table, contrary_table, opposing_claim, props_b_raw)
                if rows2:
                    assump_rows_b_for_graph = rows2

        for r in assump_rows_b_for_graph:
            aid = f"{opposing_cluster_id}::A::{r['assumption']}"
            add_node(aid, r["assumption"], "assumption", opposing_cluster_id, False, r.get("cnt"))
            add_edge(aid, opposing_claim_node_id, "support")

        ass_b = [r["assumption"] for r in assump_rows_b_for_graph if r.get("assumption")]
        if ass_b:
            selected_claim_defense_attack_pairs = db.query(
                f"""SELECT DISTINCT c.proposition, c.assumption, p.cnt
                    FROM `{contrary_table}` c
                    JOIN `{topic_table}` p ON p.proposition = c.proposition
                    WHERE c.isContrary = 1 AND p.claim = %s
                      AND c.assumption IN ({_in_clause(ass_b)})
                    ORDER BY p.cnt DESC, c.proposition ASC, c.assumption ASC""",
                [selected_claim, *ass_b])

    for r in selected_claim_assumption_rows_for_graph:
        if support_origin == "assumption" and r["assumption"] == supporting:
            continue
        aid = f"{selected_cluster_id}::A::{r['assumption']}"
        add_node(aid, r["assumption"], "assumption", selected_cluster_id, False, r.get("cnt"))
        add_edge(aid, selected_claim_node_id, "support")

    # ---- attack edges (proposition -> assumption only) --------------------------------------
    proposition_ids_by_raw: dict[str, list[str]] = {}
    assumption_ids_by_raw: dict[str, list[str]] = {}
    for n in node_map.values():
        info = parse_node_id(n["id"])
        if not info:
            continue
        if n["type"] == "proposition":
            proposition_ids_by_raw.setdefault(info["raw"], []).append(n["id"])
        elif n["type"] == "assumption":
            assumption_ids_by_raw.setdefault(info["raw"], []).append(n["id"])

    gathered: list[dict] = []
    if proposition_ids_by_raw and assumption_ids_by_raw:
        seen_keys: set[str] = set()
        prop_raws = list(proposition_ids_by_raw)
        assum_raws = list(assumption_ids_by_raw)
        if opposing_claim:
            rows = db.query(
                f"""SELECT c.proposition, c.assumption,
                           p.claim AS proposition_claim, a.claim AS assumption_claim
                    FROM `{contrary_table}` c
                    JOIN `{topic_table}` p ON p.proposition = c.proposition
                    JOIN `{topic_table}` a ON a.assumption = c.assumption
                    WHERE c.isContrary = 1
                      AND c.proposition IN ({_in_clause(prop_raws)})
                      AND c.assumption IN ({_in_clause(assum_raws)})""",
                [*prop_raws, *assum_raws])
            for r in rows:
                k = f"{r['proposition']}::{r['assumption']}::{r['proposition_claim']}::{r['assumption_claim']}"
                if k in seen_keys:
                    continue
                seen_keys.add(k)
                gathered.append(r)
        else:
            frontier_p = {supporting} if support_origin == "proposition" else set()
            frontier_a = {supporting} if support_origin == "assumption" else set()
            for _ in range(attack_depth):
                if not frontier_p and not frontier_a:
                    break
                fp, fa = list(frontier_p) or [""], list(frontier_a) or [""]
                rows = db.query(
                    f"""SELECT c.proposition, c.assumption,
                               p.claim AS proposition_claim, a.claim AS assumption_claim
                        FROM `{contrary_table}` c
                        JOIN `{topic_table}` p ON p.proposition = c.proposition
                        JOIN `{topic_table}` a ON a.assumption = c.assumption
                        WHERE c.isContrary = 1
                          AND (c.proposition IN ({_in_clause(fp)}) OR c.assumption IN ({_in_clause(fa)}))""",
                    [*fp, *fa])
                frontier_p, frontier_a = set(), set()
                for r in rows:
                    k = f"{r['proposition']}::{r['assumption']}::{r['proposition_claim']}::{r['assumption_claim']}"
                    if k in seen_keys:
                        continue
                    seen_keys.add(k)
                    gathered.append(r)
                    frontier_p.add(r["proposition"])
                    frontier_a.add(r["assumption"])

    for r in gathered:
        if r["proposition"] not in proposition_ids_by_raw or r["assumption"] not in assumption_ids_by_raw:
            continue
        for p_id in proposition_ids_by_raw.get(r["proposition"], []):
            p_info = parse_node_id(p_id)
            if not p_info:
                continue
            claim_src = cluster_claim_by_id.get(p_info["clusterId"])
            if r.get("proposition_claim") and claim_src and r["proposition_claim"] != claim_src:
                continue
            for a_id in assumption_ids_by_raw.get(r["assumption"], []):
                a_info = parse_node_id(a_id)
                if not a_info:
                    continue
                claim_tgt = cluster_claim_by_id.get(a_info["clusterId"])
                if r.get("assumption_claim") and claim_tgt and r["assumption_claim"] != claim_tgt:
                    continue
                if attack_mode == "cross" and p_info["clusterId"] == a_info["clusterId"]:
                    continue
                if opposing_claim and claim_src and claim_tgt:
                    allowed_for_edges = {c for c in (selected_claim, opposing_claim) if c}
                    if claim_src not in allowed_for_edges or claim_tgt not in allowed_for_edges:
                        continue
                add_edge(p_id, a_id, "attack")

    all_nodes = list(node_map.values())
    all_edges = list(edge_map.values())

    selected_claim_assumption_rows = unique_by_label_and_count(
        selected_claim_assumption_rows_for_graph or [], "assumption")
    opposing_claim_assumption_rows = unique_by_label_and_count(assump_rows_b_for_graph or [], "assumption")
    opposing_cluster_id = f"arg::{topic}::{supporting}::{opposing_claim}"
    opposing_claim_proposition_rows = unique_by_label_and_count(
        [{"proposition": n["label"], "cnt": n["count"]} for n in all_nodes
         if n["clusterId"] == opposing_cluster_id and n["type"] == "proposition"], "proposition")
    selected_claim_proposition_rows = unique_by_label_and_count(
        [{"proposition": n["label"], "cnt": n["count"]} for n in all_nodes
         if n["clusterId"] == selected_cluster_id and n["type"] == "proposition"], "proposition")

    selected_claim_assumption_labels = [r["label"] for r in selected_claim_assumption_rows]
    selected_claim_proposition_labels = [r["label"] for r in selected_claim_proposition_rows]
    opposing_claim_proposition_labels = [r["label"] for r in opposing_claim_proposition_rows]
    opposing_claim_assumption_labels = [r["label"] for r in opposing_claim_assumption_rows]

    layer1_attack_pairs = find_attack_pairs_from_graph(
        all_nodes, all_edges, opposing_claim_proposition_labels, selected_claim_assumption_labels)

    attacking_opposing_props = [r for r in opposing_claim_proposition_rows
                                if any(p["attacker"] == r["label"] for p in layer1_attack_pairs)]
    effective_opposing_props = attacking_opposing_props or opposing_claim_proposition_rows
    opposing_labels_for_framework = [r["label"] for r in effective_opposing_props]
    effective_layer1_attack_pairs = [p for p in layer1_attack_pairs
                                     if p["attacker"] in opposing_labels_for_framework]

    opposing_assumption_label_set = set(opposing_claim_assumption_labels)
    selected_prop_count_by_label = {
        str(r.get("proposition") or "").strip(): (float(r.get("cnt") or 0) or None)
        for r in (selected_claim_propositions_all or [])
    }
    defense_prop_labels = list(dict.fromkeys(
        str(r.get("proposition") or "").strip()
        for r in (selected_claim_defense_attack_pairs or [])
        if str(r.get("assumption") or "").strip() in opposing_assumption_label_set
        and str(r.get("proposition") or "").strip()
    ))
    defense_prop_rows = [{"label": label, "count": selected_prop_count_by_label.get(label)}
                         for label in defense_prop_labels]
    defense_support_pairs = [r for r in (selected_claim_support_pairs_all or [])
                             if str(r.get("proposition") or "").strip() in defense_prop_labels]
    defense_assumption_rows = unique_by_label_and_count(defense_support_pairs, "assumption")
    defense_assumption_labels = [r["label"] for r in defense_assumption_rows]

    defense_attack_pairs = []
    for key in dict.fromkeys(
        f"{str(r.get('proposition') or '').strip()}::{str(r.get('assumption') or '').strip()}"
        for r in (selected_claim_defense_attack_pairs or [])
        if str(r.get("proposition") or "").strip() in defense_prop_labels
        and str(r.get("assumption") or "").strip() in opposing_assumption_label_set
    ):
        attacker, target = key.split("::", 1)
        defense_attack_pairs.append({"attacker": attacker, "target": target})

    selected_support_pairs_visible = [
        r for r in (selected_claim_support_pairs_all or [])
        if str(r.get("proposition") or "").strip() in selected_claim_proposition_labels
        and str(r.get("assumption") or "").strip() in selected_claim_assumption_labels]
    opposing_support_pairs_visible = [
        r for r in (opposing_claim_support_pairs_all or [])
        if str(r.get("proposition") or "").strip() in opposing_labels_for_framework
        and str(r.get("assumption") or "").strip() in opposing_claim_assumption_labels]
    defense_support_pairs_visible = [
        r for r in (defense_support_pairs or [])
        if str(r.get("proposition") or "").strip() in defense_prop_labels
        and str(r.get("assumption") or "").strip() in defense_assumption_labels]

    defense_layer_label = (make_synthetic_defense_layer_label(selected_claim)
                           if (defense_prop_rows or defense_assumption_rows) else None)

    # ---- register the canonical framework ---------------------------------------------------
    ctx = FrameworkContext(
        topic=topic, supporting=supporting, selectedClaim=selected_claim,
        opposingClaim=opposing_claim if (opposing_claim and opposing_labels_for_framework) else None,
        defenseLayerLabel=defense_layer_label,
    )

    register_framework_atom(ctx, {"label": selected_claim, "type": "claim",
                                  "clusterClaim": selected_claim, "level": 0})
    if ctx.opposingClaim:
        register_framework_atom(ctx, {"label": ctx.opposingClaim, "type": "claim",
                                      "clusterClaim": ctx.opposingClaim, "level": 3})
    if ctx.defenseLayerLabel:
        register_framework_atom(ctx, {"label": ctx.defenseLayerLabel, "type": "claim",
                                      "clusterClaim": ctx.defenseLayerLabel, "level": 6})

    focus_type = "proposition" if support_origin == "proposition" else "assumption"
    register_framework_atom(ctx, {"label": supporting, "type": focus_type,
                                  "clusterClaim": selected_claim, "count": support_count,
                                  "isFocus": True, "level": 1})
    register_support_rule(ctx,
                          {"label": supporting, "type": focus_type, "clusterClaim": selected_claim},
                          {"label": selected_claim, "type": "claim", "clusterClaim": selected_claim}, 1)

    for row in selected_claim_assumption_rows:
        register_framework_atom(ctx, {
            "label": row["label"], "type": "assumption", "clusterClaim": selected_claim,
            "count": row["count"],
            "isFocus": support_origin == "assumption" and row["label"] == supporting, "level": 1})
        register_support_rule(ctx,
                              {"label": row["label"], "type": "assumption", "clusterClaim": selected_claim},
                              {"label": selected_claim, "type": "claim", "clusterClaim": selected_claim}, 1)

    for row in selected_support_pairs_visible:
        register_compound_support_rule(ctx, [
            {"label": row["proposition"], "type": "proposition", "clusterClaim": selected_claim},
            {"label": row["assumption"], "type": "assumption", "clusterClaim": selected_claim},
        ], {"label": selected_claim, "type": "claim", "clusterClaim": selected_claim}, 1)

    if ctx.opposingClaim:
        for row in effective_opposing_props:
            register_framework_atom(ctx, {"label": row["label"], "type": "proposition",
                                          "clusterClaim": ctx.opposingClaim,
                                          "count": row["count"], "level": 2})
            register_support_rule(ctx,
                                  {"label": row["label"], "type": "proposition", "clusterClaim": ctx.opposingClaim},
                                  {"label": ctx.opposingClaim, "type": "claim", "clusterClaim": ctx.opposingClaim}, 2)
        for row in opposing_claim_assumption_rows:
            register_framework_atom(ctx, {"label": row["label"], "type": "assumption",
                                          "clusterClaim": ctx.opposingClaim,
                                          "count": row["count"], "level": 4})
            register_support_rule(ctx,
                                  {"label": row["label"], "type": "assumption", "clusterClaim": ctx.opposingClaim},
                                  {"label": ctx.opposingClaim, "type": "claim", "clusterClaim": ctx.opposingClaim}, 4)
        for pair in effective_layer1_attack_pairs:
            register_attack_pair(ctx,
                                 {"label": pair["attacker"], "type": "proposition", "clusterClaim": ctx.opposingClaim},
                                 {"label": pair["target"], "type": "assumption", "clusterClaim": selected_claim}, 2)
        for row in opposing_support_pairs_visible:
            register_compound_support_rule(ctx, [
                {"label": row["proposition"], "type": "proposition", "clusterClaim": ctx.opposingClaim},
                {"label": row["assumption"], "type": "assumption", "clusterClaim": ctx.opposingClaim},
            ], {"label": ctx.opposingClaim, "type": "claim", "clusterClaim": ctx.opposingClaim}, 4)

        if ctx.defenseLayerLabel:
            for row in defense_prop_rows:
                register_framework_atom(ctx, {"label": row["label"], "type": "proposition",
                                              "clusterClaim": ctx.defenseLayerLabel,
                                              "count": row["count"], "level": 5})
                register_support_rule(ctx,
                                      {"label": row["label"], "type": "proposition", "clusterClaim": ctx.defenseLayerLabel},
                                      {"label": ctx.defenseLayerLabel, "type": "claim", "clusterClaim": ctx.defenseLayerLabel}, 5)
            for row in defense_assumption_rows:
                register_framework_atom(ctx, {"label": row["label"], "type": "assumption",
                                              "clusterClaim": ctx.defenseLayerLabel,
                                              "count": row["count"], "level": 7})
                register_support_rule(ctx,
                                      {"label": row["label"], "type": "assumption", "clusterClaim": ctx.defenseLayerLabel},
                                      {"label": ctx.defenseLayerLabel, "type": "claim", "clusterClaim": ctx.defenseLayerLabel}, 7)
            for pair in defense_attack_pairs:
                register_attack_pair(ctx,
                                     {"label": pair["attacker"], "type": "proposition", "clusterClaim": ctx.defenseLayerLabel},
                                     {"label": pair["target"], "type": "assumption", "clusterClaim": ctx.opposingClaim}, 5)
            for row in defense_support_pairs_visible:
                register_compound_support_rule(ctx, [
                    {"label": row["proposition"], "type": "proposition", "clusterClaim": ctx.defenseLayerLabel},
                    {"label": row["assumption"], "type": "assumption", "clusterClaim": ctx.defenseLayerLabel},
                ], {"label": ctx.defenseLayerLabel, "type": "claim", "clusterClaim": ctx.defenseLayerLabel}, 7)

    def _rows(rows, *keys):
        out = []
        for r in rows or []:
            item = {k: r.get(k) for k in keys if k != "count"}
            if "count" in keys:
                item["count"] = int(float(r.get("cnt") or 0))
            out.append(item)
        return out

    return {
        "frameworkCtx": ctx,
        "request": request,
        "meta": {
            "selectedClaim": selected_claim,
            "opposingClaim": ctx.opposingClaim,
            "defenseLayerLabel": ctx.defenseLayerLabel,
            "defenseLayerSynthetic": bool(defense_layer_label),
            "defenseLayerSourceClaim": selected_claim,
            "attackMode": attack_mode,
            "attackDepth": attack_depth,
            "layerMode": request["layerMode"],
            "focusOnly": focus_only,
            "showAllContrary": show_all_contrary,
            "semantics": request["semantics"],
            "strategy": request["strategy"],
            "contraryCandidatesCount": contrary_candidates_count,
            "k": K,
            "selectedClaimPropositionsAll": _rows(selected_claim_propositions_all, "proposition", "count"),
            "selectedClaimSupportPairsAll": _rows(selected_claim_support_pairs_all, "proposition", "assumption", "count"),
            "selectedClaimAssumptionsAll": _rows(selected_claim_assumptions_all, "assumption", "count"),
            "opposingClaimSupportPairsAll": _rows(opposing_claim_support_pairs_all, "proposition", "assumption", "count"),
            "selectedClaimAttackPairs": _rows(selected_claim_attack_pairs, "proposition", "assumption"),
            "selectedClaimDefenseAttackPairs": _rows(selected_claim_defense_attack_pairs, "proposition", "assumption", "count"),
            "defenseSupportPairs": _rows(defense_support_pairs, "proposition", "assumption", "count"),
        },
    }


def get_aba_graph(query: dict) -> dict:
    """Port of getAbaGraph — same response shape their frontend consumes."""
    request = parse_aba_graph_request(query or {})
    canonical = build_canonical_framework_from_db(request)
    ctx = canonical["frameworkCtx"]
    selection = select_framework_view(ctx, request["layerMode"])
    graph_build = build_graph(ctx, selection)
    framework_build = build_pyarg_payload(ctx, selection)

    # --- the added contribution, derived from the SAME selection ------------------------
    new_aba = build_new_aba_view(ctx, selection)
    translation = build_translation_view(new_aba)

    attack_edges_count = sum(1 for e in graph_build["edges"] if e["data"]["type"] == "attack")
    attackers_count = len({p["attackerKey"] for p in selection["attackPairs"]})
    targets_count = len({p["targetKey"] for p in selection["attackPairs"]})

    return {
        "clusters": graph_build["clusters"],
        "nodes": graph_build["nodes"],
        "edges": graph_build["edges"],
        "displayRows": graph_build["displayRows"],
        "framework": {"payload": framework_build["payload"], "warnings": framework_build["warnings"]},
        # Extra views. `framework` above stays byte-identical to mu_team's, so the benchmark
        # is unaffected by their presence.
        "newAba": new_aba,
        "translation": translation,
        "evaluation": None,
        "meta": {
            **canonical["meta"],
            "layerMode": selection["layerMode"],
            "layerMaxLevel": selection["maxLevel"],
            "attackEdgesCount": attack_edges_count,
            "attackersCount": attackers_count,
            "targetsCount": targets_count,
            "pyarg": None,
            "pyargPayload": framework_build["payload"],
            "pyargWarnings": framework_build["warnings"],
        },
    }
