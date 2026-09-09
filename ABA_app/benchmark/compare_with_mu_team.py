"""Structural diff: my FastAPI port vs mu_team's live Node app, same DB, same request."""
import json
import sys
import urllib.parse
import urllib.request

sys.path.insert(0, r"d:\user\ReaLearn\internship\ABA_app")
from app.services.aba_graph_service import get_aba_graph  # noqa: E402

MU = "http://127.0.0.1:3000/api/aba-graph"

CASES = [
    {"topic": "check-in", "supporting": "easy_check-in", "sentiment": "All"},
    {"topic": "check-in", "supporting": "easy_check-in", "sentiment": "All", "layer_mode": "layer1"},
    {"topic": "check-in", "supporting": "no_evident_not_easy_check-in", "sentiment": "All"},
    {"topic": "staff", "supporting": "friendly_staff", "sentiment": "All"},
    {"topic": "staff", "supporting": "helpful_staff", "sentiment": "All", "k": "5"},
    {"topic": "staff", "supporting": "friendly_staff", "sentiment": "All", "show_all_contrary": "1"},
    {"topic": "price", "supporting": "value_for_money", "sentiment": "All"},
    {"topic": "price", "supporting": "good_value", "sentiment": "All", "layer_mode": "layer1"},
    {"topic": "check-out", "supporting": "possibility_for_late_check_out", "sentiment": "All"},
    {"topic": "check-out", "supporting": "late_departure_no_extra_cost", "sentiment": "Positive"},
    {"topic": "price", "supporting": "value_for_money", "sentiment": "All", "attack_depth": "2"},
    {"topic": "staff", "supporting": "friendly_staff", "sentiment": "All", "k": "50"},
    {"topic": "check-in", "supporting": "easy_check-in", "sentiment": "All", "focus_only": "0", "show_all_contrary": "1"},
    {"topic": "check-in", "supporting": "easy_check-in", "sentiment": "Positive"},
    {"topic": "staff", "supporting": "friendly_staff", "sentiment": "All", "focus_only": "0"},
    {"topic": "staff", "supporting": "friendly_staff", "sentiment": "All", "attack_mode": "cross"},
]


def fetch_mu(params):
    url = MU + "?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=180) as r:
        return json.loads(r.read().decode("utf-8"))


def norm_nodes(g):
    return sorted(
        (n["data"]["id"], n["data"]["label"], n["data"]["type"], n["data"]["clusterId"],
         n["data"]["clusterSentiment"], _num(n["data"]["count"]), bool(n["data"]["isFocus"]),
         _num(n["data"]["level"]))
        for n in g["nodes"]
    )


def norm_edges(g):
    # ids are positional (e_1, e_2...) and depend on insertion order; compare the relation.
    return sorted((e["data"]["source"], e["data"]["target"], e["data"]["type"]) for e in g["edges"])


def _num(v):
    return None if v is None else float(v)


def norm_payload(p):
    return {
        "language": sorted(p["language"]),
        "assumptions": sorted(p["assumptions"]),
        "contraries": p["contraries"],
        "helperContraries": p.get("helperContraries") or {},
        "rules": sorted((tuple(sorted(r["premises"])), r["conclusion"]) for r in p["rules"]),
        "query": p.get("query"),
    }


def compare(case):
    mine = get_aba_graph(dict(case))
    theirs = fetch_mu(case)
    diffs = []

    if norm_nodes(mine) != norm_nodes(theirs):
        a, b = norm_nodes(mine), norm_nodes(theirs)
        only_mine = [x for x in a if x not in b][:3]
        only_theirs = [x for x in b if x not in a][:3]
        diffs.append(f"NODES differ (mine {len(a)} vs theirs {len(b)}); only-mine={only_mine} only-theirs={only_theirs}")
    if norm_edges(mine) != norm_edges(theirs):
        a, b = norm_edges(mine), norm_edges(theirs)
        diffs.append(f"EDGES differ (mine {len(a)} vs theirs {len(b)}); "
                     f"only-mine={[x for x in a if x not in b][:3]} only-theirs={[x for x in b if x not in a][:3]}")
    if mine["displayRows"] != theirs["displayRows"]:
        diffs.append(f"DISPLAYROWS differ: mine={[len(r) for r in mine['displayRows']]} "
                     f"theirs={[len(r) for r in theirs['displayRows']]}")
    pm, pt = norm_payload(mine["framework"]["payload"]), norm_payload(theirs["framework"]["payload"])
    for key in pm:
        if pm[key] != pt[key]:
            diffs.append(f"PAYLOAD.{key} differs (mine {len(pm[key]) if hasattr(pm[key],'__len__') else pm[key]} "
                         f"vs theirs {len(pt[key]) if hasattr(pt[key],'__len__') else pt[key]})")
    if sorted(mine["framework"]["warnings"]) != sorted(theirs["framework"]["warnings"]):
        diffs.append(f"WARNINGS differ ({len(mine['framework']['warnings'])} vs {len(theirs['framework']['warnings'])})")

    for key in ("selectedClaim", "opposingClaim", "defenseLayerLabel", "attackEdgesCount",
                "attackersCount", "targetsCount", "contraryCandidatesCount", "layerMode",
                "layerMaxLevel"):
        if mine["meta"].get(key) != theirs["meta"].get(key):
            diffs.append(f"META.{key}: mine={mine['meta'].get(key)!r} theirs={theirs['meta'].get(key)!r}")
    return diffs


passed = failed = 0
for case in CASES:
    label = ", ".join(f"{k}={v}" for k, v in case.items())
    try:
        diffs = compare(case)
    except Exception as exc:
        print(f"[ERROR] {label}\n         {type(exc).__name__}: {exc}")
        failed += 1
        continue
    if diffs:
        failed += 1
        print(f"[DIFF]  {label}")
        for d in diffs:
            print(f"         - {d}")
    else:
        passed += 1
        print(f"[MATCH] {label}")

print(f"\n{passed} identical, {failed} differing, out of {len(CASES)} cases")
