#### PATCHARAKORN ####

"""End-to-end HTTP parity check: this FastAPI app vs mu_team's Node app, live, same database.

Where compare_with_mu_team.py diffs the graph service in-process, this hits both servers over
HTTP and compares the actual JSON responses — so it also covers routing, query parsing and
serialization.

    cd mu_work/Task4/Project_ABA && npm start                    # theirs on :3000
    cd internship/ABA_app && python -m uvicorn app.main:app --port 8000
    python benchmark/compare_http.py
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request

MINE = "http://127.0.0.1:8000"
THEIRS = "http://127.0.0.1:3000"

# Keys this app adds on top of theirs; excluded from the diff by design.
ADDED_KEYS = {"newAba", "translation"}


def get(base: str, path: str, params: dict | None = None):
    url = base + path + ("?" + urllib.parse.urlencode(params) if params else "")
    with urllib.request.urlopen(url, timeout=300) as r:
        return json.loads(r.read().decode("utf-8"))


def post(base: str, path: str, body: dict):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read().decode("utf-8"))


def canonical(value):
    """Order-insensitive canonical form: edge ids are positional in both apps, so lists of
    dicts are compared as sorted sets of their contents."""
    if isinstance(value, dict):
        return {k: canonical(v) for k, v in sorted(value.items()) if k not in ADDED_KEYS}
    if isinstance(value, list):
        return sorted((canonical(v) for v in value), key=lambda x: json.dumps(x, sort_keys=True, default=str))
    return value


def diff(label: str, mine, theirs) -> bool:
    if canonical(mine) == canonical(theirs):
        print(f"[MATCH] {label}")
        return True
    print(f"[DIFF]  {label}")
    if isinstance(mine, dict) and isinstance(theirs, dict):
        for key in sorted(set(mine) | set(theirs)):
            if key in ADDED_KEYS:
                continue
            if canonical(mine.get(key)) != canonical(theirs.get(key)):
                a, b = mine.get(key), theirs.get(key)
                sa = a if not hasattr(a, "__len__") or isinstance(a, str) else f"<{len(a)} items>"
                sb = b if not hasattr(b, "__len__") or isinstance(b, str) else f"<{len(b)} items>"
                print(f"         - {key}: mine={str(sa)[:90]} theirs={str(sb)[:90]}")
    return False


GRAPH_CASES = [
    {"topic": "check-in", "supporting": "easy_check-in", "sentiment": "All"},
    {"topic": "check-in", "supporting": "easy_check-in", "sentiment": "All", "layer_mode": "layer1"},
    {"topic": "staff", "supporting": "friendly_staff", "sentiment": "All"},
    {"topic": "staff", "supporting": "friendly_staff", "sentiment": "All", "layer_mode": "layer1"},
    {"topic": "staff", "supporting": "friendly_staff", "sentiment": "All", "show_all_contrary": "1"},
    {"topic": "price", "supporting": "value_for_money", "sentiment": "All"},
    {"topic": "price", "supporting": "value_for_money", "sentiment": "All", "layer_mode": "layer1"},
    {"topic": "check-out", "supporting": "possibility_for_late_check_out", "sentiment": "All"},
    {"topic": "check-out", "supporting": "possibility_for_late_check_out", "sentiment": "All", "layer_mode": "layer1"},
]

results = []

results.append(diff("GET /api/health", get(MINE, "/api/health"), get(THEIRS, "/api/health")))
results.append(diff("GET /api/topic-ratios", get(MINE, "/api/topic-ratios"), get(THEIRS, "/api/topic-ratios")))

for topic in ("staff", "check-in", "price", "check-out"):
    for sentiment in ("positive", "negative"):
        params = {"topic": topic, "sentiment": sentiment}
        results.append(diff(f"GET /api/review-data {topic}/{sentiment}",
                            get(MINE, "/api/review-data", params),
                            get(THEIRS, "/api/review-data", params)))

for case in GRAPH_CASES:
    label = "GET /api/aba-graph " + ", ".join(f"{k}={v}" for k, v in case.items())
    mine = get(MINE, "/api/aba-graph", case)
    theirs = get(THEIRS, "/api/aba-graph", case)
    results.append(diff(label, mine, theirs))

    # Evaluation is only compared on layer1 frameworks. On full layer2 frameworks NEITHER app
    # can produce an answer, for two separate reasons, both verified:
    #   * mu_team's Express backend calls express.json() with no limit option, so any payload
    #     over the 100kb default is rejected with PayloadTooLargeError (their README documents
    #     a JSON_BODY_LIMIT env var, but server.js never wires it up). The staff layer2
    #     payload is well over that.
    #   * py_arg instantiates every derivation as an argument before computing semantics, and
    #     does not finish on the staff layer2 framework (239 assumptions / 1338 rules) even for
    #     Grounded. Theirs would hang indefinitely; this one times out with a clear message.
    # That is a shared limitation of the approach, not a difference between the two ports.
    for semantics in ("Grounded", "Preferred") if case.get("layer_mode") == "layer1" else ():
        payload = {**mine["framework"]["payload"],
                   "semantics_specification": semantics, "strategy_specification": "Credulous"}
        results.append(diff(f"POST /api/pyarg/evaluate [{semantics}] {case['supporting']}",
                            post(MINE, "/api/pyarg/evaluate", payload),
                            post(THEIRS, "/api/pyarg/evaluate", payload)))

print(f"\n{sum(results)} identical, {len(results) - sum(results)} differing, out of {len(results)} checks")
