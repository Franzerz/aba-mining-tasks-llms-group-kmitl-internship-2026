# ABA App

A from-scratch rebuild of mu_team's `Task4/Project_ABA` web app, addressing the bug you flagged
("the graph is represented wrong with AA and ABA") and implementing your advisor's formal ABA
generalization (the multi-valued contrary function `C*` and the Translation Rule).

Same visual language as mu_team's site, different stack: **React + Tailwind (Vite) frontend,
FastAPI backend, no database.**

## What was wrong with mu_team's version (verified by reading their code directly)

mu_team's app built **two separate graphs from the same underlying data**: a hand-rolled,
heuristic "argument tree" (drawn to the screen) and a separately re-derived JSON payload (sent
to `py_arg` for evaluation). These could silently disagree — e.g. every assumption without a
real attacker got a synthetic self-contrary injected into the `py_arg` payload so `ABAF()`
wouldn't crash, but that synthetic contrary was **never drawn**, so the picture could show an
assumption as undefended while the framework actually being evaluated had a hidden contrary
for it. There's also no "argument" concept anywhere in their rendering — attack edges are
drawn directly between raw sentence-level atoms, never between constructed arguments — so the
picture is neither a faithful AA graph nor a complete ABA diagram, just an ad hoc hybrid.

**The fix here:** one pipeline, three inspectable stages, all derived from the same objects —
so the picture and the evaluated framework cannot diverge.

## The three views (your advisor's whiteboard pipeline, made visible)

| View | What it is | Where it comes from |
|---|---|---|
| **New ABA (C\*)** | The honest, pre-translation framework where one assumption may have **several** distinct contraries: `C*: A → P(L)` | `app/aba_builder.py` |
| **Reduced ABA (C)** | Standard flat ABA, one contrary per assumption — **what `py_arg` actually consumes** | `app/aba_translator.py` |
| **AA** | The Dung instantiation: constructed arguments + derived defeats | `py_arg`'s own `generate_af()` |

Why `C*` is needed: Task 3 finds **multiple different Propositions that each independently
attack the same Assumption**. Standard ABA can only name one contrary per assumption. An
earlier version of this app funneled every attacker through one synthetic placeholder atom —
functionally similar, but not an honest model. `C*` represents it directly, and the
**Translation Rule** (clone-splitting) reduces it back to something `py_arg` can evaluate:

```
assumption B with C*(B) = {C1, C2}      -->   assumptions B#C1, B#C2
rule  h <- B                                  C(B#C1) = C1,  C(B#C2) = C2
                                              rules  h <- B#C1,  h <- B#C2
```

Measured on real data — 25 of check-in's 48 assumptions genuinely have more than one contrary,
so this is not a hypothetical case.

### Bridging rules — an addition beyond the whiteboard example (**raise this with your advisor**)

The whiteboard example only covers `B` appearing in a rule **body**; its `C1`/`C2` were plain
atoms. In our data the contrary atoms are frequently **assumptions themselves** — a Proposition
that Task 3 voted as attacking A is itself an assumption with its own contrary. Splitting turns
the atom `B` into clones, so after translation nothing derives the bare atom `B` any more; every
contrary pointing at `B` becomes underivable, and **`py_arg` finds zero defeats — the attack
graph comes out completely empty.** (Measured before the fix on check-in: 54 distinct contrary
atoms, 0 of them derivable, 0 defeats.)

So the translator also emits, for every clone, a bridging rule:

```
B <- B#Ci        "B holds if it holds under any one of its lines of defense"
```

`B` is no longer an assumption after splitting (only its clones are), so this keeps the
framework flat. Effect, measured:

| aspect | clones | AA arguments | defeats before | defeats after |
|---|---|---|---|---|
| check-out | 14 | 15 | 0 | 5 |
| check-in | 142 | 148 | 0 | 94 |
| price | 448 | 461 | 0 | 400 |
| facility | 1657 | 1718 | 0 | 1411 |
| staff | 2184 | 2207 | 0 | 1889 |

This is a formulation choice worth confirming with your advisor, since it goes beyond what the
whiteboard covered.

## Architecture

- **Backend:** Python + FastAPI, calling `py_arg` (`python-argumentation` on PyPI, import name
  `py_arg`) **in-process** — no subprocess/JSON bridge like mu_team needed (they were bridging
  Python from Node.js; since this backend is already Python, that bridge disappears).
- **Data:** reads Task 1 and Task 3 gold files directly from `internship/ABA_mining/Dataset/` —
  no database, no export/import step. Always reflects what's on disk in the pipeline project.
- **Frontend:** React 18 + Tailwind, built with Vite. Cytoscape.js for graph rendering.
  FastAPI serves the built `frontend/dist/` with an SPA fallback so client-side routes survive
  a refresh.

### Pages

- **Home** — research dashboard: gold dataset size, topic coverage, sentiment split, pipeline.
- **Review** — mu_team's topic-card + dual positive/negative panel layout, showing each mined
  body literal, how often it occurs, its derived contrary, and the review spans it came from.
- **Graph Studio** — the three views above, with aspect/semantics selectors, extension sets,
  legend, and the human-voted Task 3 pairs.

## How the framework is built (`app/aba_builder.py`)

From each Task 3 gold row `(Original A, A, B, Vote)`:

- **Assumptions** = every distinct `A` and `B`.
- **`C*(A)`** = `{Original A}` (the auto-derived default) **∪** every `B` where `Vote == "Yes"` —
  i.e. the human-confirmed attackers become genuine additional contraries.
- **`C*(B)`** = `{have_evident_B}` (Task 2's convention for negative-sentiment literals).
- **Rules**: `good_{aspect} <- A`, `bad_{aspect} <- B`.

**Known limitation:** our freshly-regenerated Task 2 output (llama3.2) uses different literal
normalization than the literals baked into the Task 3 gold files (mu_team's original
extraction) — e.g. `easy_to_check_in` vs `easy_check-in` — so they can't be cross-referenced
yet. This builder is self-contained within Task 3's gold data (real, not fabricated) rather
than pulling in Task 2. Reconciling the two vocabularies is a follow-up.

## `py_arg` API notes

Verified directly against the installed package, not just by reading mu_team's usage:

- `ABAF(assumptions, rules, language, contraries)` — **enforces flatness itself** (raises
  `ValueError` if any assumption appears as a rule head) and requires every assumption to have
  a contrary. You don't need to hand-verify these; the constructor does it.
- `abaf.generate_af()` — the real ABA→AA (Dung) instantiation. For every assumption's contrary,
  it recursively constructs *every* possible argument (minimal assumption set) that derives it,
  then computes defeats between all constructed arguments. Every semantics function in `py_arg`
  calls this internally, so it's already the library's own single source of truth; we call it
  once and reuse the result.
- **Confirmed upstream bug:** in `py_arg.aba_classes.semantics.get_grounded_extensions` the
  *module* is named for grounded semantics, but the *function inside it* is literally named
  `get_preferred_extensions` (a typo in the installed package). This is why mu_team's
  `pyarg_runner.py` had to guess across several function names — not their bug, upstream.
  Wrapped and aliased correctly in `app/pyarg_service.py`.
- **Performance — measured, not assumed:** `grounded` uses a proper least-fixed-point algorithm
  (check-in 0.01s, staff 2.70s; always terminates, always exactly one extension). `stable` did
  not finish in 60+s on the 142-argument check-in framework and had to be killed — it's an
  unpruned binary-branching search over every AA-level argument. `admissible`/`conflict_free`
  enumerate the full assumption powerset. **`grounded` is the only semantics reliably fast at
  this scale**; the others are offered but wrapped in a 15s timeout
  (`EXTENSION_TIMEOUT_SECONDS`) that returns a clear error instead of hanging the server.

## Running it

```bash
cd internship/ABA_app
pip install -r requirements.txt

cd frontend && npm install && npm run build && cd ..
python -m uvicorn app.main:app --port 8000
```

Open `http://127.0.0.1:8000/`.

For frontend development with hot reload (two terminals):

```bash
python -m uvicorn app.main:app --port 8000    # terminal 1
cd frontend && npm run dev                     # terminal 2 -> http://localhost:5173
```
Vite proxies `/api` to the FastAPI server, so both run side by side.

## Verified

- All **five** aspects build, including facility (read from its `.xlsx` gold sheet):

  | aspect | assumptions (C\*) | clones (C) | AA args | defeats | gold pairs |
  |---|---|---|---|---|---|
  | check-out | 9 | 14 | 15 | 5 | 8 |
  | check-in | 48 | 142 | 148 | 94 | 252 |
  | price | 48 | 448 | 461 | 400 | 455 |
  | facility | 246 | 1657 | 1718 | 1411 | 12,060 |
  | staff | 295 | 2184 | 2207 | 1889 | 6,256 |

- Translator still reproduces the whiteboard's worked example exactly, and every framework
  passes `py_arg`'s own flatness/contrary validation.
- All API endpoints round-trip correctly (`/api/aspects`, `/api/graph/{aspect}`,
  `/api/dataset-summary`, `/api/topic-ratios`, `/api/review-data`), including a 400 on an
  invalid sentiment.
- Review-page numbers match the gold CSV directly (Staff 476 rows, Check-out 8).
- Vite production build succeeds; Home page visually confirmed rendering in a real browser.
- **Not yet visually confirmed:** the Review and Graph Studio pages in a browser. The API data
  behind both is verified, but please open them and report anything that looks off (node
  overlap in the Cytoscape canvas, layout issues) before calling the GUI work complete.

## Next steps

1. Open Review and Graph Studio in a browser and report anything visually wrong.
2. Confirm the bridging-rule formulation with your advisor (see above).
3. Reconcile Task 2's regenerated literals with Task 3's gold literals, so a fresh Task 2 run
   feeds straight into this builder instead of the Task-3-only construction.
4. Add the `contrary_v5`-style LLM-predicted attacks as a second toggle alongside gold votes, to
   compare what the reasoner concludes under gold vs. predicted attack relations.
