# ABA App

A rebuild of mu_team's `Task4/Project_ABA` web app that implements the advisor's ABA
generalization (the multi-valued contrary function `C*` and the Translation Rule) and follows the
construction in **CLAR 2025** (Racharak et al., *Towards Assumption-Based Argumentation Mining in
Hotel Reviews*, `internship/ABA_mining/Doc/CLAR 2025.pdf`).

Same visual language as mu_team's site, different stack: **React + Tailwind (Vite) frontend,
FastAPI backend, no database.**

## Why it was rebuilt

mu_team's app built **two separate graphs from the same data**: a heuristic "argument tree" for
the screen and a separately derived payload for `py_arg`. They could disagree — assumptions
without an attacker got a synthetic contrary in the `py_arg` payload that was never drawn. Here
there is one pipeline and three views of the same objects, so the picture and the evaluated
framework cannot diverge.

## The three views

| View | What it is | Built in |
|---|---|---|
| **Simple ABA** (default) | One node per guest point: a literal and the assumption it rests on are the two premises of one rule and always stand or fall together, so they are drawn as a single node named after the assumption | `frontend/src/graphElements.js` |
| **New ABA (C\*)** | Pre-translation: one assumption may have **several** contraries, `C*: A → P(L)` (CLAR 2025, Sect. 2) | `app/aba_builder.py` |
| **Reduced ABA (C)** | Standard flat ABA, one contrary per assumption — what `py_arg` evaluates | `app/aba_translator.py` |

The **AA** view was removed: the work is about ABA, and with every attacker an observed fact the
Dung instantiation only restated the reduced framework. The AA data still backs the "AA arguments"
and "Defeats" figures, and `py_arg`'s `generate_af()` still runs — it is the picture that is gone.

Translation Rule (clone-splitting), from the advisor's whiteboard:

```
assumption B with C*(B) = {C1, C2}      -->   assumptions B#C1, B#C2
rule  h <- B                                  C(B#C1) = C1,  C(B#C2) = C2
                                              rules  h <- B#C1,  h <- B#C2
```

## How a topic's framework is built (`app/aba_builder.py`)

As in CLAR 2025 (Fig. 2, Fig. 7, Sect. 4.2):

```
proposition (observed fact)   easy_check-in                       f_A:  A <-
assumption                    no_evident_not_easy_check-in
contrary of that assumption   customer_wait_10_minute_for_check-in   (a negative literal)

good_X <- A, no_evident_not_A          C*(no_evident_not_A) = { B : "is B contrary?" = Yes }
bad_X  <- B, have_evident_B            C*(have_evident_B)   = { A : "is A contrary?" = Yes }
```

- The first question is Task 3's **Contrary(P)Body(N)** category, the second is
  **Contrary(N)Body(P)**.
- **Contrary edges** are drawn as the **attack** they stand for: a dashed arrow **from** the
  attacking literal **to** the assumption it contradicts. That is how the gold sheets read — in a
  Contrary(N)Body(P) sheet, column D (the praise literal) attacks column B (the complaint
  assumption) — it matches CLAR 2025's Fig. 7 ("contrary of"), and it runs the same way as the
  defeats py_arg derives. The contrary *function* C\*(A) points the other way, from an assumption
  to its contraries; the picture shows the relation, not the function.
- **Literals mined with both sentiments:** three literals appear in both positive and negative
  spans (facility: `not_big_hotel`; location: `close_to_airport`, `proximity_to_airport`).
  - Such a literal is one fact with two assumptions (`no_evident_not_X` and `have_evident_X`),
    and it supports both claims.
  - The focus picker lists it once, under the negative side.

### Default contraries

CLAR 2025 defines the contrary function as **total and non-empty**, and `py_arg` refuses a
framework otherwise, so an assumption nothing attacks still needs one contrary. It gets the
default one, the negation of what it assumes:

```
no_evident_not_A  ->  have_evident_not_A        have_evident_B  ->  no_evident_B
```

No rule derives a default contrary, so it never attacks anything, and it is never drawn. It still
exists in the framework and in the API response (`atom_roles: "placeholder"`).

**Exception:** where a topic mined both `X` and `not_X`, the default name `no_evident_not_X` *is*
the real assumption behind `X`.
- This happens twice: staff (`kind_staff` / `not_kind_staff`) and facility (`easy_parking` /
  `not_easy_parking`).
- Using the same atom would create an attack no annotator voted for, so those two defaults are
  named `…_default` instead.

### Bridging rules — only where needed (**raise with the advisor**)

If an assumption `B` is itself someone's contrary, splitting it leaves nothing that derives `B`,
and those attacks vanish. For such a `B` the translator adds `B <- B#Ci` per clone, which keeps
the framework flat. With the current builder no contrary is an assumption (contraries are
observed facts, as in the paper), so **none are emitted**. Adding them unconditionally, as an
earlier version did, only doubled the rule count.

### Deviation from the paper (**raise with the advisor**)

CLAR 2025 writes **one rule per review span**, with every support and assumption of that span in
the body. This app writes **one rule per literal**. The drawn graph is the same either way, but
the Translation Rule replicates a rule once per combination of clones in its body. With
per-span rules that explodes: on staff, about **11.4 billion** reduced rules from the voted positive side alone, against 2,368 today,
and a matching number of arguments. The semantic difference is that an attack knocks out one
point of a review, not the whole review's argument.

### Observation for the advisor

Every attacker is an observed fact, and facts cannot be attacked, so an attacked assumption can
never be reinstated. In every topic the grounded extension is exactly the set of unattacked
assumptions, and the other semantics can only agree with it. The attack relation carries all
the information; the choice of semantics carries none.

## Which hotel (`app/datasets.py`)

The site shows one hotel's data at a time, chosen from the picker in the top nav and carried in
the URL as `?hotel=`. La Veranda is the default, so existing links keep working unchanged.

Each hotel is one `HotelDataset` entry saying where its data lives and in which shape. The two
present ones are genuinely different, and the app never blurs them:

| | `la-veranda` | `green-harbor` |
|---|---|---|
| spans | the annotated gold CSV | `Dataset/green_new_hotel.csv` (a Task 1 run's output) |
| literals | human-annotated, in `Body N` / `Cont. Body N` column pairs | only in the vote files |
| Task 3 votes | the numbered Silver/Gold workbooks | `Task3/green_new_hotel/Task3_Vote/<Topic> - Contrary(X)Body(Y).csv` |
| topics voted | 5, `Contrary(P)Body(N)` only | 9, **both** categories |
| `Contrary(N)Body(P)` | the PN votes read symmetrically (see below) | its own answers, voted separately |
| other topics | LLM-predicted contraries | none |

Two consequences worth knowing before reading a number off the screen:

- **Green Harbor's NP votes are not a mirror of its PN votes.** They disagree on about 29% of
  pairs, so each side is read from its own file. Mirroring them, as the gold workbooks do, would
  silently overwrite a third of the annotators' answers.
- **A Green Harbor literal has no occurrence count.** Its vote files name a literal but not the
  spans it came from, so the Review page shows `—` in the Count column rather than a `0` that
  would read as "never appeared". The `has_counts` flag on `/api/review-data` drives this.

### Adding another hotel

Add one entry to `DATASETS` in `app/datasets.py`; nothing else needs touching, because every
reader takes the hotel key and the picker is built from the registry.

```python
SEASIDE = HotelDataset(
    key="seaside", label="Seaside Hotel", blurb="…",
    spans_csv=ABA_MINING_ROOT / "Dataset" / "seaside.csv",
    spans_format=TASK1_SEQUENCE,          # or GOLD_WIDE if its literals are annotated
    task3_format=VOTE_DIR,                # or GOLD_WORKBOOKS
    task3_dir=TASK3_GOLD_DIR / "seaside",
    vote_dir=TASK3_GOLD_DIR / "seaside" / "Task3_Vote",
)
```

`spans_format` and `task3_format` are the only real choices: they say which reader parses the
files. A layout matching neither needs a new branch in `review_data._read_all` (spans) or
`contrary_sources` (votes) — add the format constant alongside the existing ones rather than
special-casing the hotel key anywhere.

### Claim atoms and colliding literals

A mined literal can be spelled exactly like the claim it backs: Green Harbor's price votes
contain the literal `good_price`, its location votes `good_location`. They are not the same
proposition — one is "a guest said the price was good", the other is the claim derived from such
evidence — and treating them as one atom breaks the framework twice: the claim becomes a fact, so
it holds whatever attacks it, and its support rule turns into the self-loop
`good_price <- good_price, …`, a rule cycle `fast_af` rejects outright.

So `_head_atom` names the claim only once every atom the data brings is known, and appends
`_claim` on a collision (`good_price_claim`). The literal keeps its own name, because that is
data; the claim is the app's own atom, so it is the one that moves. Topics with no collision are
unaffected — La Veranda's frameworks are unchanged down to the last defeat.

## Where the contraries come from (`app/contrary_sources.py`)

**Votes win wherever they exist, on both sides.** Five topics were annotated — check-in,
check-out, price, staff, facility — and the annotators answered one question per literal PAIR, on
the Contrary(P)Body(N) sheet: *is B a contrary of A?* Contrary(N)Body(P) asks the same pair the
other way round (assumption `have_evident_B`, candidate `A`), so it carries the same answer. The
gold workbooks' own Contrary(N)Body(P) sheet is filled in by a spreadsheet formula that does
exactly this, and mirroring the votes reproduces it row for row — verified on price against the
sheet, and 100% pair coverage on all five topics.

| Source | Where | Shown as |
|---|---|---|
| **Human votes** (Task 3 gold) | check-in, check-out, price, staff, facility — **both** contrary sets | brown dashed edges, "Human-voted" |
| **LLM predictions** — `llama3.2` + `contrary_v1` | every *other* topic, both contrary sets | **blue** dashed edges, "(LLM)", "LLM contraries" chips on Home/Review |

Both sets of a voted topic therefore hold the same number of "Yes" answers (price 400, staff
1,889, facility 1,411, check-in 94, check-out 5); they differ in which assumptions those answers
attack, so the two frameworks are not the same size.

- **Why `llama3.2` + `contrary_v1`:** it is the best `llama3.2` setting in the Task 3 evaluation
  (`ABA_mining/outputs/eval/task3_server/`).
  - Estimated F1 over the five voted topics is **0.50** (precision 0.38, recall 0.74), against
    0.34 for answering "Yes" to everything.
  - On Contrary(P)Body(N) it says "Yes" about twice as often as the annotators do, so **most
    predicted contraries on that side are likely wrong**.
  - On Contrary(N)Body(P) it is far more cautious (1–20% "Yes" on the larger topics). It was
    never scored against the mirrored votes; on price it said Yes 91 times where the annotators'
    answers give 400, so the LLM-only topics' complaint side is probably badly under-attacked.
  - The UI never presents these predictions as votes, and the plain-language summary says so.
- **One topic, one source:** a topic is entirely human-voted or entirely LLM-predicted, never a
  mix of the two.
- **Contrary set selector:** Graph Studio shows **one contrary set at a time**, Contrary(P)Body(N)
  by default, with Contrary(N)Body(P) and "Both sets" one click away (`?contraries=pn|np|both`).
  - Only the sets a topic actually has are offered, and a link carrying a set the topic lacks
    falls back to Contrary(P)Body(N).
  - Picking a set moves the head claim to the side it attacks — Contrary(P)Body(N) to `good_X`,
    Contrary(N)Body(P) to `bad_X` — since the other side would show support edges and no attacks.
  - With both sets, the two kinds of attack run in opposite directions over the same literals and
    the picture turns into a criss-cross.
  - The server builds and evaluates the framework with only the chosen set. The side that is
    left out keeps its default contraries, so the extensions always match the picture.
  - The Review page's per-literal popup defaults to both sets on an LLM topic, since one
    literal's neighbourhood stays small, and to Contrary(P)Body(N) — the voted set — on a voted one.
- **Topics with no possible pair:** a topic mined with only one sentiment (booking-issue: six
  complaints, no praise) has no pair to ask about.
  - It still gets a framework, built from its gold-span literals, with every contrary a
    default contrary: nothing in it can contradict anything.
  - The UI marks it "no pairs".
- **Only complete runs are used:** a half-finished run would make assumptions look unattacked
  when they simply haven't been asked about. Topics appear as their runs complete.

Regenerating the LLM contraries (from `internship/ABA_mining`):

```bash
python build_task3_llm_pairs.py      # full pair sets -> Dataset/Task3/llm_pairs/
bash run_task3_llm_pairs.sh          # llama3.2 over them -> outputs/task3_llm_pairs/ (resumable)
```

## Large frameworks (`app/fast_af.py`)

`py_arg` compares every argument with every other argument, and its grounded semantics is a
naive fixed point. On a synthetic framework the size of `food` (~12,800 arguments) that took
**32 s + 119 s**; `room` is several times larger.

- **The fast path:** `fast_af.py` builds the same arguments (same names) and the same defeats
  with indexes, and computes grounded semantics with the standard labelling algorithm. At room
  size (~112k arguments) this takes about **1.3 s**.
- **Proof it matches:** `tests/test_fast_af.py` checks it is **identical to `py_arg`** on every
  human-voted topic plus handmade edge cases (underivable atoms, reinstatement, self-attack).
- **When it's used:** only for frameworks with at least 2,500 assumptions. Every human-voted
  topic is still evaluated by `py_arg` itself. The response says which engine ran.
- **Big frameworks only offer `grounded`:** `py_arg` enumerates the other semantics
  exhaustively, so they could never finish at that size.
- **Serving it:** responses are cached until the topic's data changes, and sent gzip-compressed
  (food: 7.7 MB → 0.5 MB).
- **Drawing it:** views above 12,000 nodes and links are not drawn; the page asks for a
  **Focus on** literal instead.
- **Partial responses:** for frameworks this large, `/api/graph/{topic}` is marked `partial`.
  - It sends the New ABA view whole, but the reduced and AA views only for
    `?focus=<literal>`, as a superset of what the browser's focus filter keeps.
  - Checked exhaustively on food and location (every focus literal, both views): identical to
    filtering the full data.

## Speed

| What was slow | Cause | Now |
|---|---|---|
| Home / Review / Studio lists: **2.5 s per request** | a size-capped cache over 18 prediction sets thrashed, re-reading room's files every time | one cache entry per set: ~50 ms |
| First request after startup: **12.8 s** | openpyxl opening facility's 7-sheet gold workbook (~14 s) and 250 MB prediction CSVs (each row stores its whole prompt) | slim CSV copies in `.cache/` keyed by file size + mtime, and everything pre-built in a background thread at startup: ~1 s |
| Drawing a big view: **~10 s frozen tab** (staff reduced view) | dagre's crossing minimisation | dagre up to 4,000 elements (≤ ~1.5 s); above that, a linear-time column layout (~40 ms) that keeps each literal next to its own assumption as in CLAR 2025 Fig. 7, with faint edges, hidden while panning |
| First page load | Cytoscape (615 KB) in the main bundle | loaded only when a graph opens (main bundle 185 KB) |

`.cache/` is safe to delete; it is rebuilt on demand.

## Graph Studio controls

- **Head claim** — one claim at a time, `good_X` or `bad_X` (`?head=`). Drawing both at once put
  every literal of the topic on screen and buried the picture in edges; `good_X` is the default.
- **Focus on** — one literal's neighbourhood, or the whole framework (`?focus=`). It follows the
  head claim: with `good_X` chosen it lists only the literals that back `good_X`, since the other
  side is not on screen. Switching claim drops a focus that no longer belongs to it, and a link
  naming a literal from either side (the Review page's Graph button) moves the claim to match.
  The few literals mined with both sentiments appear under both claims.
- **Contrary set** — see above; only the sets the topic has are offered.

An assumption nothing attacks still has a contrary, because the contrary function is total, but
that **default contrary** is derived by nothing and attacks nothing, so it is never drawn. It
still exists in the framework and in the API response (`atom_roles: "placeholder"`); it is only
left out of the picture and the legend.

## Merging near-identical points

Topics repeat themselves: check-in alone has `easy_check-in`, `easy_check-in_at_3am`,
`easy_self_check-in`, `easy_late_night_check-in`, `fast_check-in`, `quick_check-in`… **Merge
similar** folds names within an edit distance of each other into one node (`easy_check-in +8`),
so the picture shows themes instead of phrasings.

- **Levels** are a share of the longer name: Near-duplicates (0.8), Similar wording (0.65), Loose
  (0.55). At 0.65 the nine names above become one node; at 0.55 unrelated `staff_…` points start
  joining them.
- Names are compared **after** stripping `no_evident_not_` / `have_evident_`, so a level means the
  same thing in every view, and only same-role nodes are ever compared.
- Single-linkage union-find, with names compared only when they share a word; the slowest topic
  (room, 638 nodes) merges in under 0.4 s.
- It is **spelling, not meaning**: it groups `easy_check-in` with `easy_self_check-in` and never
  with `quick_check-in`. Grouping by meaning needs embeddings.
- Merging is a drawing aid applied **last**, after the head and focus filters. It never changes the
  framework, so the counts, the extensions and "What holds up" stay at full resolution.

## What holds up (accepted points)

A sidebar panel for someone running the hotel rather than reading a framework: of the points about
the selected claim, which ones nothing contradicts. Praise and complaints are counted separately
("17 of 42 hold"), the surviving points are listed with their review counts, and the defeated ones
are behind a "knocked out by a guest saying the opposite" toggle. Clicking any point focuses the
graph on it.

It is read off the extension the server computed — an assumption is accepted when every clone of it
is in the extension — so it can never disagree with the picture. Checked against the grounded
extension on all five voted topics (check-in 23, price 14, staff 37, facility 116, check-out 4).

## Review page graphs

Every supporting literal has a **Graph** button that opens its neighbourhood in a popup:
- **What it shows:** the literal, its assumption, the claim it backs, what attacks it and what
  it attacks, in any of the three views, with a link into Graph Studio.
- **Linkable:** the popup's state is in the URL, e.g. `/review?topic=Room&graph=clean_room&side=Positive`.
- **Disabled rows:** literals the topic's framework doesn't contain have the button disabled.
  The voted Task 3 sets leave out a few span literals, e.g. two in price.

The table has two columns, **Supporting** and **Count**, plus the Graph button. The gold file's
Cont. Body field — CLAR 2025's "contrary literal", which in the framework is the literal's
**assumption** (`no_evident_not_X` / `have_evident_X`) — moved into the expanded row, next to the
head claim, rather than taking a column of its own.

## Plain-language explanation (`app/llm_service.py`, `frontend/src/components/LlmSummary.jsx`)

Under the graph, **What this graph says** explains the current picture for a non-specialist:

- **At a glance:** praise and complaint points, backing links, attacks.
- **Summary bullets:** always available, counted from the view on screen.
- **Explain with AI:** has a local Ollama model rewrite the summary.
- **What it sees:** the facts sent to the model are the numbers on screen, with all framework
  vocabulary removed. LLM-predicted clashes are flagged as unchecked.

The model shares Ollama's runner with the Task 3 batch (same `num_ctx`). Ollama here runs one
request at a time.
- **Timing:** an explanation takes ~5 s when the model is idle, and waits minutes while a batch
  run is using it.
- **Accuracy:** a 3B model still misreads parts of the picture now and then, so the text is a
  reading aid, not a source of truth.

## Architecture

- **Backend:** FastAPI calling `py_arg` (`python-argumentation`) in-process.
- **Data:** read straight from `internship/ABA_mining/` (Task 1/2 gold, Task 3 gold, LLM pair
  sets and predictions). There is no import step.
  - The Task 1/2 gold is **Original ABA Dataset for Version 2 [Oct 23, 2025]** (`gold_csv` in
    `ABA_mining/configs/paths.yaml`).
  - Reviews are counted by its `ID` column: IDs run 1–785, but 520 and 563 have no rows, so
    the file holds **783 reviews**. In the raw hotel file (A-1) each of those two is a single
    off-topic row whose text is a spreadsheet error.
  - 744 of the 783 reviews give at least one hotel opinion (2,078 opinion rows; 406 off-topic
    rows and 1 empty row are left out).
  - The unvoted topics' literal sets (and so their LLM pair sets) come from this file too.
    `build_task3_llm_pairs.py` keeps a pair's ID across rebuilds, so a rebuild only sends new
    pairs to the model.
- **Frontend:** React 18 + Tailwind (Vite), Cytoscape.js with dagre (ABA views) and fcose (AA).
  FastAPI serves `frontend/dist/` with an SPA fallback.

Pages:
- **Home** — dataset figures and **What guests talk about**: for each topic, how many of the
  783 reviews mention it and how many praised or complained (hover a bar for the numbers), plus
  where its framework's contraries come from.
  - An *opinion* is one review's praise or complaint about one topic (a row of the gold file,
    the pipeline's "span").
  - The topic statistics count *reviews*: a review that both praises and complains about a
    topic counts on both sides.
- **Review** — mu_team's topic cards plus dual positive/negative literal tables.
- **Graph Studio**:
  - the three views, head-claim and focus filters, and the default-contrary toggle;
  - the explanation panel, legend and extensions;
  - where the contraries came from, with a sample of pairs.

## `py_arg` notes (verified against the installed package)

- `ABAF(...)` enforces flatness and requires every assumption to have a contrary.
- In `py_arg.aba_classes.semantics.get_grounded_extensions` the function is misnamed
  `get_preferred_extensions` (upstream typo); `app/pyarg_service.py` wraps it under the right
  name.
- `grounded` is the only semantics that finishes at this scale; the others run under a 15 s
  timeout for small frameworks and are refused for large ones.

## Running it

```bash
cd internship/ABA_app
pip install -r requirements.txt
cd frontend && npm install && npm run build && cd ..
python -m uvicorn app.main:app --port 8000        # http://127.0.0.1:8000/
python -m unittest tests.test_fast_af -v          # fast path == py_arg
```

Frontend development with hot reload: run uvicorn as above and `npm run dev` in `frontend/`
(Vite proxies `/api`).

## Current numbers (grounded semantics)

Each topic with the contrary sets it actually has: the human-voted ones with their single voted
set, Contrary(P)Body(N), and the LLM ones with both (`?contraries=both`). `human` = Task 3 votes;
`llm` = llama3.2 + contrary_v1 predictions.

| topic | pos side / neg side | "Yes" rate pos / neg | assumptions (C\*) | clones (C) | AA arguments | defeats | grounded | engine |
|---|---|---|---|---|---|---|---|---|
| check-out | human / — | 62% / — | 9 | 9 | 10 | 5 | 4 | py_arg |
| check-in | human / — | 37% / — | 48 | 117 | 123 | 94 | 23 | py_arg |
| price | human / — | 88% / — | 48 | 414 | 427 | 400 | 14 | py_arg |
| facility | human / — | 12% / — | 247 | 1,527 | 1,588 | 1,411 | 116 | py_arg |
| staff | human / — | 30% / — | 295 | 1,926 | 1,949 | 1,889 | 37 | py_arg |
| taxi-issue | llm / llm | 100% / 83% | 7 | 12 | 19 | 11 | 1 | py_arg |
| booking-issue | no pairs | — | 5 | 5 | 5 | 0 | 5 | py_arg |
| food | llm / llm | 59% / 8% | 281 | 7,728 | 7,887 | 7,728 | 0 | indexed |
| location | llm / llm | 28% / 8% | 560 | 19,356 | 19,848 | 19,335 | 21 | indexed |
| room | llm / llm | 34% / 2% | 636 | 35,761 | 36,231 | 35,715 | 46 | indexed |

- **Grounded extension:** in every topic it is exactly the set of unattacked assumptions (see
  the observation above). In food every assumption is attacked, so it is empty.
- **Serving:** a cached response is served in ~0.05 s.
- **Partial-response check:** every focus literal of food, location and room, in both views and
  for both the default set and "Both sets", is identical to filtering the full data
  (5,884 checks). Re-checked after the voted topics stopped reading LLM predictions: staff, 590
  checks, 0 mismatches.

## Open points

1. Advisor: bridging-rule formulation, per-literal rules vs. the paper's per-span rules, and
   the observation that the semantics add nothing when every attacker is a fact.
2. The LLM contraries are low-precision; a stronger model (gemma4:31b scored F1 0.67 on the
   five voted topics) would give a much better picture if run on the server.
