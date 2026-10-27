# Section 6, Experimental Setup — with the variability TODO resolved

This file answers

> `[TODO: Add standard deviations, or bootstrap confidence intervals over reviews; ...]`

for **both Task 2 and Task 3**. The first half is paper text, ready to paste. The second half is
working notes that are **not** part of the paper: what the method means in plain words, the
measurements behind it, and how to reproduce them.

The other TODO in Section 6 (re-scoring T1.2 at a single θ) is a separate task, left untouched.

**Sources.** Task 2 from `outputs/Task2_new_eval/task2/gt` (the *v2* evaluator: 6 models × 6
prompt versions × 3 runs). Task 3 from `outputs/eval/task3_server` (4 models × 8 prompts × 5
topics × 3 runs — the three runs are the `Test 1/2/3` rows of each `EVAL_SUMMARY.txt`).

The older `outputs/eval/task2` tree uses a superseded metric on a different scale (micro F1 ≈
0.58 where v2 gives ≈ 0.84 for the same run) and is **not** the source of Table 5; it is ignored.

---

## Paper text

### 6 Experimental Setup

**Variability.** Decoding is deterministic, so repeating a configuration changes little. Over
the three runs of each Task 2 configuration the micro F1 moves by at most 0.004 (standard
deviation ≤ 0.0042), and 118 of 144 configuration/metric pairs have a standard deviation of
exactly zero; head-matching accuracy is 0.9716 in every run of every configuration. On Task 3
the picture is the same: 124 of 151 configurations have an identical F1 in all three runs, and
the largest standard deviation is 0.0118. Reporting run-to-run standard deviations would
therefore print a column of near-zeros and suggest a precision the data does not support.

The uncertainty that matters is instead *which items the corpus happens to contain*. We report
95% bootstrap confidence intervals (B = 2000, percentile method), resampling with replacement
the unit that was sampled: **reviews** for Task 2, and **candidate pairs** for Task 3. For
Task 2 reviews rather than instances are resampled because the instances mined from one review
share its text and its annotator, so resampling instances would treat dependent observations as
independent and understate the interval. In both tasks the score is a function of summed
per-item counts, so a resample is a re-summation of counts the evaluator already records and
needs no further model calls.

Over 740 reviews the Task 2 intervals have a half-width of 0.008–0.025; Qwen-2.5's best
configuration scores 0.914 [0.907, 0.922], while Llama 4 Scout v4 scores 0.517 [0.493, 0.542].
On Task 3 the interval width is governed by the size of the topic's pair set, and differs by an
order of magnitude across topics: the median half-width is 0.021 for Staff (6,256 pairs) and
0.024 for Facility (11,686), but 0.084 for Check-in (252) and **0.25 for Check-out, which has
only 8 pairs**. Check-out F1 is therefore not interpretable on its own, and an unweighted mean
over topics inherits that noise; we report a pair-weighted average alongside it.

### Suggested table caption addition

> Intervals are 95% percentile bootstrap (B = 2000) over reviews (Task 2) or candidate pairs
> (Task 3); see Section 6. Entries read `point [low, high]`.

---

## Working notes (not part of the paper)

### What a bootstrap confidence interval is

The score in the table is computed on **one particular sample** — these 740 reviews, or these
455 price pairs. Had the sample been slightly different, the score would have come out slightly
different. A confidence interval answers: *by how much?*

We cannot collect more data, so we imitate doing so with what we have:

1. Draw the same number of items **at random, with replacement**, from the items we have. "With
   replacement" means an item can be drawn twice, or not at all — that is what makes each draw a
   slightly different sample.
2. Recompute the score on that imaginary sample.
3. Repeat 2,000 times → 2,000 scores.
4. Sort them, take the 2.5th and 97.5th percentile. That range is the **95% confidence
   interval**.

Narrow interval = the score is stable. Wide interval = it depends heavily on which items you
happened to get. "Bootstrap" is just the name for resampling your own data instead of collecting
more.

**Why the unit differs between tasks.** A Task 2 instance belongs to a review, and the instances
of one review are not independent (same text, same writer, same annotator) — so a whole review
is drawn in or out together. A Task 3 item is a candidate pair, and pairs are not grouped by
review in the same way, so the pair is the unit there.

**Why this replaces the standard deviation.** The standard deviation across three runs measures
how much the *model* wobbles when asked again. At temperature 0 it barely wobbles, so that
number is near zero and tells a reader nothing. The bootstrap measures how much the *sample*
could have differed, which is the real source of uncertainty, and is far larger.

### Formulas

Let an *item* be a Task 2 instance or a Task 3 candidate pair, and write `TP_i`, `FP_i`, `FN_i`
for the counts the evaluator records for item `i`.

**Micro precision, recall, F1** — sum the counts first, divide once:

```
P  = Σ TP_i / (Σ TP_i + Σ FP_i)
R  = Σ TP_i / (Σ TP_i + Σ FN_i)
F1 = 2PR / (P + R)
```

This is why a bootstrap is cheap: the score depends on the data only through these three sums,
so re-scoring a resample is a re-addition, not a re-match.

**Sample standard deviation over the R = 3 runs** of one configuration, with `x_r` the score of
run `r`:

```
mean = (1/R) Σ_r x_r
sd   = sqrt( Σ_r (x_r - mean)^2 / (R - 1) )
```

`R - 1` (not `R`) because the mean is estimated from the same three numbers.

**Bootstrap percentile interval.** With `N` items in the sample and `B = 2000` resamples:

```
for b = 1 .. B:
    draw indices j_1 .. j_N uniformly from {1..N} with replacement
    score_b = F1( Σ_k TP_{j_k}, Σ_k FP_{j_k}, Σ_k FN_{j_k} )

CI_95 = [ percentile_2.5(score_1..score_B),  percentile_97.5(score_1..score_B) ]
SE    = sample standard deviation of score_1..score_B
```

For Task 2 the index set is the **reviews** (`N = 740`), and all instances of a drawn review are
taken with it. For Task 3 it is the **candidate pairs** (`N` = the topic's pair count). Because
F1 depends only on the summed counts, Task 3's resample is drawn equivalently as one multinomial
over the four outcome classes:

```
(TP*, TN*, FP*, FN*) ~ Multinomial( N, (TP, TN, FP, FN) / N )
```

**Why interval width tracks sample size.** The standard error of a proportion estimated from `N`
items falls as `1/sqrt(N)`. Going from Check-out's 8 pairs to Staff's 6,256 is a factor of 782 in
`N`, i.e. about `sqrt(782) ≈ 28` in precision — which is why the median half-width drops from
±0.25 to ±0.021.

### Measurement 1 — Task 2, run-to-run standard deviation

Largest standard deviation over any configuration and metric: **0.00414**. 118 of 144
configuration/metric pairs are exactly 0.00000. Head accuracy is 0.9716 everywhere. Every
non-zero value belongs to `llama4_scout` or `llama3.2`; `gemma4_31b` and `llama3.3_70b` are
bit-stable across runs.

### Measurement 2 — Task 2, bootstrap over reviews (run 1, 740 reviews)

Micro P/R/F1 over supports and assumptions, as in Table 5. Best prompt per model in bold.

| model | ver | Precision | Recall | F1 |
|---|---|---|---|---|
| gemma4_31b | v1 | 0.934 [0.925, 0.943] | 0.904 [0.891, 0.915] | **0.919 [0.910, 0.927]** |
| gemma4_31b | v3 | 0.951 [0.943, 0.959] | 0.795 [0.781, 0.810] | 0.866 [0.856, 0.876] |
| gemma4_31b | v5 | 0.949 [0.941, 0.956] | 0.861 [0.848, 0.875] | 0.903 [0.894, 0.912] |
| gemma4_31b | v6 | 0.949 [0.941, 0.956] | 0.843 [0.829, 0.857] | 0.893 [0.884, 0.902] |
| llama3.2 | v1 | 0.838 [0.826, 0.849] | 0.835 [0.821, 0.850] | **0.837 [0.827, 0.847]** |
| llama3.2 | v2 | 0.917 [0.909, 0.926] | 0.757 [0.741, 0.774] | 0.829 [0.819, 0.841] |
| llama3.2 | v3 | 0.870 [0.858, 0.883] | 0.678 [0.664, 0.692] | 0.762 [0.752, 0.773] |
| llama3.2 | v4 | 0.800 [0.786, 0.812] | 0.835 [0.821, 0.849] | 0.817 [0.806, 0.827] |
| llama3.2 | v5 | 0.645 [0.630, 0.660] | 0.744 [0.729, 0.758] | 0.691 [0.679, 0.702] |
| llama3.2 | v6 | 0.886 [0.874, 0.898] | 0.567 [0.551, 0.584] | 0.691 [0.678, 0.705] |
| llama3.3_70b | v1 | 0.872 [0.861, 0.881] | 0.904 [0.893, 0.915] | **0.888 [0.880, 0.896]** |
| llama3.3_70b | v2 | 0.878 [0.867, 0.889] | 0.827 [0.813, 0.842] | 0.852 [0.842, 0.862] |
| llama3.3_70b | v3 | 0.909 [0.896, 0.921] | 0.780 [0.765, 0.796] | 0.840 [0.829, 0.851] |
| llama3.3_70b | v4 | 0.880 [0.867, 0.891] | 0.800 [0.784, 0.816] | 0.838 [0.827, 0.848] |
| llama3.3_70b | v5 | 0.879 [0.869, 0.890] | 0.886 [0.873, 0.898] | 0.883 [0.874, 0.891] |
| llama3.3_70b | v6 | 0.931 [0.922, 0.940] | 0.845 [0.831, 0.857] | 0.886 [0.877, 0.895] |
| llama4_scout | v1 | 0.912 [0.901, 0.921] | 0.715 [0.694, 0.738] | 0.802 [0.787, 0.816] |
| llama4_scout | v2 | 0.929 [0.920, 0.938] | 0.730 [0.707, 0.753] | **0.818 [0.803, 0.833]** |
| llama4_scout | v3 | 0.936 [0.927, 0.944] | 0.718 [0.699, 0.736] | 0.812 [0.800, 0.825] |
| llama4_scout | v4 | 0.916 [0.902, 0.930] | 0.361 [0.338, 0.385] | 0.517 [0.493, 0.542] |
| llama4_scout | v5 | 0.915 [0.904, 0.926] | 0.684 [0.660, 0.706] | 0.783 [0.767, 0.798] |
| llama4_scout | v6 | 0.942 [0.935, 0.950] | 0.708 [0.689, 0.726] | 0.809 [0.796, 0.821] |
| qwen3.8_27b | v1 | 0.936 [0.929, 0.944] | 0.893 [0.881, 0.904] | **0.914 [0.907, 0.922]** |
| qwen3.8_27b | v2 | 0.933 [0.924, 0.942] | 0.831 [0.816, 0.844] | 0.879 [0.869, 0.888] |
| qwen3.8_27b | v3 | 0.943 [0.934, 0.951] | 0.786 [0.772, 0.800] | 0.857 [0.847, 0.867] |
| qwen3.8_27b | v4 | 0.933 [0.924, 0.942] | 0.841 [0.826, 0.855] | 0.884 [0.874, 0.894] |
| qwen3.8_27b | v5 | 0.934 [0.925, 0.942] | 0.877 [0.864, 0.889] | 0.905 [0.896, 0.913] |
| qwen3.8_27b | v6 | 0.942 [0.933, 0.950] | 0.844 [0.830, 0.856] | 0.890 [0.881, 0.898] |

Bootstrap standard error across all cells: up to 0.0127.

### Measurement 3 — Task 3, run-to-run spread and bootstrap over pairs

**Scope: `Contrary(P)Body(N)` (G1) only.** This is not a simplification but what the runs
cover. Of the four candidate-pair groups, G1 has 135 scored configurations across all four
models and all five topics; **G2 (`Contrary(N)Body(P)`) was never run** — no prediction files
exist for it; and G3/G4 exist for `llama3.2` on `facility` alone (8 prompts each). Intervals for
the G2 columns and the "Same (G3,G4)" column of Table 6 therefore cannot be produced from the
current outputs.

**Run-to-run.** 151 configurations (4 models × 8 prompts × 5 topics, `Contrary(P)Body(N)`);
124 have an identical F1 in all three runs; the largest standard deviation is **0.01184**.

**Bootstrap.** The interval width is set almost entirely by how many candidate pairs the topic
has — this is the headline result for Table 6:

| topic | pairs | median CI half-width | worst |
|---|---|---|---|
| Check-out | 8 | **±0.2500** | ±0.5000 |
| Check-in | 252 | ±0.0839 | ±0.1077 |
| Price | 455 | ±0.0249 | ±0.0540 |
| Staff | 6,256 | ±0.0205 | ±0.0821 |
| Facility | 11,686 | ±0.0238 | ±0.0887 |

Examples (Test 1, B = 2000):

| model | prompt | topic | pairs | F1 | 95% CI |
|---|---|---|---|---|---|
| gemma4_31b | contrary_v1 | price | 455 | 0.9153 | [0.8939, 0.9339] |
| gemma4_31b | contrary_v1 | staff | 6,256 | 0.5858 | [0.5665, 0.6051] |
| gemma4_31b | contrary_v1 | check-out | 8 | 1.0000 | [1.0000, 1.0000] |
| gemma4_31b | contrary_v6 | check-out | 8 | 0.8889 | [0.5000, 1.0000] |
| qwen3.8_27b | contrary_v1 | check-out | 8 | 0.7500 | [0.0000, 1.0000] |
| qwen3.8_27b | contrary_v1 | check-in | 252 | 0.2407 | [0.1372, 0.3559] |

**Check-out cannot be read as a score.** With 8 pairs, an F1 of 0.75 has a 95% interval of
[0.00, 1.00] — it is consistent with the model being perfect and with it being useless. This
directly supports the existing TODO on Table 6 ("the Cross column is an unweighted mean over
topics, so Check-out (8 pairs) weighs as much as Staff (6,256)"): the fix is a pair-weighted
average, and Check-out should carry its interval wherever it is printed.

### Two data problems this surfaced (Task 2)

1. **`gemma4_31b` v2 and v4 are total failures, not scores of 0.** Their error rate is
   `2078/2078 = 1.0000` — every row flagged invalid, so TP=0 and F1=0.0000. Omitted from the
   table above; averaging them in as zeros would badly misstate Gemma. Report as failures or
   re-run.
2. **`qwen3.8_27b_old` duplicates `qwen3.8_27b`.** Every metric is identical across all 36 run
   files; the directories differ only in report formatting. Counting both would double-weight
   Qwen in any average. Omitted above.

### How to reproduce

```bash
# Task 2 — spread across runs, then bootstrap over reviews
python bootstrap_task2.py --root outputs/Task2_new_eval/task2/gt --reps 2000
python bootstrap_task2.py --spread-only

# Task 3 — spread across Test 1/2/3, then bootstrap over candidate pairs
python bootstrap_task3.py --reps 2000
python bootstrap_task3.py --spread-only --model qwen3.8_27b
```

Both scripts read only the evaluator's own output. `bootstrap_task2.py` takes the per-instance
`=> instance TP=.. FP=.. FN=..` lines from `*_workings.txt`; `bootstrap_task3.py` takes the
per-run TP/TN/FP/FN from each `EVAL_SUMMARY.txt`. Those are exactly the counts the published
scores are computed from, which is why each bootstrap's point estimate reproduces the published
score.

### Still to do

The paper's tables name GPT-4o, Gemini 2.5 Pro, Gemma-3 (4B), Qwen-2.5 (7B) and DeepSeek-R1
(7B). The models measured here are `gemma4_31b`, `qwen3.8_27b`, `llama3.3_70b`, `llama4_scout`
and `llama3.2` — different models and sizes. Either the tables' model list or this measurement
set needs reconciling before the numbers go in; both scripts run unchanged on whichever outputs
are the real ones. Table 6 additionally prints a G2 column per topic and a "Same (G3,G4)"
average; neither can carry an interval until those runs exist (see the scope note under
Measurement 3). Since the gold mirrors G1 onto G2 by construction, a G2 run is still worth
having — the paper uses the G1/G2 gap to measure how far a model's labels are from symmetric,
which is exactly what a mirrored gold cannot tell you on its own.
