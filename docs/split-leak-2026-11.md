# The panorama leak in isolation (November 2026)

**Status: Run held, no result exists.** This page is the design and the pre-registered rule for the remaining item
of [#3](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/3). It goes in with the code
(`analysis/split_leak.py`, its job files and tests) before any job runs. Stage 0's three planning figures will be
added here before Stage 1 is submitted. Results, with sha256s, job ids and the commit, come in a later PR. The
run-time choices (seeds, stages, caps) are provisional until Jon decides J1–J3 (the plan on #3).

## Question

The 2025-08 models were trained on a file-level split. Labels of one panorama could sit in train and in test, and
so could two labels of one physical feature seen from adjacent panoramas. Round 3 splits by panorama and merges
groups within 3 m (`split_data.py`). How much did the file-level split inflate test numbers, with everything else
(crops, labels, recipe) held fixed?

## Why the design looks like this

- Round 3's test sets are small. CurbRamp has 1,479 test rows on 1,319 panoramas, and its test AUC interval is
  about ±0.009. Only the minority of rows in multi-label groups can leak at all.
- An independent label-level re-split would compare two different small test sets, and sampling noise would swamp
  the leak.
- So the leaky split is a minimal change to the production split. The confirmatory test is a randomised contrast
  inside the leaky model, which needs no second arm.

## Design

### The leaky split: a coupled permutation

`make-split` recomputes the grouped split with `split_data.assign_splits` (same manifest, same rawLabels
coordinates, seed 42, the same held-out deployments, the 3 m merge) and stops unless the result equals round 3's
`split.csv` byte for byte. It then permutes the split labels among the rows of multi-member groups (a shared
panorama or a 3 m chain), within each class, across the whole label type.

- `test_heldout` is row-identical to the grouped split.
- Every row in a single-label group keeps its split.
- Per-split row counts and class balance are exactly equal in both arms.
- Per-deployment class shares are not held equal, as in a real file-level split. `check` reports only an aggregate
  of the shift (largest and median absolute per-deployment change in incorrect share, and a χ² test of deployment ×
  split per class), with no deployment column.
- The permutation has its own random stream, `default_rng(SeedSequence([seed, SCATTER_TAG]))`, never the
  `default_rng(seed)` stream that `assign_splits` uses.

The leaky file adds `split_mode` (`label_level`), `split_grouped` and `split_group` (the merged group key) before
`split`. It is written under the run directory on scratch, never next to the production split, and never enters
the repo. `fine_tune_classifier.py` records `split_mode` in `training_run.json` and refuses to train a marked split
into `<Type>/dinov2/`. `publish_to_hf.py` refuses a model that records `split_mode`, and refuses any `split.csv`
with a panorama in two of its four splits (no marker needed).

### Exposure

- A val or test row is **exposed** when a *direct* companion is in the leaky train split: a label on the same
  `pano_id`, or one within 3 m of the row itself (`split_data.haversine_m`, the split's own distance).
- k is the row's number of direct companions. The strata are k = 1, 2, 3+.
- A row whose only train companions are further along a merged chain is **chain-only**, a separate descriptive
  class.
- The merged group is used only as the bootstrap cluster.

`check` compares the exposed val/test rows with the iid expectation, the sum of 1 − 0.3^k over val/test rows
with k ≥ 1. It stops when the count falls under 90% of that.

### Arms and seeds (provisional pending J1)

| | Label-level arm (L) | Grouped arm (G) |
|---|---|---|
| Split | one leaky file per training seed (scatter seed = training seed) | round 3's `split.csv` (seed 42) |
| Seeds | 42 (Stage 1), then 43 and 44 (Stage 2) | the stored round-3 runs for 42/43/44, or fresh if Jon chooses |
| Recipe | round 3 unchanged: DINOv2-L, 25 epochs, best by val macro-F1, hard targets | same |
| Weights | always deleted after scoring | deleted after scoring if fresh |

The inference is conditional on the production grouped split.

### What is measured

Each model scores its own val, test and `test_heldout` rows.

**Confirmatory: the exposure contrast inside the L models (test rows only).**

- The permutation randomises which multi-member rows get a companion in train, so exposed against unexposed rows
  inside one model is a causal estimate. It needs no grouped arm.
- Estimand: exposed minus unexposed error rate at 0.5, stratified by k (1, 2, 3+). Mean true-class probability is
  the second metric; per-stratum AUC is descriptive.
- Strata are pooled with Cochran–Mantel–Haenszel weights, n_exp · n_unexp / (n_exp + n_unexp), over k, seeds and
  the five types. A stratum with fewer than 30 unexposed rows is left blank, and the results table says which.
- Descriptive extensions: per type; same-panorama against nearby-only companions; same-class against
  opposite-class companions; chain-only rows; the same contrast on val (which chose the checkpoint).

**The number the issue asks for: inflation.**

- Per type: mean over seeds of AUC(L on L test) − AUC(G on G test), and the same for val. Pooled: the
  macro-average of the five.
- The multi-member part of test is a different row set in each arm, so the comparison is unpaired where it
  matters. The report also gives (Δ on the multi-member part) × (its share of test).

**Secondary.**

- Held-out paired Δ (L − G on identical `test_heldout` rows): does a file-level split also cost true
  generalisation?
- Singleton test rows, paired Δ.
- Gap of gaps: (AUC_L,test − AUC_L,heldout) − (AUC_G,test − AUC_G,heldout).
- Macro-F1 and balanced accuracy at 0.5; best-epoch val macro-F1.
- Operational: the threshold reaching precision 0.92 for Agree on each arm's own val, and the precision it
  realises on `test_heldout`.
- Slices: type × severity and type × tag per arm on its own test rows, with each arm's training counts, and the
  paired slice deltas on `test_heldout` (`analysis/slices.py`).

### Intervals

- **Deciding interval:** a 95% percentile bootstrap, 2,000 replicates, bootstrap seed 3. Within each type, whole
  merged groups are drawn from the union of groups with a row in G test or any L test (val and `test_heldout`
  from their own groups). One draw is shared by both arms and every seed. Seeds are resampled independently per
  arm, with the same seed draw for every type. Types are drawn independently of each other.
- **Sensitivity check:** a t interval on the seed-level values. For the contrast that is a one-sample t on the
  per-seed pooled contrasts; for the inflation, a Welch interval on the per-seed macro AUCs of the two arms. Three
  seeds make the seed level of a bootstrap coarse.
- If the two give different outcomes under the rule, the reading is "unresolved".

## Pre-registered rule (committed before any L model exists)

The labels in quotes are the exact strings `rule_outcome()` prints.

- "leak shown" if the pooled contrast's upper bound on (exposed − unexposed) error is below 0, with three seeds.
- "exposure hurts" if its lower bound is above 0. This is reported as its own outcome, not as a leak.
- "not shown" otherwise, reported with the interval.
- **Inflation** (pooled; per type is descriptive): "shown" if the interval lies above 0; "no material inflation" if
  it lies inside ±0.006 AUC; otherwise "unresolved, upper bound X".
- Against the stored baselines, the inflation labels carry
  "(descriptive: runs from different dates and environments)". They are confirmatory only with a fresh G arm.
- The 0.006 margin is round 3's own single-seed caveat ([`retrain-2026-10.md`](retrain-2026-10.md), "What this does
  not establish": two runs of one setting differed by 0.006 in best validation macro-F1).
- A run stopped after Stage 1 is reported as provisional, whatever its direction: both labels end in "(provisional:
  1 seed)". No stopping rule uses Stage 1's numbers.
- If every stratum is blank, the contrast reads "not estimable"; that is not an outcome of the rule.
- The constants live in `analysis/split_leak.py`, and a test checks that this page quotes them.

## Stages (all held; provisional pending J1–J3)

| Stage | Needs | Does |
|---|---|---|
| 0 | klone CPU, minutes | `make-split`, `check`, `bound` |
| 1 | about 30 GPU-h | L arm, five types, seed 42 |
| 2 | about 60 GPU-h | L arm, seeds 43 and 44 |
| optional | about 95 GPU-h | fresh G arm, three seeds, interleaved with L in the same array (`FRESH_G=1`) |

**Stage 0 `bound`** uses the stored seed-42 predictions and the exposure census. It writes three figures into this
page before Stage 1:

- A **memorisation bound** on the inflation. The multi-member rows of the grouped model's test set are scored as if
  scattered and memorised: each direct companion is in train with probability 0.7, and an exposed row gets a
  perfect score (or, in a second column, its train companions' class). The AUC is then recomputed. It is not a
  strict bound, because it leaves out the effect of selecting the checkpoint on a leaky val.
- The expected inflation interval half-width at three seeds, and so whether the ±0.006 band can be reached. If not,
  this page says in advance that a null will read "unresolved".
- The minimum detectable error-rate difference for the confirmatory contrast at three seeds (power 0.8; it ignores
  clustering, so it is optimistic).

### Stage 0 figures

To be added before Stage 1 is submitted (`$RUNS/report/bound.md`).

## Results

Run held, no result exists.

## Limits

- The inference is conditional on round 3's grouped split, its crops and its labels. It says nothing about the
  2025-08 file split itself (different crops, labels and tilt correction), which is not reproduced.
- Against the stored grouped runs, the inflation compares runs from different dates and environments; only a fresh
  G arm makes it confirmatory.
- Exposure is randomised only among multi-member rows, so the contrast says nothing about singleton rows (which
  cannot leak).
- The deciding interval of the contrast is conservative. In the tests' null simulation (equal-quality arms, no
  memorisation, 200 replications) it covered 0 in 99–100% of replications, against about 95% with the seeds held
  fixed: each seed is a fresh scatter of the same rows, so resampling seeds on top of groups counts the scatter's
  randomisation twice. The inflation interval covered 96–99%. The seed-level t interval on the inflation covered
  only 81–94% (it holds seed variance alone), so under no effect "unresolved" because the two intervals disagree
  will come up more often than one time in twenty.
- Types are resampled independently; a panorama carrying labels of several types links them, which this ignores.
- The minimum detectable difference ignores clustering within merged groups, so the real one is larger.

## Provenance (to be filled by the results PR)

- Commit of the run, `design.json` (sha256 of every manifest, production split, leaky split and hard-example list;
  the sha256 of each rawLabels file used, without names; the held-out deployments as a count only), `bound.json`,
  `results.json`.
- Job ids (from `split_leak_jobs.txt`) and the sha256 of every predictions file.

## Reproduce

From the repo root on klone, in the activated environment:

```bash
DATA=/gscratch/makelab/$USER/sidewalk-validator-ai/retrain-2026-10 RUNS=/gscratch/scrubbed/$USER/split-leak
STAGE=0 DATA=$DATA RUNS=$RUNS analysis/split_leak_submit.sh      # CPU: make-split, check, bound -> $RUNS/report
STAGE=1 DATA=$DATA RUNS=$RUNS analysis/split_leak_submit.sh      # L, seed 42 (refuses before Stage 0's outputs)
STAGE=2 DATA=$DATA RUNS=$RUNS analysis/split_leak_submit.sh      # L, seeds 43 and 44
DRY_RUN=1 STAGE=1 DATA=$DATA RUNS=$RUNS analysis/split_leak_submit.sh   # print, submit nothing
```

`MAX_PARALLEL` (default 4) caps the GPU array; `FRESH_G=1` adds the grouped arm. The report job writes
`tables.md`, `seeds.md`, `slices.md` and `results.json` to `$RUNS/report/`; none of them names a deployment.
