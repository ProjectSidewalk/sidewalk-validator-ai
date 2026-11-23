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

- **Bootstrap interval:** a 95% percentile bootstrap, 2,000 replicates, bootstrap seed 3. Within each type, whole
  merged groups are drawn from the union of groups with a row in G test or any L test (val and `test_heldout`
  from their own groups). One draw is shared by both arms and every seed. Types are drawn independently of each
  other.
  - **Exposure contrast** (confirmatory and descriptive): groups only, with the three seeds held fixed.
  - **Inflation and the other L − G deltas:** seeds are also resampled, independently per arm, with the same seed
    draw for every type.
- **t interval:** on the seed-level values. For the contrast that is a one-sample t on the per-seed pooled
  contrasts; for the inflation, a Welch interval on the per-seed macro AUCs of the two arms (unpaired: an L run and
  a G run with the same seed number are different trainings on different splits, not a pair).
- Both intervals decide. If they give different outcomes under the rule, the reading is "unresolved". For the
  inflation the Welch interval is co-deciding, not only a check: in the review's null simulations (see the
  amendment and Limits) the bootstrap interval under-covered when seed variance dominated and the Welch interval
  when sampling variance did, and the two excluded 0 together in 2–7% of replications.
- 2,000 replicates, because with 200 the percentile endpoints under-cover (93% in the review's simulations); 400
  and above were nominal.

### Amendment, 2026-11-23 (before any run)

Made in review of the PR that adds this page. When it was made, no leaky split, no Stage 0 figure and no L model
existed; nothing had been submitted to klone.

1. **The contrast's bootstrap holds the seeds fixed.** The plan resampled seeds on top of groups for every
   estimate. Each row carries its three seeds' split and exposure with it, so drawing groups already resamples the
   scatter's randomisation; drawing seeds as well counts it twice. In the review's null simulations (1,500
   replications at 400 or more replicates, with and without per-model quality differences, no memorisation) that
   form covered 0 in 98–100% of replications, against 95.7% for groups only, and its interval was about 1.5 times
   as wide. The implementer's simulations agreed (99.7% against 95.3%).
   Exposure is randomised inside each model, so under the null a better or worse model shifts its exposed and
   unexposed rows alike and adds no seed-level variance to the contrast. The inflation keeps its seed resampling,
   since seed-to-seed quality differences are real variance there (with per-model quality differences, the
   groups-only form covered the inflation in only 19–37%).
2. **Seeds are counted from the data.** A seed whose strata are all blank contributes no per-seed contrast, and
   with fewer than two the t interval does not exist. A label is confirmatory only with three estimable seeds (for
   the inflation, the smaller of the two arms' counts) and its t interval; otherwise it ends in
   "(provisional: k of 3 seeds)", or "(provisional: no t interval)" when the seeds are there but the interval is
   not. A t interval that exists still decides disagreement at two seeds; provisional is added to "unresolved",
   not put in its place.

## Pre-registered rule (committed before any L model exists)

The labels in quotes are the exact strings `rule_outcome()` prints.

- "leak shown" if the pooled contrast's upper bound on (exposed − unexposed) error is below 0, with three
  estimable seeds.
- "exposure hurts" if its lower bound is above 0. This is reported as its own outcome, not as a leak.
- "not shown" otherwise, reported with the interval.
- **Inflation** (pooled; per type is descriptive): "shown" if the interval lies above 0; "no material inflation" if
  it lies inside ±0.006 AUC; otherwise "unresolved, upper bound X".
- Against the stored baselines, the inflation labels carry
  "(descriptive: runs from different dates and environments)". They are confirmatory only with a fresh G arm.
- The 0.006 margin is round 3's own single-seed caveat ([`retrain-2026-10.md`](retrain-2026-10.md), "What this does
  not establish": two runs of one setting differed by 0.006 in best validation macro-F1).
- A label is provisional, whatever its direction, with fewer than three estimable seeds or no t interval (see
  the amendment): it ends in "(provisional: k of 3 seeds)" or "(provisional: no t interval)", added after
  "unresolved (the bootstrap and t intervals disagree)" when the two intervals disagree. A run stopped after
  Stage 1 reads "(provisional: 1 of 3 seeds)". No stopping rule uses Stage 1's numbers.
- If every stratum is blank, the contrast reads "not estimable"; that is not an outcome of the rule, and it is
  marked provisional.
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
- With a real effect, the contrast's bootstrap interval (seeds fixed) describes these three trained models, not
  every training run; the seed-level t interval carries that, and "leak shown" needs the two to agree.
- In null simulations the inflation's bootstrap interval covered 0 in 95–99% of replications, but only 81–85% when
  every trained model had its own quality; its Welch interval covered 76–92% under the plain null. Each alone is
  too narrow in one of the two cases, which is why both decide (above).
- The k = 3+ stratum pools rows whose chance of being unexposed differs by an order of magnitude (2.7% at k = 3,
  under 0.1% at k = 6), so inside it exposure is confounded with k. At 30 unexposed rows it will almost always be
  blank.
- Types are resampled independently; a panorama carrying labels of several types links them, which this ignores.
  `check` writes the share of test rows whose panorama also carries a test row of another type
  (`cross_type_test_share` in `design.json`), which sizes this limit.
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
