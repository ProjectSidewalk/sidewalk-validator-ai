# AI-placed labels in training: a held-out slice, the round-4 ablation, and full-pool ranked lists (November 2026)

Issues [#79](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/79) and
[#80](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/80); follows
[`ai-labels-2026-10.md`](ai-labels-2026-10.md). The policy itself is recorded in
[`decisions.md`](decisions.md) ("AI-placed labels: included only tagged, with a held-out evaluation slice").

**Status: tooling done, nothing run.** No data was generated, no model trained and nothing scored for this page.
Every result section below says "not yet run" and gives the exact command. Run outputs go to
`~/Data/sidewalk-validator-ai-ai-label-holdout-2026-11/` on the Mac and `/gscratch/makelab/jfroehli/ai-label-holdout-2026-11/`
on klone, never into the repo (label lists can include private deployments; the relabel key holds a user_id).

## Question

1. **#79.** Round 3 trained with `--include-ai-labels`, and the selection keeps every `incorrect` row, so it consumed
   every lead-rejected Vancouver AI label. Does training on tagged AI-placed labels make the CurbRamp validator better
   or worse at judging AI-placed labels it has never seen, and what does it cost on the human-placed test and held-out
   splits?
2. **#80.** Every October number is on AI labels that came through the validation queue. On the full RampNet pool
   (about 62 k unvoted Vancouver labels, about 12 k Richmond), how often is the validator's Disagree a bad pin that a
   lead would reject?

## The held-out set (#79)

`data_generation.py` calls `ai_label_holdout.apply` between `eligible_labels` and `select_training_labels`:

- **Who is a candidate.** Every eligible AI-placed label (the training vote rule: net human margin, or an expert
  verdict with `EXPERTS`), `correct` and `incorrect` alike. Nothing is excluded by construction.
- **Who is held out.** A label is held out when its draw, the first 12 hex digits of
  `sha256("ai-holdout:<seed>:<city>:<label_id>")` read as a fraction, is below `--ai-holdout-frac` (default 0.5).
  The draw is the label's own, so a new pull moves at most the labels whose votes changed, and a smaller fraction
  holds out a subset of a larger one. The salt keeps it independent of the correct-class downsample rank.
- **Both policies carve.** With or without `--include-ai-labels` the same labels are held out. Without the flag
  those rows would have been dropped anyway, so the selection hash of an existing run does not move; and both
  ablation arms below score the identical set, which their two `ai_label_holdout.csv` files confirm (same sha256).
- **What is written.** `<out>/CurbRamp/ai_label_holdout.csv` (`label_uid, city, label_id, label_class,
  class_source, pano_source, tier, stratum, holdout_rank`, sorted by `label_uid`; written on every run, empty when the
  pool has no AI label) and `generation_run.json["ai_labels"]`:
  `policy`, `holdout_frac`, `holdout_salt`, `holdout_seed`, `holdout_file`, `holdout_sha256`, and per deployment ×
  class `holdout_counts`, `included_counts` (AI rows that reached training; empty under `excluded`) and
  `eligible_ai_counts` (before carving). `--select-only` prints the counts and writes no file.
- **Tier** comes from the submission campaign of the label's `time_created` (`ai_label_holdout.CAMPAIGNS`, moved
  from `analysis/ai_labels_eval.py`): `core` (RampNet 0.55), `band` (0.30 to 0.55), `mixed`. Every Vancouver label
  is `core`.
- **Evaluation slice.** `evaluate.py` reports AI-placed test and held-out rows as an `ai_authored` slice whenever the
  split has any (it never does when the held-out AI labels stay out of `split.csv`, which is the point; the slice is
  there for runs that let AI labels into the test split).

Expected size, from the October counts (eligible under the training rule, before carving): about 3,600 AI-placed
CurbRamp labels, about 180 of them `incorrect`; at 0.5, about 90 rejected AI labels in each half.

**Held-out counts: not yet run.** Filled from `generation_run.json["ai_labels"]` of the `with` run:

| deployment | tier | eligible correct | eligible incorrect | held out correct | held out incorrect | in training (with) |
|---|---|---:|---:|---:|---:|---:|
| vancouver-wa | core | | | | | |
| richmond-va | core / band / mixed | | | | | |
| laurens-ia | core / band | | | | | |

### Caveats

- **Per label, not per panorama.** Among the October voted AI labels, 73 % in Laurens (860 of 1,180), 27 % in
  Richmond (294 of 1,072) and 10 % in Vancouver (274 of 2,794) share a panorama with another voted AI label. In the
  `included_tagged` arm a held-out label can therefore sit on a panorama the model trained on, which favours that
  arm. Report the ablation per deployment (Laurens separately) and read a Laurens gain with that in mind. Carving by
  `city:pano_id` instead would remove it; it is a one-line change in `ai_label_holdout.carve` if the review prefers it.
- **The reference is the training rule's class** (crowd margin or expert), taken from the holdout file, not the
  single-vote majority `ai_labels_eval.py report` uses. The two differ on lightly voted labels; most AI labels have
  one vote, so with `EXPERTS` set the eligible pool is mostly expert-decided labels (October: 3,659 of 5,046 voted AI
  labels had an expert vote), and without it the pool is small.
- **The human splits differ slightly between arms.** `split_data.py` groups by panorama and balances classes, so
  adding AI rows can move some human panoramas between splits. The held-out AI slice is identical in both arms; the
  human test and held-out numbers are each arm's own.

## Protocol: the round-4 CurbRamp ablation (#79)

Two `data_generation.py` runs into two output directories, same seed 42, same cities and rawLabels snapshot as round
3, one with `--include-ai-labels`; then the usual split, training and evaluation for each; then the held-out AI labels
scored by both checkpoints. Not yet run. On klone, from a checkout with the Depth-Anything-V2 submodule and the depth
checkpoint in place, inside a GPU allocation (the commands are what `data_generation.slurm` runs, for one type):

```
B=/gscratch/makelab/jfroehli/ai-label-holdout-2026-11
CACHE=/gscratch/makelab/jfroehli/sidewalk-validator-ai/retrain-2026-10/rawlabels     # round 3's snapshot
for ARM in with without; do
  FLAG=$([ $ARM = with ] && echo --include-ai-labels)
  python -u data_generation.py --label_type CurbRamp --out $B/$ARM --rawlabels-cache $CACHE \
      --pano-store "$PANO_STORE" --depth-encoder vitb --tilt-beta 0.95 --seed 42 $FLAG
  python -u split_data.py --label_type CurbRamp --out $B/$ARM --seed 42 --holdout-cities "$HOLDOUT" \
      --hard-examples analysis/hard_examples_2026-10-17.csv --rawlabels $CACHE --merge-radius-m 3
done
cmp $B/with/CurbRamp/ai_label_holdout.csv $B/without/CurbRamp/ai_label_holdout.csv   # must be identical
# train + evaluate each arm (evaluate.py --per-city runs inside the job), 25 epochs, seed 42
LT=CurbRamp VARIANT=baseline TAG=with    DATA=$B/with    RUNS=/gscratch/scrubbed/$USER/aiho sbatch analysis/confident_learning.slurm
LT=CurbRamp VARIANT=baseline TAG=without DATA=$B/without RUNS=/gscratch/scrubbed/$USER/aiho sbatch analysis/confident_learning.slurm
```

`$HOLDOUT` is round 3's held-out city list (from its `split.csv`). Round 3 passed private deployments explicitly, so
add `--cities` with round 3's list (`generation_run.json["cities"]`) to both `data_generation.py` calls; never write a
private deployment's name into this page. Run with the same `EXPERTS` as round 3: most AI labels have one vote, so
without it few are eligible at all. `--map` is optional; with it the ablation adds the deployed gate per arm.

Then the held-out AI labels, scored by both checkpoints (Mac for `select` and `ablation`, klone for the GPU stages):

```
python analysis/ai_labels_eval.py select --rawlabels $CACHE --holdout $B/with/CurbRamp/ai_label_holdout.csv \
    --cities vancouver-wa,richmond-va,laurens-ia --out $B/heldout
RUN=$B/heldout DEST=/gscratch/makelab/jfroehli/pano-stage/ai-label-holdout sbatch analysis/ai_labels_stage.slurm
RUN=$B/heldout STORE=/gscratch/makelab/jfroehli/pano-stage/ai-label-holdout CKPT=<with best checkpoint> \
    PRED=predictions_with.csv sbatch analysis/ai_labels_gen.slurm
RUN=$B/heldout STORE=... CKPT=<without best checkpoint> PRED=predictions_without.csv sbatch analysis/ai_labels_gen.slurm
python analysis/ai_labels_eval.py ablation --run $B/heldout --pred-a $B/heldout/predictions_with.csv \
    --pred-b $B/heldout/predictions_without.csv --map <an accuracy map, e.g. round 3's served curbramp.json> \
    --out $B/heldout/ablation
```

The `select` rawLabels directory must be the same snapshot the generation runs used (a held-out label missing from it
stops `select`). The held-out set is small (about 1,800 labels on fewer panoramas), so staging and cropping fit the
jobs' default time limits.

### Result: not yet run

Held-out AI-placed labels (`ablation.md`), A = `included_tagged`, B = `excluded`, paired panorama bootstrap:

| slice | n | incorrect | AUC A | AUC B | Δ AUC [95% CI] | Disagree precision A / B [Wilson] | Disagree recall A / B [Wilson] |
|---|---:|---:|---:|---:|---|---|---|
| pooled | | | | | | | |
| city = vancouver-wa | | | | | | | |
| city = richmond-va | | | | | | | |
| city = laurens-ia | | | | | | | |
| tier = core | | | | | | | |
| tier = band | | | | | | | |

Beside it, from each arm's `metrics.json` (`evaluate.py --per-city`) and `analysis/retrain_report.py`: human-placed
test and held-out AUC, within-city and held-out-city with paired panorama-bootstrap intervals, and the type ×
severity and type × tag slices with training counts (`analysis/slices.py`), A against B. AI-placed labels carry
neither severity nor tags (RampNet submits none; as in [#69](https://github.com/ProjectSidewalk/sidewalk-validator-ai/pull/69)),
so the severity and tag slices are of the human-placed rows only.

| slice (human-placed) | AUC A | AUC B | Δ [95% CI] | training rows A / B |
|---|---:|---:|---|---|
| test | | | | |
| test_heldout | | | | |
| severity / tag rows from `slices` | | | | |

## Protocol: full-pool ranked Disagree lists (#80)

Score every AI-placed CurbRamp label in vancouver-wa and richmond-va with the round-3 checkpoint, keep the Disagree
verdicts, rank them (`band` tier first, then the log-odds `rank_score`), cap at 300 (Vancouver) and 150 (Richmond), add
50 decoys each, and have a lead vote on the masked Gallery list, after
[#78](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/78). Labels the rater ever voted on are left
out (pull rawLabels right before building). Laurens is skipped: 75 % already validated and the rater cast nearly every
vote. Not yet run.

```
R=/gscratch/makelab/jfroehli/ai-label-holdout-2026-11/all-ai
python analysis/ai_labels_eval.py select --all-ai --rawlabels <fresh pull> --cities vancouver-wa,richmond-va \
    --round3-split <round 3>/CurbRamp/split.csv --out $R
RUN=$R DEST=/gscratch/makelab/jfroehli/pano-stage/all-ai sbatch --time=24:00:00 analysis/ai_labels_stage.slurm
RUN=$R STORE=/gscratch/makelab/jfroehli/pano-stage/all-ai CKPT=<round-3 CurbRamp checkpoint-4340> \
    sbatch --time=24:00:00 analysis/ai_labels_gen.slurm
RATER=<user_id> EXPERTS=... python analysis/ai_labels_relabel.py build --run $R --city vancouver-wa --pool all \
    --cap 300 --rawlabels-before <pull right before building>/vancouver-wa.csv --out $R/ranked-vancouver-wa
RATER=<user_id> EXPERTS=... python analysis/ai_labels_relabel.py build --run $R --city richmond-va --pool all \
    --cap 150 --rawlabels-before <pull>/richmond-va.csv --out $R/ranked-richmond-va
# after rating: a fresh pull of each city into <out>/rawlabels_after/<city>.csv, then
RATER=<user_id> python analysis/ai_labels_relabel.py score --run $R --out $R/ranked-vancouver-wa
```

**Size.** About 78 k labels on about 33 k panoramas (28,881 Vancouver and 4,613 Richmond panoramas hold AI
CurbRamp labels in the October pull). Both jobs are multi-hour: the stage job pulls 1,000 panoramas per ssh
connection in sequence (about 34 connections, spaced by the transfer of each batch, which stays inside the makelab
rate rule), and the crop job cuts one depth-sized crop per label on one GPU. The default limits (6 h stage, 4 h crops)
will not do; the `--time` overrides above are a guess, and both jobs resume on resubmission. The staged copy needs
disk for every panorama (check `du` on the first batches before letting it run). `--round3-split` flags the 220
Vancouver labels round 3 trained, validated or tested on; `build` uses unseen labels only.

### Result: not yet run

`results.md` per city (`hit_rate_markdown`): hit rate = rater Disagree / rater's decided votes (Unsure counted beside,
not in the denominator), Wilson 95 %.

| slice | Vancouver n | Vancouver hit rate | Richmond n | Richmond hit rate |
|---|---:|---|---:|---|
| ranked, all | | | | |
| ranked, tier band | | | | |
| ranked, tier core | | | | |
| ranked, unvoted | | | | |
| ranked, other-voted | | | | |
| ranked, rank 1-50 / 51-100 / ... | | | | |
| decoys (validator Agree) | | | | |

Comparators: against a project lead the validator's Disagree on a RampNet label was right 38 % [29, 49] of the time
(33 of 86, Vancouver voted labels, [#71](https://github.com/ProjectSidewalk/sidewalk-validator-ai/pull/71)); by the
crowd reference about 1 in 3 of its Disagrees were bad pins against 1 in 50 overall. #80's decision rule: a hit rate
around 40 % or better, with recurring RampNet campaigns, reopens a Disagree-only routing predicate for the AI role
([SidewalkWebpage#5790](https://github.com/ProjectSidewalk/SidewalkWebpage/issues/5790)).

## Reproduce

Everything above, with the sha256s to pin once run (`generation_run.json["ai_labels"]["holdout_sha256"]` of both
arms, `selection.json`, `ablation.json["inputs"]`, each relabel `manifest.json`):

| artefact | sha256 |
|---|---|
| `with/CurbRamp/ai_label_holdout.csv` (= `without/...`) | not yet run |
| `heldout/labels.csv` | not yet run |
| `heldout/predictions_with.csv` / `predictions_without.csv` | not yet run |
| `all-ai/labels.csv` | not yet run |
| `ranked-*/private-key-do-not-share/key.csv` | not yet run |

Unit tests (CPU): `pytest -q tests/test_ai_label_holdout.py tests/test_generation_run.py tests/test_evaluate.py
tests/test_ai_labels_eval.py tests/test_ai_labels_relabel.py`.
