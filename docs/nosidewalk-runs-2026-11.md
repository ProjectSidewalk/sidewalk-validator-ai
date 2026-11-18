# NoSidewalk runs: validate the run, not the label? (November 2026)

**Status: exploratory analysis for
[#102](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/102); nothing here changes serving.** The
step (c) decision of 2026-11-14 on [`nosidewalk-2026-11.md`](nosidewalk-2026-11.md) (serve the five fold models as a
per-label ensemble, one masked round per deployment) stands whatever this page finds.

NoSidewalk labels are usually placed in sequence: one labeler walks a stretch without a sidewalk and drops a label
every panorama or so. A **run** here is one labeler's NoSidewalk labels on one block face, `(city, street_edge_id,
street_side, user_id)`, contiguous or not. The validator scores each label alone. #102 asks four things:

1. **Do the labels of a run stand or fall together?** Measured on existing expert votes, descriptive only.
2. **Does pooling the image score over the run or the face beat the single-label score?**
3. **Do neighbour features that need no image** (run length, labelers on the face, distance to the nearest
   NoSidewalk label, CurbRamp / Obstacle / SurfaceProblem labels on the same face) **add to the image score?**
4. **Should serving vote per run?** A recommendation under a rule fixed before the pooled results were computed.

Code: [`analysis/nosidewalk_runs.py`](../analysis/nosidewalk_runs.py) (`step-a-neighbours`, `agreement`,
`pooling`, `features`, `report`) and [`analysis/pull_expert_vote_times.sh`](../analysis/pull_expert_vote_times.sh).

## Units and rules

- Labels are keyed `city:label_id` (label ids are per city), faces `city:street_edge_id:side`. Faces and runs of a
  label come only from the pull that defined them: between the 2026-10-18 and 2026-11-05 pulls `street_edge_id`
  changed for about a tenth of NoSidewalk labels, so step (a) uses the 2026-11-05 pull (it reproduces every split
  row's face), step (b) the 2026-11-11 pulls, and the type comparison of item 1 the 2026-10-18 all-type pull.
- **A label's verdict in item 1 is its own expert vote**: every expert Agree / Disagree on that label points the same
  way; crowd votes are ignored (the stricter `data_generation.expert_verdict`, where one contradicting crowd vote
  drops the label, is a sensitivity). Never `label_class`: the face rule copies one verdict across a face, which
  would make agreement 1 by construction. Labels placed by an expert or by SidewalkAI are left out and counted.
- Expert and labeler identities stay in memory. Outputs carry a per-city `labeler_idx` and the experts' positions
  (1, 2). Label-level files stay outside the repo, in `~/Data/sidewalk-validator-ai-nosidewalk-2026-11/runs/`.
- Deployments that are not public in the `/v3/api/cities` snapshot of the run day are dropped before any statistic.
- **Every pooled score has its own map.** A pooled score is the mean logit of the per-label `p_correct` over the
  group, and averaging compresses it toward the group mean, so a served accuracy read off the single-label map no
  longer means what it says. Each group definition gets a map fitted on step (a)'s pooled out-of-fold scores of the
  same definition (leave-one-fold-out within step (a); all folds for step (b)); `serve_pooled` refuses any other map.

**What was seen before the rule of item 4 was written.** The planner and the plan's reviewer had computed item 1's
per-city agreements and references and the expert-vote timing in Seattle, Chicago and Oradell. Nobody had computed
any pooled AUC or pooled gate on step (a) or step (b).

## Results

<!-- nosidewalk-runs-results:start -->
<!-- nosidewalk-runs-results:end -->

## Limits

- **Item 1 cannot separate the run from the sitting where it matters.** The votes are unmasked. In Seattle and
  Chicago almost every within-run pair was judged by one expert within an hour, often on a page that shows one
  labeler's labels together (Seattle's are mostly imported "Old data" votes with real timestamps, Chicago's mostly
  LabelMap). There, "the labels of a run stand or fall together" and "the expert judged the run as a block" predict
  the same numbers. Oradell has pairs judged days apart, but 98 % of its labels are correct, so its agreement sits
  close to chance by arithmetic.
- **Which labels got an expert vote is not random**: experts looked at faces and labelers they had reason to look
  at. The nulls compare against the same labels, not against the population.
- **Step (a) truth is entangled with item 1.** 546 of the 885 split rows carry a verdict copied from their face, and
  the label-level expert verdicts on one face come from the same sittings. Step (a) is therefore secondary; the
  step (b) ratings (masked, one label per face) are the primary test of pooling.
- **Step (b) is small.** 235 decided ratings, 31 incorrect, 2 of them in the pass stratum. Weighted rates are set by
  agree-below labels whose inclusion weights reach about 365.
- **Selection through the rated label.** Step (b)'s faces were drawn through one rated label each, 60 / 20 / 20 by
  stratum, and the labels beside it follow that label's stratum.
- **Crop paths.** Step (a)'s pilot crops came from `data_generation.py`, step (b)'s and this page's re-scored ones
  from `ai_labels_eval.py crops`; both arms of item 2 use the re-scored values (crop parity above).
- **Distance is a straight line** between label points, a proxy for distance along the street.
- Shared panoramas: a panorama can carry labels of two faces, so face-grouped resampling is not panorama-grouped.

## Follow-ups (one issue, filed by Jon or at his word)

- **A masked round that rates 2-3 labels per run**, in separate sittings. It is the only clean test of item 1 and
  the pre-registered test of a per-run gate. It needs a rater.
- Scoring labels on neighbouring faces within a distance along the street, and true along-street distance from the
  street geometry.

## How to rerun

```
D=~/Data/sidewalk-validator-ai-nosidewalk-2026-11
# experts in the environment only, in the order EXPERT1_ID, EXPERT2_ID
export EXPERTS="<name>=<uuid>,<name>=<uuid>" EXPERT1_ID=<uuid> EXPERT2_ID=<uuid>
python analysis/nosidewalk_runs.py step-a-neighbours --pull $D/data/rawlabels-ns-2026-11-05 --gate $D/gate2 \
    --cities-json $D/runs/cities_2026-11-17.json --out $D/runs/step-a
# klone (docs/running-on-hyak.md): stage + crop + score with the kept gate2 weights, labels split in two by city
RUN=$R DEST=$STORE sbatch analysis/ai_labels_stage.slurm
RUN=$R/a RUNS=<gate2 runs> STORE=$STORE sbatch --dependency=afterok:<stage> nosidewalk_round.slurm   # and $R/b
ssh makelab1 "SCHEMAS='<public schemas>' EXPERT1_ID=... EXPERT2_ID=... bash -s" \
    < analysis/pull_expert_vote_times.sh > $D/runs/expert_vote_times.csv
python analysis/fetch_all.py $D/runs/rawlabels-contra-2026-11-17 --label-types CurbRamp,Obstacle,SurfaceProblem,NoSidewalk
python analysis/nosidewalk_runs.py agreement --ns-pull $D/data/rawlabels-ns-2026-11-05 \
    --all-pull ~/Data/sidewalk-validator-ai-retrain-2026-10/round3/rawlabels --vote-times $D/runs/expert_vote_times.csv \
    --cityparams $D/round-b/gallery_jon/cityparams_snapshot.conf --cities-json $D/runs/cities_2026-11-17.json \
    --out-json $D/runs/agreement.json
K=$D/round-b/private-key-do-not-share
python analysis/nosidewalk_runs.py pooling --step-a $D/runs/step-a --scores $D/runs/step-a-scores/a $D/runs/step-a-scores/b \
    --gate $D/gate2 --gate1 $D/gate --priors $D/data/whole/deployment_priors.json --key $K/key.csv \
    --sheet $K/votes_jon.csv --face-scores $D/round-b/neighbours/face_scores.csv --out-json $D/runs/pooling.json \
    --out-dir $D/runs
python analysis/nosidewalk_runs.py features --step-a $D/runs/step-a --scores $D/runs/step-a-scores/a \
    $D/runs/step-a-scores/b --gate $D/gate2 --key $K/key.csv --sheet $K/votes_jon.csv \
    --face-scores $D/round-b/neighbours/face_scores.csv --ns-pull $D/data/rawlabels-ns-2026-11-05 \
    --round-b-pull $D/round-b/rawlabels --contra-pull $D/runs/rawlabels-contra-2026-11-17 \
    --cities-json $D/runs/cities_2026-11-17.json --out-json $D/runs/features.json
python analysis/nosidewalk_runs.py report --agreement $D/runs/agreement.json --pooling $D/runs/pooling.json \
    --features $D/runs/features.json --cities-json $D/runs/cities_2026-11-17.json \
    --out-json docs/nosidewalk-runs-2026-11/runs.json --out-md docs/nosidewalk-runs-2026-11/runs.md \
    --doc docs/nosidewalk-runs-2026-11.md
```

Every subcommand stops on an input whose sha256 differs from `PINS` in the module. Seeds: `--seed 0`, 1,000
permutations and 1,000 bootstrap resamples; 50 m for group D; vote-gap strata at 1 h and 1 day.
