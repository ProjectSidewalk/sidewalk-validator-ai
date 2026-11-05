# A VLM adjudicator of the AI's Disagree votes, against the expert gold set (November 2026)

Experiment for [#77](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/77) and the scoring step of
[#12](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/12), under the umbrella
[#74](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/74): can an open-weight vision-language model
decide whether a contested label is correct more often than the crowd does, measured against the expert gold labels
of rounds 1 and 2? And does weighting the crowd by validator reliability get closer to the expert than the plain
majority does?

**Status: not yet run.** The code, the klone jobs and the launcher are in this branch. The gold table below was
built locally from the private rating files (aggregates only). Every other results block is a placeholder that
`analysis/adjudicator.py report` fills once the jobs have run. Nothing here should be read as a result yet.

## Why

The round-3 validators learn from the crowd's margin rule, and that rule matches an expert only about three times
in four on contested labels: 74.5% [64.8, 82.2] on round 1's voted Disagrees
([PR #66](https://github.com/ProjectSidewalk/sidewalk-validator-ai/pull/66)). Round 2
([#78](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/78)) confirmed the pattern by stratum: where
the crowd's margin says correct, the experts side with the crowd. Phase 0 of #74 stopped the AI's Disagree votes.
Before any automatic method relabels training rows, it has to beat the crowd on the same items. Confident learning
([#76](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/76), [`confident-learning-2026-10.md`](confident-learning-2026-10.md))
finds suspect rows from the validator's own scores. This study asks a different judge, a VLM given the label type's
definition and the rating rule, and holds it to an explicit bar.

## The gold set

`adjudicator.py gold` joins round 1 (300 items, one rater, every deployment) and round 2 (500 items, two raters,
Seattle and Chicago, 100 rated by both and adjudicated) into one table keyed on `city_id:label_id`. Round 1's
strata are recomputed from each item's tallies with the round-2 rule (`gt_rounds.assign_stratum`), so round 1's
`voted_disagree` splits into `light_voted` and the two margin strata. A shared round-2 item takes its adjudicated
gold label; the other round-2 items take their rater's verdict. Can't-tell verdicts and the 11 items rated unmasked
are kept but are not *decided*, and only decided items are scored. The AI vote and the human tallies are refreshed
from the 2026-10-27 pool pull. Private deployments are pooled in every table.

<!-- BEGIN GENERATED: gold_counts (analysis/adjudicator.py) -->

| Round | Stratum | CurbRamp | NoCurbRamp | Obstacle | SurfaceProblem | Crosswalk | all |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | decoy_agree | 20 / 20 | 19 / 20 | 1 / 1 | 17 / 20 | 20 / 20 | 77 / 81 |
| 1 | decoy_unsure | 0 / 0 | 0 / 0 | 15 / 19 | 0 / 0 | 0 / 0 | 15 / 19 |
| 1 | light_voted | 14 / 15 | 6 / 6 | 5 / 8 | 3 / 4 | 9 / 9 | 37 / 42 |
| 1 | lone | 20 / 20 | 17 / 20 | 17 / 20 | 16 / 20 | 20 / 20 | 90 / 100 |
| 1 | margin_correct | 2 / 2 | 3 / 3 | 2 / 2 | 1 / 1 | 2 / 2 | 10 / 10 |
| 1 | margin_incorrect | 3 / 3 | 11 / 11 | 10 / 10 | 14 / 15 | 9 / 9 | 47 / 48 |
| 2 | decoy_agree | 32 / 32 | 29 / 32 | 20 / 21 | 28 / 32 | 32 / 32 | 141 / 149 |
| 2 | decoy_unsure | 0 / 0 | 0 / 0 | 11 / 11 | 0 / 0 | 0 / 0 | 11 / 11 |
| 2 | light_voted | 11 / 12 | 10 / 10 | 8 / 8 | 8 / 8 | 10 / 10 | 47 / 48 |
| 2 | lone | 28 / 30 | 23 / 24 | 11 / 22 | 16 / 20 | 26 / 26 | 104 / 122 |
| 2 | margin_correct | 16 / 16 | 10 / 12 | 11 / 12 | 7 / 10 | 13 / 14 | 57 / 64 |
| 2 | margin_incorrect | 26 / 26 | 22 / 22 | 18 / 18 | 17 / 18 | 22 / 22 | 105 / 106 |
| all | all | 172 / 176 | 150 / 160 | 129 / 152 | 127 / 148 | 163 / 164 | 741 / 800 |

Cells are decided / items. Decided = a correct or incorrect verdict reached masked; 48 can't tell, 0 unanswered and 11 unmasked items are not decided. 131 decided items are round-3 train rows; 13 items were flagged by confident learning.

`gold.csv` sha256 `2b472651cb01011a8640b86eb4103e1794ab482186ba3a3471e7d019273edcae`.

<!-- END GENERATED: gold_counts -->

Round 2's lone Obstacle cell is thin: 11 of 22 items are decided. Ten of the 11 unmasked items (one excluded
author, [#85](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/85)) are in that cell, and one more
is can't tell.

## Method

**Prompt.** One user turn, image first. The text is the scoreboard's definition of the type
(`vlm_baseline.DEFINITIONS`), then a rule paragraph per type (`adjudicator.RULE`, the round-2 rating rule written
out), the centre sentence, the labeler's severity and the yes/no question. Nothing about the AI, the other
validators or the crowd is in it. Tags are off by default (`--show-tags` is the `with-tags` variant, Q1). The
CurbRamp prompt at severity 2:

> A curb ramp is a short ramp cut into or built up to a curb so that people using wheelchairs, walkers or strollers
> can move between the sidewalk and the street, often with yellow tactile paving at the bottom. Driveways, alleys
> and flush building entrances are not curb ramps. A curb ramp that is present at the labelled point but in poor
> condition (steep, narrow, no tactile warning, damaged, badly aligned) is a CORRECT label; its problems belong in
> the severity and tags. The label is INCORRECT only if there is no curb ramp at the labelled point (for example a
> driveway, an alley, a flush entrance or plain curb), or the ramp is somewhere else in the image. The labelled
> point is at the centre of the image. The labeler rated its severity 2 on a 1 (minor) to 5 (severe) scale. Is
> this a correct curb ramp label? Answer yes or no.

Prompt version `adjudicator-v1`. Template sha256 (placeholders unrendered, all five types)
`c20a02e142e9ce0e4296171f20106028bce5230aa29306f1648db0ad8caabcf3`. With tags:
`96f40ddc90e4af46dd8d8bc8c5067398705613a2c9be5f35ecf936fbd76abb05`. Every score row also carries the sha256 of
its own rendered prompt. `python analysis/adjudicator.py prompt` prints all five.

**Scoring.** As in [`vlm-scoreboard.md`](vlm-scoreboard.md)'s vLLM leg: one generated token, greedy, the top-20
log-probabilities of that token, and `p_incorrect` = P(no) / (P(yes) + P(no)) over the yes/no spellings
(`vlm_vllm.score_top_logprobs`). The verdict is incorrect when `p_incorrect > 0.5`. A second request on the gold set
asks "In one sentence, why?" after the model's own answer (greedy, at most 60 tokens). The reasons feed the
disagreements packet, not the numbers. Zero-shot only: the scoreboard's 8-shot examples are training rows whose
targets are what is under test.

**Models** (the top of the scoreboard with full coverage; `analysis/adjudicator.slurm` presets):

| Preset | Repo @ revision | Serving | Thinking |
|---|---|---|---|
| `glm5.3-flash` | `zai-org/GLM-5.3-Flash` @ `46a050bec13002ad29ee795c16e61cf4fa3c4a9c` | vLLM 0.32.0, TP 4, 4x H200 | template has no off switch: assistant prefill `<think></think>` (`continue_final_message`) in both requests, `reasoning_effort=low` |
| `gemma4-31b` | `google/gemma-4-31B-it` @ `842da3794eaa0b77d5f08bae87a17459d91ff475` | vLLM 0.32.0, TP 2, 2x L40S, `max_model_len` 4096 | `enable_thinking=false` |
| `qwen3vl-32b` | `Qwen/Qwen3-VL-32B-Instruct` @ the cache's `refs/main`, resolved and recorded at run time | vLLM 0.32.0, TP 2, 2x L40S | Instruct (none) |

Every meta file records the requested and the served revision, the vLLM version, the image tag, the full
`vllm serve` line, the rendered prompt tail (from `/tokenize`, to check the prefill landed), the Slurm job id and the
median seconds per crop.

**Crops.** Items with a round-3 crop use it. The rest are cut on klone by the unchanged
`ai_labels_eval.py crops` leg (the training crop pipeline) from a pano stage of at most 600 panoramas. Their
geometry comes from the public rawLabels cache. Private-deployment and excluded-author items need one read-only pull
(`analysis/pull_label_points.sh`). Without it they are reported as not scorable (Q7).

## The bar

A (model, type) pair **clears** when all three hold (`adjudicator.bar_decision`):

1. at least 30 decided items of the type have a crowd majority (`n_pair >= 30`);
2. on those same items, the 95% lower bound of (model agreement with the expert) minus (crowd majority agreement
   with the expert) is above 0. The bound comes from a label-level paired bootstrap, 2,000 resamples, seed 0;
3. the model's agreement on every decided item of the type, the lone items without a crowd majority included, is at
   least the crowd's on the paired items.

The round-1 reference (74.5% [64.8, 82.2], n = 94) is printed as a fixed row. The decision uses the same-item crowd
rate instead, because the items differ between rounds. McNemar's exact test on the discordant pairs is reported
beside the bootstrap.

## Results

<!-- BEGIN GENERATED: agreement (analysis/adjudicator.py) -->

_Not yet run._

<!-- END GENERATED: agreement -->

**Paired with the crowd** (decided items with a crowd majority):

<!-- BEGIN GENERATED: paired (analysis/adjudicator.py) -->

_Not yet run._

<!-- END GENERATED: paired -->

**By stratum:**

<!-- BEGIN GENERATED: strata (analysis/adjudicator.py) -->

_Not yet run._

<!-- END GENERATED: strata -->

**By severity and tag** (`analysis/slices.py`, with the round-3 training rows per slice):

<!-- BEGIN GENERATED: slices (analysis/adjudicator.py) -->

_Not yet run._

<!-- END GENERATED: slices -->

**The reliability-weighted crowd** ([#12](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/12)),
in the same layout. Three crowd targets are scored against the expert on the same items:

- the plain majority from the pool;
- the plain majority from the 2026-10-19 rawLabels snapshot;
- the reliability-weighted majority (`validator_reliability.py`'s `vote_weights.csv`, built 2026-10-22 from that
  snapshot).

The stage refuses a snapshot made on or after the first rating day, so no rater's gold vote is in the crowd. Only
items from a public deployment with at least one human vote in the snapshot are covered.

<!-- BEGIN GENERATED: reliability (analysis/adjudicator.py) -->

_Not yet run._

<!-- END GENERATED: reliability -->

**The bar, per model and type:**

<!-- BEGIN GENERATED: bar (analysis/adjudicator.py) -->

_Not yet run._

<!-- END GENERATED: bar -->

If no pair clears, the margins above are the record, so nobody re-runs this to find out.

## Disagreements packet

`adjudicator.py disagreements` builds, per model, a reading packet of up to 30 items where the model and the expert
disagree, at most 6 per type, in a seeded order. Each item shows the crop, the type, the severity and the model's
verdict and reason. The expert verdict, the AI vote and the crowd are not shown. The key is written outside the
repository. It is for reading the model's failure modes, not for rating.

## Treatments

Only for (model, type) pairs that clear the bar (`relabel` refuses the rest unless `--force`, which the meta file
records). The **contested rows** of a type are its round-3 train rows that are an AI Disagree in the pool, or
flagged by confident learning. Expert-sourced targets are never changed (`--no-spare-expert` overrides). Two split
variants, written as `confident_learning.py` writes its own (val, test and test_heldout rows checked byte for byte):

- `adj-replace`: on contested rows the model scored, the target becomes the model's verdict; changed rows get
  `class_source = adjudicator`.
- `adj-soft`: a `soft_target` column for `--target soft`. Every train row gets the smoothed vote rate
  `(agree + 1) / (agree + disagree + 2)`. A contested scored row gets the model as `w` = 2 extra votes,
  `(agree + 1 + w * p_correct) / (agree + disagree + 2 + w)`. Example: 0 Agree, 2 Disagree and `p_correct` = 1
  give (0 + 1 + 2) / (0 + 2 + 2 + 2) = 0.5. `label_class` is unchanged, so validation stays hard.

Gold-set train rows keep their baseline membership, so the only difference from the baseline is the treatment
(`--drop-gold` marks them `train_dropped` instead, Q3).

<!-- BEGIN GENERATED: variants (analysis/adjudicator.py) -->

_No variants yet: `relabel` and `variants` run only for pairs that clear the bar._

<!-- END GENERATED: variants -->

## Retrain plan

Each variant trains with the confident-learning job and is scored on the round-3 rows with `evaluate.py --tag`, so
the comparison is the #76 report path:

```bash
LT=CurbRamp VARIANT=adj-replace DATA=$DATA RUNS=/gscratch/scrubbed/$USER/adj-2026-11 sbatch analysis/confident_learning.slurm
LT=CurbRamp VARIANT=adj-soft TARGET=soft DATA=$DATA RUNS=/gscratch/scrubbed/$USER/adj-2026-11 sbatch analysis/confident_learning.slurm
python analysis/confident_learning.py report --data-dir $DATA --label-types CurbRamp --variants adj-replace,adj-soft \
    --gold $OUT/gold_for_cl.csv --out docs/adjudicator-2026-11/retrain
```

`gold_for_cl.csv` (written by `gold`: the decided items as `label_uid, verdict`) stays outside the repository.
`fine_tune_classifier.soft_targets` reads a `soft_target` column when the split has one (a 5-line hook).
`evaluate.py` reads the hard `label_class` and is unchanged.

## Round 2 recount (#85)

The round-2 population weights and denominators are recounted without excluded authors in
`docs/ai-disagree-relabel-2026-10/round2_recount.md` ([#85](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/85),
separate PR). Nothing in this study depends on it: the gold set scores items, not population rates.

## Budget

Everything runs on `ckpt-all`, which is free. On the gold set, about 800 crops with two requests each:

- GLM, 4x H200: 0.031 s/crop for the verdict, about 0.5 s for the reason, plus about 47 min of load and JIT.
  About 4 GPU-hours.
- Gemma 4 31B, TP 2: about 1.5 to 2 s/crop at concurrency 16. About 1.5 GPU-hours.
- Qwen3-VL-32B: about 1.5 GPU-hours.

That is about 7 GPU-hours in total.

A pool run covers about 92k AI Disagrees, verdict only:

- GLM: about 3.5 GPU-hours.
- Gemma: about 8 GPU-hours, resumable across 8-hour jobs.

The crops dominate a pool run. `stage-list` prints the panorama count at 10 MB each (about 600 GB for 60k panoramas).
Run `--rows train` first, since those crops exist. Run the pool only for types that clear the bar, and only after Jon
approves the staging size. Each meta file's `s_per_crop` replaces these estimates once the jobs run.

## Provenance

<!-- BEGIN GENERATED: provenance (analysis/adjudicator.py) -->

_Not yet run._

<!-- END GENERATED: provenance -->

Gold inputs as built for the table above (sha256): round 1 `joined_scored.csv`
`145aac54b3e5c219fbd32a085471495c62fbc01e4baa8612ecc465cec21ebc0e`, round 2 `key.csv`
`57116bdfa129b66f6944e5520be772e314a046cec9189f22fdee56659e790aed`, `gold_shared.csv`
`c560bdc612ca1a156691d25857bab3961a818abdc4488cedf32267e3e9e585ab`, `unmasked.csv`
`de3e5fd60329e962fd99e700b2d049e3ca74c3bc0432b8861d6801d40aac7f95`, `relabel_pool.csv`
`dd5a6d363ce7f6c05ac6faf250f2526090bd2199541e6164b76029cd35483d09`, `cityparams.conf`
`9731618a43f90d1b3dde757f19f4420a6dc26c42ec0e69c0eae7572c2e0d246a`.

## Reproducing

```bash
# 1. the gold table (Mac; private inputs, private outputs)
R=~/Data/sidewalk-validator-ai-relabel-2026-10; OUT=~/Data/sidewalk-validator-ai-adjudicator-2026-11
DATA=~/Data/sidewalk-validator-ai-cl-2026-10/retrain-2026-10
python analysis/adjudicator.py gold --r1-joined $R/packet/joined_scored.csv \
    --r2-key $R/round2/private-key-do-not-share/key.csv \
    --r2-sheets jon=$R/round2/votes_jon.csv mikey=$R/round2/votes_mikey.csv \
    --r2-gold-shared $R/round2/adjudication/gold_shared.csv --r2-unmasked $R/round2/unmasked.csv \
    --split-root $DATA --pool $R/relabel_pool.csv --out-dir $OUT
python analysis/adjudicator.py crops --gold $OUT/gold.csv \
    --rawlabels ~/Data/sidewalk-validator-ai-retrain-2026-10/round3/rawlabels --out-dir $OUT/crops
# optional (Q7): the private items' geometry, one read-only makelab1 query; then rerun crops with --label-points
#   bash $OUT/crops/label_points_request.txt  (it holds the exact ssh command)

# 2. klone: copy gold.csv, crops/labels.csv, crops/stage_files.txt (and relabel_pool.csv + cityparams.conf for
#    relabel) to /gscratch/scrubbed/$USER/adjudicator-2026-11, then from the repo checkout:
J=$(sbatch --parsable analysis/adjudicator_stage.slurm)                   # pull at most 600 panoramas
AFTER=$J STAGE=crops analysis/adjudicator_submit.sh                       # cut the missing crops (1 GPU)
STAGE=score LIMIT=8 analysis/adjudicator_submit.sh gemma4-31b             # smoke (Q4), then:
STAGE=score analysis/adjudicator_submit.sh glm5.3-flash gemma4-31b qwen3vl-32b

# 3. back on the Mac (copy $OUT/scores back first)
python analysis/adjudicator.py agree --gold $OUT/gold.csv --scores $OUT/scores \
    --models glm5.3-flash,gemma4-31b,qwen3vl-32b --split-root $DATA --out-dir $OUT/agree
python analysis/adjudicator.py reliability --gold $OUT/gold.csv \
    --vote-weights ~/Data/sidewalk-validator-ai-retrain-2026-10/round3/reliability/vote_weights.csv \
    --rawlabels ~/Data/sidewalk-validator-ai-retrain-2026-10/round3/rawlabels --rated-after 2026-10-27 --out-dir $OUT/agree
python analysis/adjudicator.py disagreements --gold $OUT/gold.csv --scores $OUT/scores --model glm5.3-flash \
    --out-dir $OUT/packet_glm --key-dir $OUT/packet_glm_key --seed <secret>
python analysis/adjudicator.py report --agree-dir $OUT/agree --gold-summary $OUT/gold_summary.json \
    --scores $OUT/scores --variants-dir $DATA

# 4. only for pairs that clear the bar (klone, then the retrain plan above)
STAGE=relabel ROWS=train TYPES=CurbRamp,Crosswalk analysis/adjudicator_submit.sh glm5.3-flash
python analysis/adjudicator.py variants --split-root $DATA --relabel $OUT/relabel --model glm5.3-flash \
    --types CurbRamp,Crosswalk
```

## Open questions (defaults in use)

- **Q1 Tags in the prompt.** Off by default; `--show-tags` runs the `with-tags` variant.
- **Q2 The bar.** The paired lower bound above 0 on the same items. A weaker alternative: the lower bound of the
  model's agreement above the 74.5% point estimate (not paired).
- **Q3 Gold train rows in the variants.** Kept in training. `--drop-gold` leaves them out.
- **Q4 Gemma 4 31B under vLLM 0.32.** Not verified. The `LIMIT=8` smoke runs first. If the image rejects the
  architecture, MiniCPM-V 4.7 (0.742 on the scoreboard) replaces it.
- **Q5 A round-3 design with a `thin_voted` stratum.** Stratum code only (#85 PR), no design file.
- **Q6 Reasons on the pool.** Off (`relabel` defaults to the verdict only). Reasons are 10 to 15 times the cost.
- **Q7 Private-deployment round-1 items.** Scored only if `pull_label_points.sh` is run.
