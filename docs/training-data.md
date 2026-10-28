# Training data: labels, classes, cities and crop geometry

What `data_generation.py` and `split_data.py` turn the public `rawLabels` exports into, and why. The decisions
with a history behind them are in [`decisions.md`](decisions.md); how to run the steps is in
[`running-on-hyak.md`](running-on-hyak.md).

## Training signal

A label is used only when the crowd was decisive or an expert decided it, counting **human** votes only:

| class | rule | share of dataset |
|---|---|---|
| `incorrect` | disagree − agree > 1 | 45% |
| `correct` | agree − disagree > 2, then randomly downsampled | 55% |
| either | otherwise, with `EXPERTS` set: the class the expert accounts voted for, if they all agree and no other human voted the other way | (in the above) |
| (dropped) | unvalidated, unsure, or contested | — |

The AI's own votes are counted in the API's `agree_count` / `disagree_count` columns like any other user's, so
training counts votes from the `validations` column, where each vote carries a `validator_type`. Using the
aggregate columns would train the model on its own output.

`EXPERTS="<name>=<user_id>,..."` (the form `analysis/analyze.py` reads) is set in the environment, never in the
repo; unset, only the margin rule applies. A single non-expert vote is a weak signal (on labels the experts also
voted on, a lone Disagree was right 24–78% of the time depending on the type), which is why the margin rule stays
for the crowd. The manifest's `class_source` column records which rule decided each label, and the deployment prior
counts margin-rule labels only, since experts do not vote on a random sample.

The 55:45 ratio is a training-balance choice, not the real prior: among labels the crowd decided in Teaneck the
balance is roughly 90:10, a proxy for production traffic rather than a measurement of it. `data_generation.py`
therefore writes the pre-downsampling class counts to `deployment_priors.json`, which `calibration/calibrate.py`
reweights to ([#4](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/4)); whether that prior is the
right one is [#56](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/56).

> [!WARNING]
> Counting human votes only shrinks the classes, most of all CurbRamp `incorrect`, because the AI's disagree votes
> no longer count. Across 16 of the 19 original training cities the `incorrect` sets go from (API columns) to
> (human votes only): CurbRamp 7,957 → 2,626, NoCurbRamp 7,842 → 5,923, Obstacle 4,412 → 3,556, SurfaceProblem
> 4,688 → 3,623, Crosswalk 1,974 → 1,034. With the 55:45 rule the `correct` class shrinks with them, so a retrain
> has much less CurbRamp data than the counts in the API columns suggest.

### Exclusions

Labels from non-GSV imagery providers (Mapillary and Panoramax, so all of Richmond VA, Laurens IA and Bayonne FR;
why, and what would change it: [`decisions.md`](decisions.md)), rows with unusable pano geometry, and legacy labels
with unconverted coordinates (negative `pano_y`: Seattle 193, CDMX 14, Pittsburgh 8, SPGG 1) are excluded, and so
are labels placed by the `SidewalkAI` account (RampNet curb ramps in Vancouver WA, Richmond VA and Laurens IA),
because production never sends those to the validator; `--include-ai-labels` keeps them, tagged `author = ai` in
the manifest. (The round-3 CurbRamp run did include them; what the validator makes of AI-placed labels is being
measured in [#69](https://github.com/ProjectSidewalk/sidewalk-validator-ai/pull/69).)

## Cities

The default is every public deployment that `/v3/api/cities` lists with a `url` (41 on 2026-10-19), sorted by
city_id; `--cities a,b` overrides it. Private deployments are not in the default list (they carry no `url`); round
3 passed them explicitly, 54 deployments in all, and every published number pools or letters them. The
city_id is what `label_uid`, crop filenames and the pano-store directories use. With `--rawlabels-cache DIR` the
listing is read from `DIR/cities.json`, and every listed city must have its `DIR/<city_id>.csv` (a missing one
stops the run and names it; nothing is downloaded into a snapshot). A missing `cities.json` is fetched and saved
there, except into a directory that already holds CSVs (a cache from before city discovery), which takes
`--refresh-cities`, the flag that also refetches it. `deployment_priors.json` and `generation_run.json` record the
list and where it came from (`city_source`: API or cache, with the fetch date when known). A resume against a
different list stops, like a changed selection, and `--select-only` against such a run writes
`deployment_priors.partial.json`. A deployment the API lists without a `url` is reached with `--cities
<city_id>=<subdomain>` (the subdomain is the `sidewalk-<subdomain>.cs.washington.edu` host part).

Before [#8](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/8) the list was 19 hardcoded cities,
three of them private today. To reproduce or resume a dataset made with them, pass the ids recorded in that run's
`generation_run.json` (or `data_generation.py`'s history) with `--cities`; this needs no API call when their CSVs
are cached, and writes `deployment_priors.partial.json`, so copy the prior from the original run.

## Training targets

Experimental, [#12](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/12). The rule above decides
which labels are used and their `label_class`. What the model is trained *against* is chosen at training time with
`fine_tune_classifier.py --target`, so no crops need regenerating:

| `--target` | target probability of `correct` for a train row |
|---|---|
| `hard` (default) | 1 or 0 from `label_class`; behaviour, `run_key.json` and `training_run.json` are unchanged |
| `soft` | `(human_agree + 1) / (human_agree + human_disagree + 2)`, from `split.csv` (a 10–0 label → 0.92, a 3–0 label → 0.8) |
| `weighted` | the same formula on reliability-weighted counts `w_agree`/`w_disagree` from `--vote-weights` |

Round 3 trained with the default hard targets (its published `training_run.json` records no `target`). Soft
targets use soft-label cross-entropy against the two logits. Validation
stays hard (`eval_loss` and `macro_f1` on `label_class`), so checkpoint selection means the same thing for every
target. A non-default target, and the vote-weights file's hash, go into `run_key.json` (a resume never mixes
targets) and `training_run.json`. A non-hard target writes to `<label_type>/dinov2-<target>` by default, never to
`<label_type>/dinov2`, where `evaluate.py` and `publish_to_hf.py` look for the production model.

Per-validator reliability comes from the `validations` column of the cached rawLabels exports, human votes only
(`validator_type == "Human"`), Agree/Disagree only: each vote is compared with the majority of the *other* human
votes on the same label, scored only when there are at least 3 of them and they do not tie, and the hit rate is
smoothed towards the pooled rate `m` with a Beta prior of 10 pseudo-votes (`--prior-strength`, which must be > 0).
For `weighted`, each vote counts `r / m` rather than `r`, so an average validator's vote weighs 1 and the counts
stay in vote units. Summing raw reliabilities would mostly add shrinkage: the smoothed reliabilities are compressed
(p10 0.66, p90 0.86 on the 2026-10-17 data) and the `+1 / +2` prior is in raw-vote units, so every target would
just move towards 0.5. Even normalised, the weighting changes little on that data: over the 68,493 train-eligible
labels the weighted and soft targets correlate at 0.999, with a median absolute difference of 0.007.
`validator_reliability.py` writes the anonymous per-user scores and the per-label weights to a data directory
(never the repository: the files hold pseudonymous user ids). `--target weighted` checks that each train row's
`n_votes` equals split.csv's `human_agree + human_disagree`, so weights built from a newer rawLabels snapshot than
the manifest are refused.

```bash
python validator_reliability.py --rawlabels rawlabels/ --out reliability/
python fine_tune_classifier.py --label_type CurbRamp --target soft          # -> CurbRamp/dinov2-soft/
python fine_tune_classifier.py --label_type CurbRamp --target weighted \
    --vote-weights reliability/vote_weights.csv                             # -> CurbRamp/dinov2-weighted/
# Score an experiment on the same split; --tag keeps its outputs apart from the production model's.
# The gold slice: test labels with >= 3 human votes, every voter's reliability >= 0.85.
python evaluate.py --label_type CurbRamp --checkpoint CurbRamp/dinov2-soft/checkpoint-N --tag soft \
    --vote-weights reliability/vote_weights.csv
```

`checkpoint-N` is the `best_model_checkpoint` recorded in that directory's `training_run.json`.

### The gold slice

The `gold` slice defaults to `--gold-min-reliability 0.85` because higher bars leave almost nothing. Measured on
the 41-city 2026-10-17 cache: 13 validators score ≥ 0.95, 231 score ≥ 0.90 and 1,359 score ≥ 0.85. These are the
labels with at least 3 human votes where every voter is at or above the bar:

| bar | unanimous Agree | unanimous Disagree | mixed |
|---|---|---|---|
| 0.95 | 0 | 0 | 0 |
| 0.90 | 169 | 10 | 12 |
| 0.85 | 5,003 | 163 | 848 |

`incorrect` labels need only 2 votes (net disagree > 1), so the ≥ 3 rule leaves most of them out: with ≥ 2 votes
the unanimous-Disagree counts are 48 at 0.90 and 640 at 0.85. Expect a gold slice heavily weighted towards
`correct`. `evaluate.py` prints the gold count per split; with `--per-city` it also breaks the gold slice out per
city, which is meant to separate a city that is weak because of its model from one whose crowd labels are noisy
(counts for every city, metrics only where the city's gold rows reach `--min-city-n`; on the round-3 data the
held-out cities have 0–3 gold rows per type, so it does not yet answer that question there).

## Crop geometry

The panorama is resized to 8192×4096. A 2048² perspective view with a 90° horizontal field of view is rendered
**centred on the label** with `label_view` + `render_view` from
[`sidewalk-panorama-tools`](https://github.com/ProjectSidewalk/sidewalk-panorama-tools)
(`sidewalk_panorama_tools.geometry`, pinned to v1.0.0 in `requirements.txt`), the implementation the serving API
shares ([#5](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/5),
[sidewalk-ai-api#17](https://github.com/ProjectSidewalk/sidewalk-ai-api/pull/17)). Human labels are clicked in the
gravity-levelled viewer frame while the stored panorama is in the camera-rig frame, so the view is rendered with
the rig-tilt correction from the label's `camera_pitch`/`camera_roll` at `beta = 0.95`
([sidewalk-panorama-tools#54](https://github.com/ProjectSidewalk/sidewalk-panorama-tools/issues/54)); AI-authored
labels are already rig-frame and are rendered without it. Depth Anything V2 (metric, VKITTI, ViT-B as the API
serves, `max_depth = 80`) runs on the view (BGR, as `infer_image` expects); the depth at the centre sets the crop
half-width as `6100 / depth_m` pixels, clipped to the view (`clipped = 1` in the manifest), then the crop's long
side is resized to 640 px and saved as WebP quality 95.

Models trained on these crops expect tilt-corrected input, so the API's tilt correction (sidewalk-ai-api#17,
deployed 2026-10-27 with API 1.2.0) is a precondition for serving them. One residual difference remains by design:
training reads crops back from WebP quality 95, while the API classifies the in-memory PIL crop; on fresh crops the
two differ by a mean |Δ| of 1.1 grey levels. The stage-by-stage comparison of the two pipelines is
[`serving-parity-2026-10.md`](serving-parity-2026-10.md).
