# The round-3 CurbRamp validator on AI-placed curb ramps (October 2026)

**Status:** measured 2026-10-28 on klone with `analysis/ai_labels_eval.py`; tables in
[`ai-labels-2026-10/tables.md`](ai-labels-2026-10/tables.md), figure in
[`ai-labels-2026-10/roc_confidence.png`](ai-labels-2026-10/roc_confidence.png). Run directory (private, not in the
repo): `~/Data/sidewalk-validator-ai-ai-labels-2026-10/` on the Mac, `/gscratch/makelab/jfroehli/ai-labels-2026-10/`
on klone. No new ground truth was collected: the reference is the human votes already on the server.

## Question

Project Sidewalk's AI validator has never seen an AI-placed label. `LabelTable.getLabelsToValidateWithAi` in
SidewalkWebpage skips labels whose author has the `Ai` role, so none of the ~1.03 M `label_ai_assessment` rows is on one,
and the RampNet curb ramps that [sidewalk-auto-labeler](https://github.com/ProjectSidewalk/sidewalk-auto-labeler) submits
reach human validators only. Before changing that, the question is whether the retrained CurbRamp validator
(`docs/retrain-2026-10.md`, round 3) can tell RampNet's right curb ramps from its wrong ones at all, and what its
deployed gate would do on them.

## Data

Three deployments hold AI-placed labels, all CurbRamp, all by the `SidewalkAI` account. Every label with at least one
human Agree/Disagree vote is scored; the reference is the human majority (Unsure ignored, ties dropped; votes by the AI
validator excluded as in `data_generation.human_vote_counts`).

RESULTS_POPULATION

Three things about this reference set:

- **One vote each.** Almost no label has two votes, so the reference is one person's opinion; and one account cast most of
  the votes in every deployment (sidewalk-auto-labeler's `docs/server-agree-check.md`). An "expert-decided" column
  (labels one of two project leads voted on, nobody contradicting; the lead accounts are configured through `EXPERTS`,
  never named here) is reported as a second reference.
- **Not a random sample.** Labels reached their validator through the validation queue, so this is the validator's
  performance on the labels humans happened to see, not on all 79 k AI labels. Vancouver and Richmond are 4 % and 8 %
  validated.
- **Few Disagrees.** Humans rejected 2.9 % (Vancouver), 4.8 % (Richmond) and 8.6 % (Laurens) of the voted labels. Agree
  precision is therefore high almost by construction; the informative numbers are the AUC (can the model rank the
  rejected ones?) and the Disagree catch.

**Providers.** Vancouver is GSV, the imagery every validator was trained on, and is the primary result. Richmond and
Laurens are Mapillary (an 11000 px rig and a GoPro Max), which `data_generation.py` excludes from training and which
production does not AI-validate at all (SidewalkWebpage#4764).

**Confidence tier.** RampNet's confidence is not stored with the label, and the run files with the detections live on
another machine, so each label's tier comes from the campaign that submitted it (the `send_to_ps.py` submission records,
`CAMPAIGNS` in the script): `core` = the 0.55 operating point, `band` = the 0.30–0.55 band added after
sidewalk-auto-labeler#20, `mixed` = Richmond's 2026-09-24 re-submission of 14 repositioned sequences at 0.30. Vancouver was
labelled in September 2025 at 0.55 and has no band.

**Severity and tags.** RampNet submits neither, so the type × severity and type × tag slices of the standing rule cannot
be made here; the handful of AI labels a validator edited afterwards are counted in the tables.

## Method

1. `select`: fresh `rawLabels` pulls of the three deployments (sha256s in the provenance section of the tables), the
   label set above, and the list of panoramas to stage.
2. Panoramas staged from the lab's pano store to klone (`analysis/ai_labels_stage.slurm`, the retrain's recipe).
3. `crops`: the crop is cut by `data_generation.py`'s own functions (`read_store_panorama`, `create_perspective_crop`,
   `save_webp`) with the round-3 settings (tilt β 0.95, Depth Anything V2 vitb, sidewalk-panorama-tools v1.0.0), so it
   matches what the API would cut (`docs/serving-parity-2026-10.md`). AI labels are **rig-frame** points, so they are
   rendered without the tilt correction, as `data_generation.py` renders `author == ai` rows and as sidewalk-ai-api
   renders `coord_frame == "rig"`. That needs no pose, which is why the Mapillary rows can be rendered although their
   level-frame pose convention is unhandled for human labels.
4. `score`: `evaluate.run_inference` with the round-3 CurbRamp checkpoint (the weights published to Hugging Face on
   2026-10-26 and served by sidewalk-ai-api 1.2.0 since 2026-10-27).
5. `report`: AUC(incorrect) with panorama-resampled 95 % intervals, accuracy, recall of incorrect at precision ≥ 0.92 on
   the model's own ranking, and the deployed gate: each verdict's confidence served through the round-3 CurbRamp
   accuracy map (`calibration/curbramp.json`, capped so Disagree is held back, as shipped) and through the uncapped map
   (what Disagree would do if it were not held back).

**Leakage.** The round-3 retrain selected with `--include-ai-labels`, so 150 Vancouver AI labels were in its train split
and 34 in val (36 in test). The headline (`unseen`) excludes train and val; `all scored` includes them. Richmond and
Laurens never entered training (non-GSV).

**Baseline.** The same checkpoint on the round-3 **test** split of human-placed CurbRamp labels (all training
deployments; the held-out-city numbers are in `docs/retrain-2026-10.md`).

## Results

RESULTS_HEADLINE

RESULTS_GATE

RESULTS_SLICES

RESULTS_DISCUSSION

## Reproduce

```
EXPERTS="<role>=<user_id>,<role>=<user_id>" python analysis/ai_labels_eval.py select \
    --rawlabels RUN/rawlabels --round3-split ROUND3/CurbRamp/split.csv --out RUN
# klone, from a checkout with the submodule and the depth checkpoint:
RUN=/gscratch/makelab/jfroehli/ai-labels-2026-10 sbatch analysis/ai_labels_stage.slurm
RUN=... CKPT=.../retrain-2026-10/CurbRamp/dinov2/checkpoint-4340 sbatch analysis/ai_labels_gen.slurm
# Mac:
python analysis/ai_labels_eval.py report --run RUN --map ROUND3/CurbRamp/calibration/curbramp.json \
    --map-uncapped ROUND3/CurbRamp/calibration/uncapped/curbramp.json --round3 ROUND3/CurbRamp --out docs/ai-labels-2026-10
```

Unit tests: `pytest -q tests/test_ai_labels_eval.py`.
