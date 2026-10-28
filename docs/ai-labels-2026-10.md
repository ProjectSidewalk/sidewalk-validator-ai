# The round-3 CurbRamp validator on AI-placed curb ramps (October 2026)

**Status:** scored 2026-10-28 UTC on klone with `analysis/ai_labels_eval.py`; tables in
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

| city | provider | AI CurbRamp labels | human-voted | panos | Disagree majority | ≥ 2 votes | expert-decided | core / band / mixed | in round-3 train / val / test | cropped | not scored |
|---|---|---:|---:|---:|---:|---:|---:|---|---|---:|---:|
| laurens-ia | mapillary | 1,575 | 1,180 | 621 | 101 (8.6%) | 5 | 1,177 | 600 / 580 / 0 | 0 / 0 / 0 | 1,180 | 0 |
| richmond-va | mapillary | 12,962 | 1,072 | 916 | 51 (4.8%) | 0 | 983 | 948 / 114 / 10 | 0 / 0 / 0 | 1,072 | 0 |
| vancouver-wa | gsv | 64,814 | 2,794 | 2,654 | 81 (2.9%) | 24 | 1,499 | 2794 / 0 / 0 | 150 / 34 / 36 | 2,774 | 20 |

20 Vancouver labels sit on panoramas the lab's store no longer holds and were not scored; three labels with equal Agree
and Disagree votes were dropped. "Expert-decided" is the share of voted labels one of the two project leads voted on with
nobody contradicting; the rest were decided by other accounts ("crowd only" below), which in Vancouver means mostly one
non-lead account. Vancouver and Richmond are 4 % and 8 % validated; Laurens 75 %.

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

### Headline

`unseen` excludes the 184 Vancouver labels in the round-3 train and val splits; `all scored` includes them. The AUC is
for ranking the human-rejected labels; the interval resamples panoramas. "recall(incorrect) @ P ≥ 0.92" is the share of
rejected labels the model's own ranking can flag at 92 % precision, the bar the deployed gate sets.

| slice | n | incorrect | panos | AUC(incorrect) [95% pano CI] | acc | bal. acc | recall(incorrect) @ P ≥ 0.92 |
|---|---:|---:|---:|---:|---:|---:|---:|
| unseen, all three | 4,842 | 208 | 4,004 | 0.926 [0.904, 0.947] | 0.950 | 0.832 | 0.000 |
| unseen, laurens-ia | 1,180 | 101 | 621 | 0.930 [0.906, 0.953] | 0.891 | 0.855 | 0.000 |
| unseen, richmond-va | 1,072 | 51 | 916 | 0.977 [0.965, 0.987] | 0.970 | 0.807 | 0.000 |
| unseen, vancouver-wa | 2,590 | 56 | 2,467 | 0.851 [0.777, 0.915] | 0.969 | 0.774 | 0.000 |
| all scored, all three | 5,026 | 233 | 4,172 | 0.932 [0.910, 0.951] | 0.951 | 0.844 | 0.000 |
| all scored, laurens-ia | 1,180 | 101 | 621 | 0.930 [0.906, 0.953] | 0.891 | 0.855 | 0.000 |
| all scored, richmond-va | 1,072 | 51 | 916 | 0.977 [0.965, 0.987] | 0.970 | 0.807 | 0.000 |
| all scored, vancouver-wa | 2,774 | 81 | 2,635 | 0.891 [0.836, 0.938] | 0.970 | 0.829 | 0.000 |

Baseline, same checkpoint, same metrics:

| slice | n | incorrect | panos | AUC(incorrect) [95% pano CI] | acc | bal. acc | recall(incorrect) @ P ≥ 0.92 |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline: human-placed, round-3 test split (all training deployments) | 1,443 | 663 | 1,283 | 0.968 [0.959, 0.976] | 0.915 | 0.914 | 0.902 |

**Vancouver (GSV, primary).** AUC 0.85 [0.78, 0.92] on the 2,590 unseen labels, against 0.97 for the same checkpoint on
human-placed labels. The gap is entirely in the crowd-referenced half (next section): against the project leads' votes
the AUC is 0.97 [0.95, 1.00], but those 1,301 labels contain only 4 rejections, so that number says little. On the 184
train/val Vancouver labels the model scores AUC 0.98 and accuracy 0.984, which is why they are excluded.

**Richmond and Laurens (Mapillary, secondary).** AUC 0.98 [0.97, 0.99] and 0.93 [0.91, 0.95], both above Vancouver,
and Richmond is at the human-placed baseline. The model has never trained on Mapillary imagery, and these are rig-frame
crops without tilt correction; it rank-orders RampNet's mistakes in them anyway.

**Nothing reaches 92 % precision on rejections.** With 2–9 % of the labels rejected, the model's ranking cannot flag
rejected labels at 92 % precision in any city (0.000 in every headline row), while on the human-placed test split, where
46 % are rejected, it flags 90 % of them. The Disagree side of the validator is a base-rate problem on AI labels,
not only a model problem.

### At the deployed gate

Each verdict's confidence through the served round-3 CurbRamp accuracy map, gate at 0.92 (sidewalk-ai-api 1.2.0). The
served map tops out at 0.80 for Disagree (the uncapped map is byte-identical; `incorrect` never reached 0.92 on the
calibration split), so **no Disagree verdict on an AI label can pass the gate as deployed**. The "ungated" columns say
what the raw Disagree verdicts are worth if that cap were lifted.

| slice | n | Agree verdicts | Agree pass | Agree coverage | Agree precision [Wilson] | Disagree verdicts | Disagree precision, ungated [Wilson] | Disagree recall, ungated | Disagree pass (served map) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| unseen, all three | 4,842 | 4516 | 4054 | 0.898 | 0.993 [0.990, 0.995] | 326 | 0.448 [0.395, 0.502] | 0.702 | 0 |
| unseen, laurens-ia | 1,180 | 988 | 749 | 0.758 | 0.991 [0.981, 0.996] | 192 | 0.427 [0.359, 0.498] | 0.812 | 0 |
| unseen, richmond-va | 1,072 | 1027 | 961 | 0.936 | 0.995 [0.988, 0.998] | 45 | 0.711 [0.566, 0.823] | 0.627 | 0 |
| unseen, vancouver-wa | 2,590 | 2501 | 2344 | 0.937 | 0.993 [0.989, 0.996] | 89 | 0.360 [0.268, 0.463] | 0.571 | 0 |
| unseen, tier core | 4,138 | 3984 | 3674 | 0.922 | 0.995 [0.992, 0.997] | 154 | 0.364 [0.292, 0.442] | 0.590 | 0 |
| unseen, tier band | 694 | 522 | 370 | 0.709 | 0.976 [0.954, 0.987] | 172 | 0.523 [0.449, 0.597] | 0.796 | 0 |
| unseen, tier mixed | 10 | 10 | 10 | 1.000 | 1.000 [0.723, 1.000] | 0 | — | — | 0 |
| baseline: human-placed round-3 test | 1,443 | 781 | 663 | 0.849 | 0.962 [0.945, 0.974] | 662 | 0.908 [0.883, 0.928] | 0.906 | 0 |

In words, for the unseen Vancouver labels:

- The gate passes an Agree vote on 2,344 of 2,590 labels (90 %); 16 of those are labels a human rejected (precision
  0.993, against a 0.978 human Agree share, so the gate removes about a third of the rejected labels from what it
  auto-agrees with, not most of them).
- The model calls 89 labels wrong. Humans agreed with 32 of those (36 %); on the human-placed test split the same
  verdict is right 91 % of the time. Of the 56 human-rejected labels it names 32 (57 %); 24 get an Agree verdict and 16
  of those pass the gate.
- The 57 Disagree verdicts humans contradicted are not low-confidence: the median confidence of a contradicted Disagree
  is 0.995, against 0.99999 for a confirmed one. The confidence table below shows the same: the verdicts on AI labels are
  as saturated as on human-placed ones.

Richmond's Disagree verdicts are better (precision 0.71, 32 of 45), Laurens's are like Vancouver's (0.43), and the
0.30–0.55 `band` tier is where the Disagree verdicts earn their keep (precision 0.52, recall 0.80, against 0.36 / 0.59
on `core`): the band is where RampNet itself is 75–85 % precise, so there are rejections to find.

### Slices

| slice | n | incorrect | human Agree share | AUC [95% pano CI] | recall(incorrect) @ P ≥ 0.92 | Agree coverage | Agree precision |
|---|---:|---:|---:|---:|---:|---:|---:|
| city / tier = laurens-ia / band | 580 | 84 | 0.855 | 0.917 [0.880, 0.948] | 0.155 | 0.704 | 0.984 |
| city / tier = laurens-ia / core | 600 | 17 | 0.972 | 0.925 [0.879, 0.966] | 0.000 | 0.800 | 0.996 |
| city / tier = richmond-va / band | 114 | 29 | 0.746 | 0.886 [0.815, 0.948] | 0.103 | 0.733 | 0.939 |
| city / tier = richmond-va / core | 948 | 22 | 0.977 | 0.990 [0.982, 0.996] | 0.000 | 0.955 | 0.999 |
| city / tier = vancouver-wa / core | 2,590 | 56 | 0.978 | 0.851 [0.777, 0.915] | 0.000 | 0.937 | 0.993 |
| votes = 1 | 4,814 | 207 | 0.957 | 0.926 [0.900, 0.948] | 0.000 | 0.898 | 0.993 |
| reference = crowd only | 1,381 | 55 | 0.960 | 0.848 [0.769, 0.917] | 0.000 | 0.940 | 0.987 |
| reference = expert-decided | 3,461 | 153 | 0.956 | 0.959 [0.948, 0.969] | 0.000 | 0.880 | 0.996 |
| city / reference = laurens-ia / expert-decided | 1,177 | 101 | 0.914 | 0.930 [0.906, 0.951] | 0.000 | 0.759 | 0.991 |
| city / reference = richmond-va / crowd only | 89 | 3 | 0.966 | 0.961 [0.899, 1.000] | 0.333 | 0.930 | 1.000 |
| city / reference = richmond-va / expert-decided | 983 | 48 | 0.951 | 0.977 [0.964, 0.988] | 0.000 | 0.936 | 0.994 |
| city / reference = vancouver-wa / crowd only | 1,289 | 52 | 0.960 | 0.841 [0.766, 0.914] | 0.000 | 0.942 | 0.986 |
| city / reference = vancouver-wa / expert-decided | 1,301 | 4 | 0.997 | 0.974 [0.948, 0.997] | 0.000 | 0.932 | 1.000 |
| pano size = 11000×5500 | 677 | 33 | 0.951 | 0.976 [0.958, 0.990] | 0.091 | 0.933 | 0.993 |
| pano size = 12288×6144 | 116 | 9 | 0.922 | 0.985 [0.960, 1.000] | 0.000 | 0.936 | 1.000 |
| pano size = 13312×6656 | 301 | 6 | 0.980 | 0.614 [0.383, 0.895] | 0.000 | 0.921 | 0.985 |
| pano size = 16384×8192 | 2,289 | 50 | 0.978 | 0.874 [0.789, 0.936] | 0.000 | 0.939 | 0.994 |
| pano size = 4096×2048 | 149 | 6 | 0.960 | 0.963 [0.895, 1.000] | 0.667 | 0.916 | 0.992 |
| pano size = 5760×2880 | 1,295 | 104 | 0.920 | 0.933 [0.910, 0.953] | 0.000 | 0.779 | 0.992 |

- **Reference.** Against the crowd-only reference the AUC is 0.85; against the expert-decided reference 0.96, at the
  human-placed baseline. In Vancouver the two halves are the same size, and the leads rejected 4 of 1,301 labels while
  the other accounts rejected 52 of 1,289. Either the two groups saw different labels (the queue decides that), or they
  apply a different bar; this experiment cannot tell, and it is the reason the Vancouver headline is 0.85 and not 0.97.
  Of the 89 Vancouver Disagree verdicts, 62 are on crowd-referenced labels, where the human agreed with the model on 30,
  and 27 on expert-referenced labels, where the lead agreed on 2.
- **Tier.** The `band` tier (0.30–0.55 RampNet confidence) holds 113 of the 208 unseen rejections in 694 labels; its AUC
  is 0.92 (Laurens) and 0.89 (Richmond), the only slices where the ranking flags any rejections at 92 % precision
  (16 % and 10 %). Richmond `core` is 0.99.
- **Votes.** 4,814 of 4,842 unseen labels have exactly one Agree/Disagree vote; the ≥ 2 slice is too small to report.
- **Panorama size.** The GSV sizes split Vancouver: 16384×8192 (2,289 labels) AUC 0.87, 13312×6656 (301 labels, 6
  rejections) 0.61 [0.38, 0.90]. The 11000×5500 Richmond rig is 0.98, the 5760×2880 GoPro (Laurens plus part of
  Richmond) 0.93.
- **Human-placed labels in the same deployments** (27, none rejected) are listed in the tables for completeness and say
  nothing.

| slice | n | p5 | p25 | p50 | p75 | p95 |
|---|---:|---:|---:|---:|---:|---:|
| unseen AI-placed, Agree verdicts | 4,516 | 0.9980 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| unseen AI-placed, Disagree verdicts | 326 | 0.6781 | 0.9679 | 0.9995 | 1.0000 | 1.0000 |
| baseline Agree verdicts | 781 | 0.9834 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| baseline Disagree verdicts | 662 | 0.9635 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

The verdicts on AI-placed labels are as saturated as on human-placed ones (p25 of the Disagree confidence 0.97), so the
served map, which is a function of confidence, cannot separate the Disagree verdicts humans confirmed from the ones they
contradicted.

### What this says, and what it cannot

1. **The validator ranks RampNet's mistakes about as well as it ranks human mistakes** when the reference is a project
   lead (AUC 0.96 vs 0.97), and worse when it is the crowd reference that Vancouver's Disagrees mostly come from (0.85).
   Which of the two is right about the contested labels is not decidable from one vote each. A masked re-label of a
   sample of the 57 contradicted Vancouver Disagrees would settle it; the same design as `docs/ai-disagree-relabel-2026-10.md`
   (where a lead agreed with the production validator's Disagree on human-placed labels 58 % of the time) applies as is.
2. **Agree is safe to automate on AI labels at the deployed gate, within the limits of this reference.** The gate
   auto-agrees with 90 % of Vancouver's AI labels at 0.993 precision; the base rate is 0.978, so the labels it passes are
   not much cleaner than the labels it sees. If the point of validating AI labels is to remove RampNet's false positives,
   an Agree-only validator removes about a third of them at this operating point.
3. **Disagree cannot fire as deployed and would be about 36–71 % precise if it could.** Sending AI labels to the
   validator today therefore means: AI labels get AI Agree votes and nothing else, since the map holds every Disagree
   back. That changes nothing about which AI labels humans see, and if an AI Agree counts toward a label's vote total it
   removes AI labels from the human queue on the strength of a 0.993-precision vote on a 0.978 base rate. The decision
   on vote counting (SidewalkWebpage#5790, #4763) matters more than the model numbers here.
4. **The band is the place to look.** RampNet's 0.30–0.55 band is where its own precision drops to 75–85 % and where
   the validator's Disagree is most precise (0.52) and most complete (0.80). If AI validation is to filter AI labels, the
   band is where it would pay, and a band-only or tier-aware gate is a cheap experiment.
5. **Mapillary works without retraining**, at least for rig-frame AI labels: Richmond matches the GSV baseline and
   Laurens's GoPro imagery is at 0.93. Production does not AI-validate Mapillary deployments at all (SidewalkWebpage#4764);
   for AI-placed labels there is no measured reason for that.

**Limits.** One vote per label, mostly from one account per deployment; a queue-selected 4–75 % of the AI labels;
2–9 % rejections, so the Disagree numbers rest on 56, 51 and 101 labels; no severity or tag slices; the Vancouver AI
labels were in the round-3 training pool until excluded here, and the Mapillary crops are untilted rig-frame renders
that no training image resembles. Nothing here was re-labelled; see point 1.

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
