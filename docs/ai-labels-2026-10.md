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
and Disagree votes were dropped. "Expert-decided" is the count of voted labels one of the two project leads voted on
with nobody contradicting ("lead-decided" below); the rest were decided by other accounts ("crowd only"). Vancouver and
Richmond are 4 % and 8 % validated; Laurens 75 %.

| city | lead votes / all votes | lead-decided | rejected by leads | other-decided | rejected by others | lead rejections in round-3 train / val / test |
|---|---:|---:|---:|---:|---:|---|
| laurens-ia | 1,179 / 1,185 (99%) | 1,177 | 101 (8.6%) | 3 | 0 (0.0%) | 0 / 0 / 0 |
| richmond-va | 983 / 1,072 (92%) | 983 | 48 (4.9%) | 89 | 3 (3.4%) | 0 / 0 / 0 |
| vancouver-wa | 1,499 / 2,818 (53%) | 1,499 | 29 (1.9%) | 1,295 | 52 (4.0%) | 21 / 4 / 4 |

The two leads cast 53 % of Vancouver's votes, 92 % of Richmond's and 99 % of Laurens's; the other Vancouver votes come
from 17 accounts, the largest with 37 % of them. The leads rejected 1.9 % of the Vancouver labels they decided, the
other accounts 4.0 %. The last column is the leakage mechanism explained under Method: every lead-rejected Vancouver
label is in a round-3 split.

Three things about this reference set:

- **One vote each.** Almost no label has two votes, so the reference is one person's opinion, and a single human vote
  on an AI label can lean toward Agree (sidewalk-auto-labeler's `docs/server-agree-check.md`). The lead-decided labels
  (the lead accounts are configured through `EXPERTS`, never named here) are reported as a second reference.
- **Not a random sample.** Labels reached their validator through the validation queue, so this is the validator's
  performance on the labels humans happened to see, not on all 79 k AI labels. Vancouver and Richmond are 4 % and 8 %
  validated.
- **Few Disagrees.** Humans rejected 2.9 % (Vancouver), 4.8 % (Richmond) and 8.6 % (Laurens) of the voted labels,
  against 46 % in the human-placed baseline. Agree precision is high almost by construction, and every precision-type
  number (Disagree precision, recall at precision ≥ 0.92) depends on that share, so the baseline is also restated at
  each slice's prevalence. The AUC does not depend on it.

**Providers.** Vancouver is GSV, the imagery every validator was trained on, and is the primary result. Richmond and
Laurens are Mapillary (an 11000 px rig and a GoPro Max), which `data_generation.py` excludes from training and which
production does not AI-validate at all ([SidewalkWebpage#4764](https://github.com/ProjectSidewalk/SidewalkWebpage/issues/4764)).

**Confidence tier.** RampNet's confidence is not stored with the label, and the run files with the detections live on
another machine, so each label's tier comes from the campaign that submitted it (the `send_to_ps.py` submission records,
`CAMPAIGNS` in the script): `core` = the 0.55 operating point, `band` = the 0.30–0.55 band added after
[sidewalk-auto-labeler#20](https://github.com/ProjectSidewalk/sidewalk-auto-labeler/issues/20), `mixed` = Richmond's 2026-09-24 re-submission of 14 repositioned sequences at 0.30. Vancouver was
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
   the model's own ranking (the log-odds of the two softmax columns, so the Disagrees that saturate float32 at
   `p_incorrect == 1.0` are not tied), and the deployed gate: each verdict's confidence served through the round-3
   CurbRamp accuracy map (`calibration/curbramp.json`, as shipped) and through the uncapped map. For CurbRamp the two
   maps are byte-identical: the `incorrect` class never reached 0.92 on the calibration split, so the cap does not bind.

**Leakage, and what it did to the reference.** The round-3 retrain selected with `--include-ai-labels`, so 150
Vancouver AI labels were in its train split and 34 in val (36 in test). The headline (`unseen`) excludes train and val;
`all scored` includes them. Richmond and Laurens never entered training (non-GSV). The selection was not random with
respect to the reference: `data_generation.eligible_labels` admits a label on a crowd vote margin or on a lead verdict,
so a one-vote crowd-decided AI label was never eligible, and the class balance keeps every `incorrect` label while
downsampling `correct`. Hence all 29 lead-rejected Vancouver labels are in a round-3 split (21 train, 4 val, 4 test),
and the 1,279 never-selected lead-decided labels are all approved. After the exclusion, the unseen Vancouver set holds
52 crowd-decided rejections and 4 lead-decided ones: **its headline is a crowd-reference number**, and no clean
lead-reference AUC exists for Vancouver.

**Baseline.** The same checkpoint on the round-3 **test** split of human-placed CurbRamp labels (all training
deployments; the held-out-city numbers are in `docs/retrain-2026-10.md`).

## Results

### Headline

`unseen` excludes the 184 Vancouver labels in the round-3 train and val splits; `all scored` includes them. The AUC is
for ranking the human-rejected labels; the interval resamples panoramas. "recall(incorrect) @ P ≥ 0.92" is the share of
rejected labels the model's own ranking can flag at 92 % precision, the bar the deployed gate sets.

| slice | n | incorrect | panos | AUC(incorrect) [95% pano CI] | acc | bal. acc | recall(incorrect) @ P ≥ 0.92 |
|---|---:|---:|---:|---:|---:|---:|---:|
| unseen, all three | 4,842 | 208 | 4,004 | 0.926 [0.904, 0.947] | 0.950 | 0.832 | 0.077 |
| unseen, laurens-ia | 1,180 | 101 | 621 | 0.930 [0.906, 0.953] | 0.891 | 0.855 | 0.059 |
| unseen, richmond-va | 1,072 | 51 | 916 | 0.977 [0.965, 0.987] | 0.970 | 0.807 | 0.039 |
| unseen, vancouver-wa | 2,590 | 56 | 2,467 | 0.851 [0.777, 0.916] | 0.969 | 0.774 | 0.089 |
| all scored, all three | 5,026 | 233 | 4,172 | 0.932 [0.911, 0.951] | 0.951 | 0.844 | 0.120 |
| all scored, laurens-ia | 1,180 | 101 | 621 | 0.930 [0.906, 0.953] | 0.891 | 0.855 | 0.059 |
| all scored, richmond-va | 1,072 | 51 | 916 | 0.977 [0.965, 0.987] | 0.970 | 0.807 | 0.039 |
| all scored, vancouver-wa | 2,774 | 81 | 2,635 | 0.891 [0.836, 0.938] | 0.970 | 0.829 | 0.222 |

Baseline, same checkpoint, same metrics:

| slice | n | incorrect | panos | AUC(incorrect) [95% pano CI] | acc | bal. acc | recall(incorrect) @ P ≥ 0.92 |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline: human-placed, round-3 test split (all training deployments) | 1,443 | 663 | 1,283 | 0.969 [0.960, 0.977] | 0.915 | 0.914 | 0.902 |

Baseline at matched prevalence (its negatives reweighted so the rejected share matches the AI slice; the AUC does not
depend on prevalence):

| slice | rejected share | Disagree precision, ungated | Disagree recall, ungated | recall(incorrect) @ P ≥ 0.92 |
|---|---:|---:|---:|---:|
| baseline as is | 45.9% | 0.908 | 0.906 | 0.902 |
| baseline reweighted to unseen, all three | 4.3% | 0.342 | 0.906 | 0.249 |
| baseline reweighted to unseen, laurens-ia | 8.6% | 0.520 | 0.906 | 0.380 |
| baseline reweighted to unseen, richmond-va | 4.8% | 0.367 | 0.906 | 0.249 |
| baseline reweighted to unseen, vancouver-wa | 2.2% | 0.204 | 0.906 | 0.249 |

**Vancouver (GSV, primary).** AUC 0.85 [0.78, 0.92] on the 2,590 unseen labels, against 0.97 for the same checkpoint on
human-placed labels. As explained under Method, 52 of the 56 rejections are crowd-decided, so this is the validator
against the non-lead accounts' votes. Against the leads' votes the model reaches 0.97 [0.95, 1.00] on the unseen
labels, but with 4 rejections, and 0.98 on the 184 train/val labels it has seen (25 rejections, accuracy 0.984), which
is why those are excluded. Whether the crowd or the model is right about the contested labels is not decidable from
one vote each (see the discussion).

**Richmond and Laurens (Mapillary, secondary).** AUC 0.98 [0.97, 0.99] and 0.93 [0.91, 0.95], both above Vancouver,
with Richmond at the human-placed baseline. The model has never trained on Mapillary imagery, and these are rig-frame
crops without tilt correction; it rank-orders RampNet's mistakes in them anyway. Both references are 92–99 % lead
votes.

**The top of the ranking is weaker against this reference, and prevalence explains only part of it.** The baseline
reweighted to the AI labels' 2–5 % rejected share still flags 25 % of its rejections at 92 % precision (38 % at
Laurens's 8.6 %); on the AI labels the model flags 4–9 %. Low prevalence does make the bar harder, but the baseline
clears it more often at the same prevalence. Two caveats: neither number has an interval (the reweighted 25 % is the
baseline's first 165 rejections ranked above its first approved label; Vancouver's 9 % is 5 of 56), and the references
differ in quality (baseline rejections are vote margins or lead verdicts, AI-label rejections mostly single crowd votes
with an Agree lean), so a few wrong approvals among the reference labels would lower precision at the top even for a
perfect ranking.

### At the deployed gate

Each verdict's confidence through the served round-3 CurbRamp accuracy map, gate at 0.92 (sidewalk-ai-api 1.2.0). The
map tops out at 0.80 for Disagree, so **no Disagree verdict on an AI label passes the gate as deployed**, and since the
uncapped map is identical, lifting the cap would change nothing. The "ungated" columns say what the raw Disagree
verdicts are worth with no gate at all.

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
  0.993 against a 0.978 human Agree share: the error rate among passed labels is 0.68 %, a third of the 2.2 % among all
  of them).
- Of the 56 human-rejected labels the gate auto-agrees with 16 and withholds 40 (71 %). It removes none of them, since
  it casts no Disagree; its value is triage: the 246 withheld labels contain 40 rejections (16 %, against 2.2 % overall).
- The model calls 89 labels wrong. Humans agreed with 32 of those (36 %). The baseline reweighted to Vancouver's 2.2 %
  rejected share gives 0.20 for the same verdict, so the Disagree verdicts on AI labels are no worse than on human-placed
  labels at matched prevalence; they are low-precision because rejections are rare. Of the 56 rejected labels the model
  names 32 (57 %).
- The 57 Disagree verdicts humans contradicted are not low-confidence: the median confidence of a contradicted Disagree
  is 0.9956, against 0.99993 for a confirmed one. The confidence table below shows the same: the verdicts on AI labels
  are as saturated as on human-placed ones, so a confidence map cannot separate them.

Richmond's Disagree verdicts are 0.71 precise (32 of 45; the reweighted baseline gives 0.37 at its prevalence),
Laurens's 0.43 (0.52 for the baseline at 8.6 %). The `band` tier (0.30–0.55 RampNet confidence) has higher Disagree
precision than `core` (0.52 vs 0.36) and recall (0.80 vs 0.59), but its AUC is no better (0.92 / 0.89 against
0.93 / 0.99 / 0.85 per city): the band holds 16 % rejections against 2.3 % in `core`, and precision follows prevalence.

### Slices

| slice | n | incorrect | human Agree share | AUC [95% pano CI] | recall(incorrect) @ P ≥ 0.92 | Agree coverage | Agree precision |
|---|---:|---:|---:|---:|---:|---:|---:|
| city / tier = laurens-ia / band | 580 | 84 | 0.855 | 0.917 [0.880, 0.949] | 0.214 | 0.704 | 0.984 |
| city / tier = laurens-ia / core | 600 | 17 | 0.972 | 0.925 [0.879, 0.966] | 0.059 | 0.800 | 0.996 |
| city / tier = richmond-va / band | 114 | 29 | 0.746 | 0.886 [0.815, 0.948] | 0.103 | 0.733 | 0.939 |
| city / tier = richmond-va / core | 948 | 22 | 0.977 | 0.990 [0.982, 0.996] | 0.000 | 0.955 | 0.999 |
| city / tier = vancouver-wa / core | 2,590 | 56 | 0.978 | 0.851 [0.777, 0.916] | 0.089 | 0.937 | 0.993 |
| votes = 1 | 4,814 | 207 | 0.957 | 0.926 [0.900, 0.948] | 0.077 | 0.898 | 0.993 |
| reference = crowd only | 1,381 | 55 | 0.960 | 0.848 [0.769, 0.917] | 0.127 | 0.940 | 0.987 |
| reference = expert-decided | 3,461 | 153 | 0.956 | 0.959 [0.948, 0.970] | 0.046 | 0.880 | 0.996 |
| city / reference = laurens-ia / expert-decided | 1,177 | 101 | 0.914 | 0.930 [0.906, 0.951] | 0.059 | 0.759 | 0.991 |
| city / reference = richmond-va / crowd only | 89 | 3 | 0.966 | 0.961 [0.899, 1.000] | 0.333 | 0.930 | 1.000 |
| city / reference = richmond-va / expert-decided | 983 | 48 | 0.951 | 0.977 [0.964, 0.988] | 0.042 | 0.936 | 0.994 |
| city / reference = vancouver-wa / crowd only | 1,289 | 52 | 0.960 | 0.841 [0.766, 0.914] | 0.135 | 0.942 | 0.986 |
| city / reference = vancouver-wa / expert-decided | 1,301 | 4 | 0.997 | 0.974 [0.948, 0.997] | 0.000 | 0.932 | 1.000 |
| pano size = 11000×5500 | 677 | 33 | 0.951 | 0.976 [0.958, 0.990] | 0.091 | 0.933 | 0.993 |
| pano size = 12288×6144 | 116 | 9 | 0.922 | 0.986 [0.961, 1.000] | 0.111 | 0.936 | 1.000 |
| pano size = 13312×6656 | 301 | 6 | 0.980 | 0.614 [0.383, 0.895] | 0.000 | 0.921 | 0.985 |
| pano size = 16384×8192 | 2,289 | 50 | 0.978 | 0.874 [0.789, 0.936] | 0.100 | 0.939 | 0.994 |
| pano size = 4096×2048 | 149 | 6 | 0.960 | 0.963 [0.895, 1.000] | 0.667 | 0.916 | 0.992 |
| pano size = 5760×2880 | 1,295 | 104 | 0.920 | 0.933 [0.910, 0.953] | 0.058 | 0.779 | 0.992 |

- **Reference.** Against the crowd-only reference the AUC is 0.85; against the lead-decided reference 0.96, at the
  human-placed baseline. The crowd-only slice is 93 % Vancouver (the leads decided nearly everything in the other two),
  and in Vancouver the lead-decided slice has 4 rejections because the round-3 selection took the other 25. The
  reference and city contrasts are therefore the same contrast seen twice, not two findings. Of the 89 Vancouver
  Disagree verdicts, 62 are on crowd-decided labels, where the human agreed with the model on 30, and 27 on
  lead-decided labels, where the lead agreed on 2 and had approved the other 25.
- **Tier.** The `band` tier holds 113 of the 208 unseen rejections in 694 labels. Its AUC is 0.92 (Laurens) and 0.89
  (Richmond); at 92 % precision the ranking flags 21 % and 10 % of its rejections, against 6 % and 0 % in the same
  cities' `core`. Richmond `core` is AUC 0.99.
- **Votes.** 4,814 of 4,842 unseen labels have exactly one Agree/Disagree vote; the ≥ 2 slice is too small to report.
- **Panorama size.** The GSV sizes split Vancouver: 16384×8192 (2,289 labels) AUC 0.87, 13312×6656 (301 labels, 6
  rejections) 0.61 [0.38, 0.90]. The 11000×5500 Richmond rig is 0.98, the 5760×2880 GoPro (Laurens plus part of
  Richmond) 0.93.
- **Human-placed labels in the same deployments** (27, none rejected) are listed in the tables for completeness and say
  nothing about the level-frame Mapillary case.

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

1. **The validator ranks RampNet's mistakes about as well as it ranks human mistakes when a project lead is the
   reference** (AUC 0.96 vs 0.97, though 149 of the 153 lead-rejected unseen labels are Mapillary), and worse against the crowd reference that decides Vancouver's unseen rejections
   (0.85). The two references cannot be compared on the same labels here (the round-3 selection consumed the
   lead-rejected Vancouver labels), and one vote per label cannot say whether the crowd or the model is right about the
   57 contested Vancouver Disagrees. The masked re-label below (the design of `docs/ai-disagree-relabel-2026-10.md`,
   [PR #66](https://github.com/ProjectSidewalk/sidewalk-validator-ai/pull/66), where a lead agreed with the production
   validator's Disagree on human-placed labels 58 % of the time) settled it for the 31 another account had approved: the
   crowd was right on 26 of 30.
2. **At the deployed gate, an Agree vote on an AI label is consistent with the single human vote 99.3 % of the time**,
   on a 97.8 % base rate, with 90 % coverage. That is one human vote, cast on an AI label, with the Agree lean that
   implies; it is not ground truth. The gate removes no false positive (it casts no Disagree on AI labels); what it does
   is sort: the 10 % it withholds carry 16 % rejections against 2 % overall.
3. **Disagree cannot fire as deployed.** The served map never reaches the gate for `incorrect`, and the uncapped map is
   the same. Sending AI labels to the validator today therefore means AI Agree votes and nothing else. If an AI Agree
   counts toward a label's vote total, that removes AI labels from the human queue on the strength of point 2; if it
   does not, nothing changes for humans. The vote-counting decision
   ([SidewalkWebpage#5790](https://github.com/ProjectSidewalk/SidewalkWebpage/issues/5790),
   [#4763](https://github.com/ProjectSidewalk/SidewalkWebpage/issues/4763)) matters more than the model numbers here.
   With no gate, the Disagree verdicts would be 36–71 % precise by city: above the human-placed baseline at the same
   prevalence in Vancouver (0.36 vs 0.20) and Richmond (0.71 vs 0.37), below it in Laurens (0.43 vs 0.52); they would
   name 57–81 % of the rejected labels.
4. **The 0.30–0.55 band is where a Disagree is worth most, because that is where the rejections are** (16 % of band
   labels against 2 % of `core`), not because the model ranks better there. If AI validation is to filter AI labels, the
   band is where it would pay, and a tier-aware gate (RampNet's confidence is known at submission time, even though it
   is not stored on the label) is a cheap experiment.
5. **Rig-frame AI labels on Mapillary imagery rank as well as GSV ones without retraining** (Richmond 0.98, Laurens
   0.93), but the gate's coverage drops to 76 % on the GoPro imagery (possibly because the map was fitted on GSV
   confidences, possibly because Laurens has 3–4× the rejected share; Richmond's rig keeps 94 %) and
   human-placed, level-frame Mapillary labels, the case behind
   [SidewalkWebpage#4764](https://github.com/ProjectSidewalk/SidewalkWebpage/issues/4764), are untested here (26
   labels, none rejected). This says AI-placed Mapillary labels could be AI-validated; it says nothing about human
   ones.

### Masked re-label of the contested Vancouver labels (2026-10-28)

The question left open above was settled the same day: a project lead (the rater) re-voted, on the live site with
every existing vote masked, on the unseen Vancouver AI labels the validator and the one human vote disagreed about,
plus decoys (`analysis/ai_labels_relabel.py`; results as the script emits them in
[`ai-labels-2026-10/relabel_results.md`](ai-labels-2026-10/relabel_results.md), provenance in
[`relabel_manifest.json`](ai-labels-2026-10/relabel_manifest.json)). Labels the rater had ever voted on were left out
(1,284 of the unseen pool, including 26 of the 57 contested labels and 2 of the 32 confirmed ones), so every original
vote in the packet is another account's; the rater's vote is the record, read back from the rawLabels API (each of the
132 rated labels gained exactly one vote, the rater's, and nothing else changed). One list of 133 labels: 31 contested
(validator Disagree, human Agree), 30 confirmed (both Disagree), 22 missed (validator Agree, human Disagree) and 50
decoys (both Agree). 132 were voted, 3 of them Unsure.

**Ordering flaw, disclosed.** The list was meant to be one seeded shuffle, but the build sorted the packet by the same
hash it had used to pick the lowest-ranked decoys, so the 50 decoys came first (positions 1–55) and the 78 disputed
labels after. The rater therefore saw an easy block and then a hard block. Contested, confirmed and missed are
interleaved within the hard block, so the comparison that matters (contested against confirmed) stayed masked, and a
late-list drift toward Disagree would work against the crowd's Agrees, so the headline below is if anything
conservative. The code now orders by an independent hash (tested); the ratings were not redone.

| cell | n | voted | Unsure | rater sides with the validator | rater sides with the original vote |
|---|---:|---:|---:|---|---|
| contested | 31 | 31 | 1 | 4/30 = 13 % [5, 30] | 26/30 = 87 % [70, 95] |
| confirmed | 30 | 30 | 1 | 27/29 = 93 % [78, 98] | 27/29 = 93 % [78, 98] |
| missed | 22 | 22 | 1 | 15/21 = 71 % [50, 86] | 6/21 = 29 % [14, 50] |
| decoy | 50 | 49 | 0 | 48/49 = 98 % [89, 100] | 48/49 = 98 % [89, 100] |

- **The crowd was right about the contested labels.** On 26 of 30 the rater agreed with the original Agree vote; the
  validator's contested Disagrees are wrong 87 % of the time. The rater's own earlier, unmasked votes on the 26
  contested labels left out say the same: 25 Agree, 1 Unsure.
- **Disagree precision against the rater is the crowd's number.** Packet only, 31/59 = 53 % [40, 65], but the packet
  is enriched in confirmed Disagrees (the exclusion removed 26 of 57 contested and 2 of 32 confirmed). Folding in the
  rater's prior votes on the 28 left out, the validator's Disagree is right on 33 of 86 decided labels, 38 % [29, 49],
  against 36 % for the crowd reference in the headline tables.
- **Where the validator approved and a single crowd vote rejected, the crowd was mostly wrong:** on 15 of 21 the rater
  sided with the validator. Where both rejected, the rater backed the crowd 27 of 29 times. So the crowd reference
  understates the validator on its Agree side: some of the 56 Vancouver "rejections" in the headline tables are the
  crowd's errors, and the 0.85 AUC against that reference is pessimistic. It does not change the Disagree verdicts,
  which were wrong on the contested labels either way.
- Overall the rater agreed with the original human vote on 107 of 129 decided labels (83 %) and with the validator on
  94 (73 %).
- **Confidence.** Within the validator's Disagrees the rater confirmed 23 of 32 above `p_incorrect` 0.999, 6 of 13
  between 0.99 and 0.999 and 2 of 14 below; with the prior votes folded in, 24 of 43, 6 of 18 and 3 of 25. The crowd
  reference shows the same gradient over all 89 Disagrees (23/45, 5/18, 4/26), so this is not new, and the top bin at
  56 % [41, 70] is far from the 0.92 gate. For the record, a contested Disagree has median confidence 0.9956 and a
  crowd-confirmed one 0.99993.

Caveats: one rater, one city, 89 Disagrees in all; the rater is a project lead whose votes are part of the reference
the round-3 model trained on; the ordering flaw above.

**Limits.** One vote per label, 53–99 % of them from two accounts; a queue-selected 4–75 % of the AI labels; 2–9 %
rejections, so the Disagree numbers rest on 56, 51 and 101 labels; no severity or tag slices; the Vancouver headline is
a crowd-reference number because the round-3 selection took every lead-rejected label; the Mapillary crops are untilted
rig-frame renders, as the 150 training AI crops are, but of imagery no training image resembles. The masked re-label above covers one city and one rater; in the other two deployments the lead who would rate cast nearly every original vote, so no masked re-label is possible there under the no-prior-votes rule.

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
# masked re-label (key stays outside the repo; pull rawLabels right before building and again after rating)
RATER=<user_id> EXPERTS=... python analysis/ai_labels_relabel.py build --run RUN --city vancouver-wa \
    --out RUN/relabel --rawlabels-before RUN/relabel/rawlabels_before/vancouver-wa.csv
RATER=<user_id> python analysis/ai_labels_relabel.py score --run RUN --out RUN/relabel
```

Unit tests: `pytest -q tests/test_ai_labels_eval.py tests/test_ai_labels_relabel.py`.
