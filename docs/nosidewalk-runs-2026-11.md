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
