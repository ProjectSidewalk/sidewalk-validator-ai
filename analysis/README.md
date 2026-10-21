# Production analysis scripts

Scripts behind the October 2026 production measurements (`docs/calibration-2026-10.md`). They contain no data.
Their inputs and outputs include private-city labels, so keep them **outside the repo** (e.g. `~/Data/...`).

| script | what it does |
|---|---|
| `pull_ai_assessments.sh` | On the database host, read-only: one row per AI-assessed label across all city schemas (raw verdict, served accuracy, confidence, human votes excluding the AI) → `ai_assessments.csv`. Optional `EXPERT1_ID`/`EXPERT2_ID` env vars add two validators' own votes. |
| `analyze_calib.py` | Served accuracy vs observed human-majority agreement, gate pass rates, reliability bins, per-city precision, from `ai_assessments.csv` → `calib.txt`. |
| `fetch_all.py` | Downloads the public `rawLabels` CSV for every public city into a cache directory (tiling by bbox if an export truncates). |
| `analyze.py` | AI vs human majority (and vs expert validators given as `EXPERTS="name=user_id,..."`) from that cache → `analysis.txt`, `ai_vs_humans.csv`. Expert columns are always role-keyed `expert1`, `expert2`, … in `EXPERTS` order, whatever the names; `hard_examples.py` reads those names. |
| `calibration_doc_numbers.py` | Prints every figure in `docs/calibration-2026-10.md` that is not in `summary.md` (city mix, precision by vote count, pooled calibration error, confidence quantiles, model-swap window, NoCurbRamp drop, Triage what-if), from `calibrate_from_production`'s `labels_oof.csv`. Names public deployments only (`--cities-json`). |
| `road_prior.py` | Issue #45, first measurement: street-network / OSM prior for CurbRamp (driveway vs curb ramp). Stages `fetch-streets` (each public city's `/v3/api/streets` CSV; intersections = shared edge endpoints), `fetch-osm` (optional, Overpass: crossings and driveway ways), `features` (private label-level CSV; expert votes from `EXPERTS`), `evaluate` (per-feature AUC, leave-one-city-out lift over the production confidence, Disagree-vs-distance plots, Crosswalk check), `stratify` (far-from-intersection shares of the AI's verdict pools), `routing` (cast CurbRamp Disagrees by road context), `packet` (blinded hand-coding packet, key outside the repo), `provenance` / `verify`; `road_prior_reproduce.sh` runs them all. Results in [`docs/road-prior-2026-10.md`](../docs/road-prior-2026-10.md). |
| `vlm_baseline.py` | Issue #16: open-weight VLMs (MLX, on a Mac) as zero-shot validators on the round-3 held-out crops, against the five production DINOv2-L validators. Subcommands `score` (per-crop yes/no score from the next-token distribution), `baseline` (production models; run from a venv with `requirements.txt` only), `timing` (batch-1 latency on an idle machine), `greedy` (greedy text vs score argmax), `report` (subset and full-sample tables, bootstraps, within-city AUC, ROC figure, provenance), `explain` and `packet` (blinded two-pass explanation-rating packet, key outside the repo), `agreement`. Results in [`docs/vlm-baseline-2026-10.md`](../docs/vlm-baseline-2026-10.md). |

The maps themselves are fitted by `python -m calibration.calibrate_from_production` from `ai_assessments.csv`.
