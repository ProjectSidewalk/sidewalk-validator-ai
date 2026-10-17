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

The maps themselves are fitted by `python -m calibration.calibrate_from_production` from `ai_assessments.csv`.
