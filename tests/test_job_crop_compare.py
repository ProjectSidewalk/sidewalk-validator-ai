"""analysis/job_crop_compare.py (issue #53): counts, restrict and compare on synthetic data."""
import doctest
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from analysis import job_crop_compare as jcc


def test_doctests():
    r = doctest.testmod(jcc)
    assert r.attempted > 0 and r.failed == 0


def _split12():
    """12 rows over two providers: 1 not from the store, 1 out of frame, 1 dims mismatch, and 9 cut, one of them with
    no recorded dims (cut anyway, counted as unverified)."""
    rows = []
    for i in range(12):
        rows.append({"label_uid": f"c:{i}", "city": "c", "label_id": i, "pano_id": f"p{i}", "label_type": "CurbRamp",
                     "label_class": "incorrect" if i % 2 else "correct", "pano_source": "gsv" if i < 8 else "Mapillary",
                     "pano_path": f"/store/p{i}.jpg", "pano_native_w": 16384, "pano_native_h": 8192,
                     "split": ["train", "val", "test", "test_heldout"][i % 4]})
    split = pd.DataFrame(rows)
    split.loc[3, "pano_path"] = None
    raw = pd.DataFrame({"label_id": range(12), "label_type": "CurbRamp", "pano_x": 100, "pano_y": 5000,
                        "pano_width": 16384.0, "pano_height": 8192.0})
    raw.loc[5, "pano_y"] = 8192          # out of frame
    raw.loc[6, ["pano_width", "pano_height"]] = [13312, 6656]  # placed on an upscaled frame
    raw.loc[9, ["pano_width", "pano_height"]] = [np.nan, np.nan]
    raw = pd.concat([raw, raw.assign(label_type="Obstacle")])  # another type in the export, ignored
    return split, raw


def test_counts_exact(tmp_path):
    split, raw = _split12()
    split.to_csv(tmp_path / "split.csv", index=False)
    cache = tmp_path / "raw"
    cache.mkdir()
    raw.to_csv(cache / "c.csv", index=False)
    t = jcc.main(["counts", "--split", str(tmp_path / "split.csv"), "--rawlabels-cache", str(cache),
                  "--out-dir", str(tmp_path / "out")])
    allrows = t[t.label_class == "all"].set_index(["pano_source", "split"])
    assert allrows.n.sum() == 12
    assert allrows.n_ok.sum() == 9 and allrows.n_no_store.sum() == 1  # 12 - 3 skipped; row 9 cut unverified
    assert allrows.n_out_of_frame.sum() == 1 and allrows.n_dims_mismatch.sum() == 1
    assert allrows.n_unverified_dims.sum() == 1
    assert allrows.loc[("gsv", "test_heldout"), "n_no_store"] == 1  # row 3
    assert allrows.loc[("mapillary", "val"), "n_unverified_dims"] == 1  # row 9, provider lower-cased
    assert set(allrows.verdict[allrows.index.get_level_values(1).isin(["test", "test_heldout"])]) == {"too few"}
    per_class = t[t.label_class != "all"]
    assert per_class.n.sum() == 12
    assert (tmp_path / "out" / "counts.csv").exists() and "| provider |" in (tmp_path / "out" / "counts.md").read_text()


def test_counts_verdict_gate():
    n = 40
    d = pd.DataFrame({"pano_source": "gsv", "split": "test_heldout",
                      "label_class": ["incorrect"] * 10 + ["correct"] * (n - 10), "job_status": "ok",
                      "dims_recorded": True})
    assert jcc.counts_table(d).query("label_class == 'all'").verdict.item() == "AUC measurable"
    d.loc[0, "job_status"] = "dims_mismatch"  # 9 incorrect left
    assert jcc.counts_table(d).query("label_class == 'all'").verdict.item() == "too few"


def test_counts_fails_on_a_different_snapshot(tmp_path):
    split, raw = _split12()
    split.to_csv(tmp_path / "split.csv", index=False)
    cache = tmp_path / "raw"
    cache.mkdir()
    raw[raw.label_id != 2].to_csv(cache / "c.csv", index=False)
    with pytest.raises(ValueError, match="no rawLabels row"):
        jcc.main(["counts", "--split", str(tmp_path / "split.csv"), "--rawlabels-cache", str(cache),
                  "--out-dir", str(tmp_path / "out")])


def test_store_check_marks_missing_files():
    d = pd.DataFrame({"pano_path": ["/a", "/b"], "pano_native_w": [10, 10], "pano_native_h": [5, 5],
                      "pano_y": [1, 1], "pano_width": [10, 10], "pano_height": [5, 5]})
    assert jcc.job_crop_status(d, store_exists=lambda p: p == "/a").tolist() == ["ok", "no_store"]


def test_restrict_keeps_every_split_and_writes_the_hash(tmp_path):
    split = pd.DataFrame({"label_uid": [f"c:{i}" for i in range(8)], "label_class": ["correct", "incorrect"] * 4,
                          "split": ["train", "val", "test", "test_heldout"] * 2, "hard": 0,
                          "crop_path": "old.webp"})
    status = ["ok"] * 8
    status[2] = "dims_mismatch"
    manifest = pd.DataFrame({"label_uid": [f"c:{i}" for i in range(8)] + ["c:99"],
                             "label_class": ["correct", "incorrect"] * 4 + ["correct"],
                             "crop_path": [f"x/c_{i}.webp" for i in range(9)], "status": status + ["ok"]})
    split.to_csv(tmp_path / "split.csv", index=False)
    (tmp_path / "job").mkdir()
    manifest.to_csv(tmp_path / "job" / "manifest.csv", index=False)
    out, digest = jcc.main(["restrict", "--split", str(tmp_path / "split.csv"), "--manifest",
                            str(tmp_path / "job" / "manifest.csv"), "--out", str(tmp_path / "job" / "split.csv")])
    assert set(out.split) == {"train", "val", "test", "test_heldout"}
    assert "c:2" not in set(out.label_uid) and "c:99" not in set(out.label_uid) and len(out) == 7
    assert (out.crop_path.str.startswith("x/")).all()  # the job crops, not round 3's
    written = pd.read_csv(tmp_path / "job" / "split.csv")
    assert written.split.tolist() == out.split.tolist()
    text = (tmp_path / "job" / "restricted_uids.txt").read_text()
    assert hashlib.sha256(text.encode()).hexdigest() == digest
    assert (tmp_path / "job" / "restricted_uids.txt.sha256").read_text().startswith(digest)
    summary = json.loads((tmp_path / "job" / "restrict_summary.json").read_text())
    assert summary["dropped_by_reason"] == {"dims_mismatch": 1} and summary["job_rows_not_in_round3"] == 1
    assert summary["kept_by_split"]["test"] == 1 and summary["round3_by_split"]["test"] == 2
    with pytest.raises(SystemExit, match="must not overwrite"):
        jcc.main(["restrict", "--split", str(tmp_path / "split.csv"), "--manifest",
                  str(tmp_path / "job" / "manifest.csv"), "--out", str(tmp_path / "split.csv")])


def test_restrict_summary_names_labels_the_job_run_did_not_select():
    split = pd.DataFrame({"label_uid": ["c:1", "c:2"], "label_class": ["correct"] * 2, "split": ["train", "test"]})
    manifest = pd.DataFrame({"label_uid": ["c:1"], "label_class": ["correct"], "status": ["ok"]})
    out = jcc.restrict_split(split, manifest)
    assert jcc.restrict_summary(split, manifest, out)["dropped_by_reason"] == {"not_selected": 1}


def test_restrict_refuses_a_changed_class():
    split = pd.DataFrame({"label_uid": ["c:1"], "label_class": ["correct"], "split": ["train"]})
    manifest = pd.DataFrame({"label_uid": ["c:1"], "label_class": ["incorrect"], "status": ["ok"]})
    with pytest.raises(ValueError, match="changed class"):
        jcc.restrict_split(split, manifest)


def _synthetic(n_panos=150, seed=0, job_better=False):
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_panos):
        for k in range(2):
            i = 2 * p + k
            y = int(rng.random() < 0.4)
            rows.append({"label_uid": f"c{p % 5}:{i}", "city": f"c{p % 5}", "pano_id": f"p{p}",
                         "pano_source": "gsv" if p < 120 else "panoramax", "label_class": ["correct", "incorrect"][y],
                         "split": "test" if p % 2 else "test_heldout", "severity": 1 + p % 3,
                         "tags": '["a"]' if i % 3 else "[]", "y": y})
    split = pd.DataFrame(rows)
    base = split.y * 1.0 + rng.normal(0, 0.9, len(split))
    depth = 1 / (1 + np.exp(-base))
    if job_better:
        job = 1 / (1 + np.exp(-(split.y * 3.0 + rng.normal(0, 0.5, len(split)))))
    else:
        job = 1 / (1 + np.exp(-(split.y * 1.0 + rng.normal(0, 0.9, len(split)))))  # same model quality, own noise
    mk = lambda p: pd.DataFrame({"label_uid": split.label_uid, "split": split.split, "true": split.label_class,
                                 "p_correct": 1 - p, "p_incorrect": p})
    return split.drop(columns="y"), mk(job), mk(depth)


def test_compare_no_difference_has_an_interval_covering_zero():
    split, job, depth = _synthetic()
    head, prov, slices = jcc.compare_tables(jcc.paired_frame(job, depth, split), n_boot=200, seed=1)
    pooled = head.iloc[0]
    assert pooled.split.startswith("pooled") and pooled.n == len(split)
    assert abs(pooled.delta) < 0.06 and pooled.delta_lo <= 0 <= pooled.delta_hi
    assert set(head.split) == {"pooled (test + test_heldout)", "test", "test_heldout"}
    assert not np.isnan(pooled.within_city_auc_job) and pooled.within_city_n_cities == 5
    assert set(slices.by) == {"severity", "tag"} and set(slices.split) == {"test", "test_heldout"}


def test_compare_detects_a_better_job_model_and_gates_providers(tmp_path):
    split, job, depth = _synthetic(job_better=True)
    head, prov, _ = jcc.compare_tables(jcc.paired_frame(job, depth, split), n_boot=200, seed=1)
    assert (head.delta > 0).all() and (head.delta_lo > 0).all()
    # panoramax has 30 panos x 2 = 60 rows split over two splits, about 12 incorrect each: measurable only when the
    # gates pass; gsv always is. No provider row is ever blank.
    assert "gsv" in set(prov.pano_source) and prov.auc_job.notna().all()
    for r in prov.itertuples():
        assert r.n >= jcc.MIN_N and min(r.n_incorrect, r.n - r.n_incorrect) >= jcc.MIN_CLASS
    split.to_csv(tmp_path / "split.csv", index=False)
    job.to_csv(tmp_path / "job.csv", index=False)
    depth.to_csv(tmp_path / "depth.csv", index=False)
    jcc.main(["compare", "--job-pred", str(tmp_path / "job.csv"), "--depth-pred", str(tmp_path / "depth.csv"),
              "--split", str(tmp_path / "split.csv"), "--out-dir", str(tmp_path / "out"), "--n-boot", "50"])
    for f in ("headline.csv", "per_provider.csv", "slices.csv", "tables.md", "provenance.json"):
        assert (tmp_path / "out" / f).exists()


def test_compare_drops_a_provider_below_the_gate():
    split, job, depth = _synthetic(n_panos=130)  # panoramax: 10 panos, 20 rows, under MIN_N
    _, prov, _ = jcc.compare_tables(jcc.paired_frame(job, depth, split), n_boot=0)
    assert set(prov.pano_source) == {"gsv"}


def test_compare_refuses_missing_scores():
    split, job, depth = _synthetic()
    with pytest.raises(ValueError, match="depth predictions have no score"):
        jcc.paired_frame(job, depth.iloc[5:], split)
