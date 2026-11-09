"""analysis/context_arms.py report (issue #14): the arms table against the baseline on the same rows, held-out
deployments lettered when private, same-seed pairing, slices with training counts, the clipping table."""
import doctest
import json
import shutil

import numpy as np
import pandas as pd
import pytest

from analysis import context_arms as ca
from test_confident_learning import make_split


def metrics(auc):
    pc = {"correct": {"f1-score": 0.8}, "incorrect": {"f1-score": 0.6}}
    rap = {c: {"0.92": {"recall": 0.4, "threshold": 0.9}, "0.95": {"recall": 0.2, "threshold": 0.95}}
           for c in ("correct", "incorrect")}
    return {"test": {"auc": auc, "per_class": pc, "recall_at_precision": rap},
            "test_heldout": {"auc": auc - 0.1, "per_class": pc, "recall_at_precision": rap}, "hard": {}}


def write_run(base, ev, name, shift, auc, rng):
    pred = pd.DataFrame({"label_uid": ev.label_uid, "split": ev.split, "true": ev.label_class})
    pred["p_incorrect"] = np.clip(np.where(ev.label_class == "incorrect", rng.uniform(0.3, 1, len(ev)) + shift,
                                           rng.uniform(0, 0.7, len(ev)) - shift), 0, 1)
    pred["p_correct"] = 1 - pred.p_incorrect
    suffix = "" if name == "baseline" else f"_{name}"
    pred.to_csv(base / f"predictions{suffix}.csv", index=False)
    json.dump(metrics(auc), open(base / f"metrics{suffix}.json", "w"))


@pytest.fixture
def data(tmp_path):
    path, df = make_split(tmp_path, n_panos=150)
    df = df.copy()
    df.loc[df.index % 3 == 0, "tags"] = '["narrow"]'
    df.to_csv(path, index=False)
    base = path.parent
    ev = df[df.split != "train"]
    rng = np.random.default_rng(3)
    for name, shift, auc in (("baseline", 0.0, 0.90), ("two_scale", 0.1, 0.93), ("baseline-s43", 0.0, 0.91),
                             ("two_scale-s43", 0.05, 0.92), ("cl-drop-s43", 0.2, 0.99)):
        write_run(base, ev, name, shift, auc, rng)
    ctx = base / "context"
    ctx.mkdir()
    pd.DataFrame({"label_uid": df.label_uid, "wide_path": "w", "view_path": "v",
                  "wide_half_px": np.where(np.arange(len(df)) % 2, 1024, 600),
                  "wide_clipped": np.where(np.arange(len(df)) % 4 == 1, 1, 0)}).to_csv(ctx / "index.csv", index=False)
    shutil.copytree(base, tmp_path / "Toy2")  # a second type, same files
    (tmp_path / "public.txt").write_text("alpha\nbeta\n")  # gamma, the held-out city, is private here
    return tmp_path, df


def run(tmp_path, *extra):
    out = tmp_path / "doc"
    assert ca.main(["report", "--data-dir", str(tmp_path), "--label-types", "Toy,Toy2,Missing", "--arms",
                    "two_scale,view_heatmap", "--public-cities", str(tmp_path / "public.txt"), "--n-boot", "20",
                    "--slice-min-n", "5", "--out", str(out), *extra]) == 0
    return out


def test_doctests():
    r = doctest.testmod(ca)
    assert r.attempted > 0 and r.failed == 0


def test_report_writes_every_file_and_pairs_the_baseline(data):
    tmp_path, df = data
    out = run(tmp_path)
    for f in ("arms.md", "arms.json", "seeds.md", "slices.md", "clipping.md", "provenance.json"):
        assert (out / f).exists(), f
    j = json.load(open(out / "arms.json"))
    rows = {r["variant"]: r for r in j["Toy"]["arms"]}
    assert set(rows) == {"baseline", "two_scale", "view_heatmap"} and "missing" in rows["view_heatmap"]
    t = rows["two_scale"]
    assert t["test_auc"] == 0.93 and t["macro_f1_test"] == pytest.approx(0.7)
    assert t["incorrect_recall_p95_heldout"] == 0.2 and t["correct_recall_p92_heldout"] == 0.4
    assert t["heldout_delta_lo"] <= t["heldout_delta"] <= t["heldout_delta_hi"] and t["heldout_delta"] > 0
    assert "test_delta" not in rows["baseline"]
    md = (out / "arms.md").read_text()
    assert "| view_heatmap | not yet run |" in md and "not yet run (no Missing/split.csv)" in md
    # held-out: gamma is private in this run, so it is lettered and never named
    assert [h["deployment"] for h in j["Toy"]["heldout"]] == ["private city A (outside the US)"]
    assert "gamma" not in md and "gamma" not in (out / "arms.json").read_text()


def test_seeds_pair_same_seed_and_ignore_other_studies(data):
    tmp_path, _ = data
    out = run(tmp_path)
    seeds = json.load(open(out / "arms.json"))["Toy"]["seeds"]
    keys = {(r["variant"], r["seed"]) for r in seeds["rows"]}
    assert keys == {("baseline", 42), ("baseline", 43), ("two_scale", 42), ("two_scale", 43)}  # no cl-drop
    r43 = next(r for r in seeds["rows"] if (r["variant"], r["seed"]) == ("two_scale", 43))
    assert r43["test_auc"] == 0.92 and "heldout_delta" in r43
    assert "| two_scale | 43 |" in (out / "seeds.md").read_text()


def test_slices_carry_training_counts_and_the_thin_flag(data):
    tmp_path, df = data
    out = run(tmp_path)
    sl = json.load(open(out / "arms.json"))["Toy"]["slices"]
    tag = {r["slice"]: r for r in sl if r["by"] == "tag" and r["split"] == "test"}
    train = df[df.split == "train"]
    assert tag["narrow"]["train_n"] == int((train.tags == '["narrow"]').sum())
    assert tag["narrow"]["needs_data"] == (tag["narrow"]["train_n"] < 100)
    md = (out / "slices.md").read_text()
    assert "training rows" in md and "two_scale Δ [95% CI]" in md and "⚠ thin" in md


def test_clipping_table_and_provenance(data):
    tmp_path, df = data
    out = run(tmp_path)
    clip = json.load(open(out / "arms.json"))["Toy"]["clipping"]
    assert clip["n"] == len(df) and clip["whole_view_share"] == pytest.approx(0.5, abs=0.01)
    assert clip["clipped_share"] == pytest.approx(0.25, abs=0.01)
    md = (out / "clipping.md").read_text()
    assert "| Toy |" in md and "| Missing | no context index yet |" in md
    prov = json.load(open(out / "provenance.json"))["files_sha256"]
    assert "Toy/metrics_two_scale.json" in prov and "Toy/context/index.csv" in prov and "Toy/split.csv" in prov
    assert all(len(v) == 64 for v in prov.values())
