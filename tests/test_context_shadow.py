"""analysis/context_arms.py shadow, preflight and check-run (issue #14): the shadow data directory keeps every write
off the round-3 directory, preflight names every missing piece before a job takes a GPU, check-run is the smoke
stage's criterion. pandas only; runs on the CPU-only CI."""
import hashlib
import json
import os
import subprocess
import sys

import pandas as pd
import pytest

from analysis import context_arms as ca
from test_confident_learning import make_split

ROOT = os.path.join(os.path.dirname(__file__), "..")
OCT = ("metrics.json", "predictions.csv", "metrics_baseline-s43.json", "predictions_baseline-s43.csv")


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def make_src(tmp_path, side_columns=True):
    """A round-3-like data directory: Toy/{split.csv, manifest.csv, generation_run.json, crops, October files},
    deployment_priors.json beside it, rawlabels/ next to it."""
    src = tmp_path / "retrain"
    src.mkdir()
    path, df = make_split(src, n_panos=60)
    if side_columns:
        df = df.assign(image_capture_date="2021-05", depth_m="7.5", high_quality_user="True", clipped="0",
                       crop_half_px="238")
        df.loc[df.index % 4 == 0, "depth_m"] = ""
        df.to_csv(path, index=False, lineterminator="\n")
    base = path.parent
    df.assign(pano_path="x").to_csv(base / "manifest.csv", index=False, lineterminator="\n")
    (base / "generation_run.json").write_text('{"settings": {}}')
    ev = df[df.split != "train"]
    pd.DataFrame({"label_uid": ev.label_uid, "split": ev.split, "true": ev.label_class, "p_correct": 0.5,
                  "p_incorrect": 0.5}).to_csv(base / "predictions.csv", index=False)
    for n in OCT:
        if n.endswith(".json"):
            (base / n).write_text("{}")
        elif n != "predictions.csv":
            (base / n).write_bytes((base / "predictions.csv").read_bytes())
    (src / "deployment_priors.json").write_text('{"Toy": {"correct": 3, "incorrect": 1}}')
    raw = tmp_path / "rawlabels"
    raw.mkdir()
    for c in df.city.unique():
        (raw / f"{c}.csv").write_text("label_id\n")
    return src, df


def make_store(tmp_path, df, keep_panos):
    store = tmp_path / "store"
    store.mkdir()
    (store / ".pano-store").write_text("test store\n")
    for c, p in set(zip(df.city, df.pano_id)):
        if p in keep_panos:
            d = store / c / p[:2]
            d.mkdir(parents=True, exist_ok=True)
            (d / f"{p}.jpg").write_bytes(b"jpg")
    return store


def test_shadow_layout_hashes_and_symlinks(tmp_path):
    src, df = make_src(tmp_path)
    dest = tmp_path / "work" / "data"
    rec = ca.shadow(str(src), str(dest), ["Toy"], today="2026-11-19")
    d = dest / "Toy"
    for c in ("correct", "incorrect"):
        assert os.path.islink(d / c) and os.readlink(d / c) == str(src / "Toy" / c)
    for n in ca.SHADOW_COPY + OCT:
        assert (d / n).read_bytes() == (src / "Toy" / n).read_bytes(), n   # byte-identical: same split hash
        assert rec["types"]["Toy"]["sources"][n] == sha(src / "Toy" / n)
    assert json.loads((dest / "shadow_run.json").read_text()) == rec
    assert rec["rawlabels"] == str(tmp_path / "rawlabels") and rec["subset"] is False
    assert rec["deployment_priors_sha256"] == sha(src / "deployment_priors.json")
    assert (dest / "deployment_priors.json").exists() and {"git_commit", "git_dirty"} <= set(rec)
    # crops resolve through the symlink exactly as evaluate.load_split resolves a relative crop_path
    assert all(os.path.exists(os.path.join(d, p)) for p in df.crop_path)
    # a rerun with unchanged sources is fine and keeps the creation date
    assert ca.shadow(str(src), str(dest), ["Toy"], today="2026-11-20")["created"] == "2026-11-19"


def test_shadow_refuses_dest_inside_src_changed_sources_and_another_source(tmp_path):
    src, _ = make_src(tmp_path)
    with pytest.raises(SystemExit, match="inside"):
        ca.shadow(str(src), str(src / "shadow"), ["Toy"])
    dest = tmp_path / "data"
    ca.shadow(str(src), str(dest), ["Toy"])
    with open(src / "Toy" / "split.csv", "a") as f:
        f.write("\n")
    with pytest.raises(SystemExit, match="different source hashes"):
        ca.shadow(str(src), str(dest), ["Toy"])
    other = tmp_path / "other"
    other.mkdir()
    with pytest.raises(SystemExit, match="shadow of"):
        ca.shadow(str(other), str(dest), ["Toy"])


def test_only_staged_subset_keeps_rows_whose_panorama_is_staged(tmp_path):
    src, df = make_src(tmp_path)
    panos = sorted(df.pano_id.unique())
    store = make_store(tmp_path, df, set(panos[::2]))
    dest = tmp_path / "smoke"
    rec = ca.shadow(str(src), str(dest), ["Toy"], only_staged=str(store))
    kept = pd.read_csv(dest / "Toy" / "split.csv", dtype=str, keep_default_na=False)
    want = df[df.pano_id.isin(panos[::2])]
    assert set(kept.label_uid) == set(want.label_uid) and kept.pano_id.map(type).eq(str).all()
    man = pd.read_csv(dest / "Toy" / "manifest.csv", dtype=str, keep_default_na=False)
    assert set(man.label_uid) == set(kept.label_uid)
    e = rec["types"]["Toy"]
    assert rec["subset"] is True and e["rows_kept"] == len(want) and e["rows_source"] == len(df)
    assert sum(sum(v.values()) for v in e["kept"].values()) == len(want)
    assert not (dest / "Toy" / "metrics.json").exists()   # October's outputs cover rows the subset lacks
    # the kept rows are written exactly as read (strings unchanged)
    orig = pd.read_csv(src / "Toy" / "split.csv", dtype=str, keep_default_na=False)
    assert kept.equals(orig[orig.label_uid.isin(kept.label_uid)].reset_index(drop=True))


def test_only_staged_refuses_a_split_left_with_one_class(tmp_path):
    src, df = make_src(tmp_path)
    keep = set(df[(df.split != "test_heldout") | (df.label_class == "correct")].pano_id) - set(
        df[(df.split == "test_heldout") & (df.label_class == "incorrect")].pano_id)
    store = make_store(tmp_path, df, keep)
    with pytest.raises(SystemExit, match="test_heldout"):
        ca.shadow(str(src), str(tmp_path / "smoke"), ["Toy"], only_staged=str(store))


def test_shadow_and_preflight_never_import_torch():
    code = ("import sys; from analysis import context_arms as ca; import data_generation, fine_tune_classifier, "
            "context_model; print(sorted(m for m in ('torch', 'transformers', 'torchvision') if m in sys.modules))")
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "[]"


# ---------------------------------------------------------------------------------------------------- preflight

GOOD = {"pandas": "2.3.1", "transformers": "4.55.0"}


@pytest.fixture
def ready(tmp_path):
    src, df = make_src(tmp_path)
    dest = tmp_path / "data"
    ca.shadow(str(src), str(dest), ["Toy"])
    hf = tmp_path / "hf"
    snap = hf / "hub" / "models--facebook--dinov2-large" / "snapshots" / "abc"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}")
    return {"src": src, "df": df, "data": str(dest), "runs": str(tmp_path / "runs"), "hf": str(hf)}


def pf(r, **kw):
    lines = []
    args = {"stage": "pilot", "hf_home": r["hf"], "versions": GOOD, "transformers_pin": "4.55.0", "out": lines.append}
    args.update(kw)
    arms = args.pop("arms", ["none", "side_info", "side_info_nohq"])
    return ca.preflight(r["data"], r["runs"], ["Toy"], arms, **args), lines


def test_preflight_passes_a_complete_pilot_and_writes_for_real(ready):
    problems, lines = pf(ready, pair_october=False, keep=ready["runs"])
    assert problems == [], problems
    assert any("missing share" in ln and "depth_m 25" in ln for ln in lines)   # every 4th row has no depth
    assert os.path.isdir(ready["runs"]) and not os.listdir(ready["runs"])     # the probe cleaned up after itself


def test_preflight_names_each_missing_piece(ready, tmp_path):
    problems, _ = pf(ready, versions={"pandas": "3.1.1", "transformers": "4.57.0"}, hf_home=str(tmp_path / "nohf"))
    text = "\n".join(problems)
    assert "pandas 3.1.1" in text and "transformers 4.57.0" in text and "HF cache" in text
    # a split.csv without a column an arm reads (would raise KeyError after the job holds its GPU)
    split = pd.read_csv(os.path.join(ready["data"], "Toy", "split.csv"), dtype=str, keep_default_na=False)
    split.drop(columns="depth_m").to_csv(os.path.join(ready["data"], "Toy", "split.csv"), index=False)
    problems, _ = pf(ready)
    assert any("lacks ['depth_m']" in p for p in problems)
    # the source changed since the shadow was made
    with open(ready["src"] / "Toy" / "manifest.csv", "a") as f:
        f.write("\n")
    problems, _ = pf(ready)
    assert any("changed or vanished" in p for p in problems)


def test_preflight_october_files_and_crops(ready):
    problems, _ = pf(ready, pair_october=True)   # the fixture has October's seed 42 and 43 files, not 44
    assert any("metrics_baseline-s44.json" in p for p in problems)
    os.remove(os.path.join(ready["data"], "Toy", "correct"))   # break the crop symlink
    problems, _ = pf(ready)
    assert any("sampled crops do not resolve" in p for p in problems)


def test_preflight_without_a_shadow(tmp_path, ready):
    problems = ca.preflight(str(tmp_path / "nowhere"), ready["runs"], ["Toy"], ["none"], hf_home=ready["hf"],
                            versions=GOOD, transformers_pin="4.55.0", write_test=False, out=lambda s: None)
    assert any("shadow_run.json" in p for p in problems)


def test_preflight_backfill_stage_writes_stage_files_and_checks_rawlabels(ready, tmp_path):
    df = ready["df"]
    panos = sorted(df.pano_id.unique())
    store = make_store(tmp_path, df, set(panos[1:]))
    problems, lines = pf(ready, arms=["two_scale", "view_heatmap"], backfill=True, pano_store=str(store))
    listing = os.path.join(ready["data"], "Toy", "stage_files.txt")
    gone = df[df.pano_id == panos[0]].iloc[0]
    assert open(listing).read() == f"{gone.city}/{panos[0][:2]}/{panos[0]}.jpg\n"
    assert any("1 of" in p and "ai_labels_stage.slurm" in p for p in problems)
    assert any("the backfill writes" in ln for ln in lines)
    # every manifest city needs <city>.csv in the shadow's rawLabels snapshot
    os.remove(tmp_path / "rawlabels" / f"{df.city.iloc[0]}.csv")
    problems, _ = pf(ready, arms=["two_scale"], backfill=True, pano_store=str(store))
    assert any("has no <city>.csv for 1 manifest cities" in p for p in problems)


def test_preflight_writes_the_stage_list_for_a_store_not_staged_yet(ready, tmp_path):
    """O1: the restage hand-off. A store that does not exist yet (the new one) lacks every panorama: the list is
    written, not skipped; the printed command never points DEST at a marked store (the stage job removes its
    marker); stage-list makes one deduplicated list in a directory it creates."""
    df = ready["df"]
    new_store = tmp_path / "new-store"
    problems, _ = pf(ready, arms=["two_scale"], backfill=True, pano_store=str(new_store))
    listing = os.path.join(ready["data"], "Toy", "stage_files.txt")
    n_panos = len(set(zip(df.city, df.pano_id)))
    assert len(open(listing).read().splitlines()) == n_panos
    assert any("no .pano-store marker" in p for p in problems)
    text = "\n".join(problems)
    assert f"DEST={new_store} sbatch" in text and "stage-list" in text
    panos = sorted(df.pano_id.unique())
    marked = make_store(tmp_path, df, set(panos[1:]))
    problems, _ = pf(ready, arms=["two_scale"], backfill=True, pano_store=str(marked))
    text = "\n".join(problems)
    assert f"DEST={marked} " not in text and f"not {marked}" in text
    out = tmp_path / "work" / "stage"            # does not exist yet
    counts = ca.stage_list(ready["data"], ["Toy"], str(new_store), str(out), out=lambda s: None)
    assert counts["Toy"] == counts["_all"] == n_panos
    lines = (out / "stage_files.txt").read_text().splitlines()
    assert len(lines) == len(set(lines)) == n_panos and all(ln.endswith(".jpg") for ln in lines)
    assert ca.main(["stage-list", "--data", ready["data"], "--types", "Toy", "--pano-store", str(marked),
                    "--out", str(out)]) == 0
    assert len((out / "stage_files.txt").read_text().splitlines()) == 1


def test_preflight_refuses_a_stage_seed_list_the_rule_does_not_read(ready):
    rule = ca.DECISION_RULE
    assert pf(ready, confirmatory_seeds=rule["seeds_confirmatory"] * 2, baseline_seeds=rule["seeds_baseline"])[0] == []
    problems, _ = pf(ready, confirmatory_seeds=[42, 43, 44, 45, 46] * 2)
    assert rule["seeds_confirmatory"] == [42, 43, 44, 45, 46] or any("seeds_confirmatory" in p for p in problems)
    problems, _ = pf(ready, baseline_seeds=[42, 43, 44])
    assert any("seeds_baseline" in p and "[42, 43, 44]" in p for p in problems)


def test_preflight_skip_backfill_needs_a_covering_index(ready):
    problems, _ = pf(ready, arms=["view_heatmap"], skip_backfill=True)
    assert any("lacks" in p and "ok manifest rows" in p for p in problems)
    ctx = os.path.join(ready["data"], "Toy", "context")
    os.makedirs(ctx)
    pd.DataFrame({"label_uid": ready["df"].label_uid}).to_csv(os.path.join(ctx, "index.csv"), index=False)
    assert pf(ready, arms=["view_heatmap"], skip_backfill=True)[0] == []


def test_preflight_cli_exit_status(ready, monkeypatch):
    monkeypatch.setattr(ca, "installed_versions", lambda: GOOD)
    monkeypatch.setattr(ca, "environment_pin", lambda name, path=None: "4.55.0")
    argv = ["preflight", "--data", ready["data"], "--runs", ready["runs"], "--types", "Toy", "--arms",
            "none,side_info_nohq", "--hf-home", ready["hf"], "--no-write-test"]
    assert ca.main(argv) == 0
    assert ca.main(argv + ["--pair-october"]) == 1


def test_environment_pin_reads_environment_yml():
    assert ca.environment_pin("transformers") == "4.55.0" and ca.environment_pin("pandas") == "2.3.1"


# ---------------------------------------------------------------------------------------------------- check-run

def finish_run(data, tag, seed, rows=None, drop=0, key=None):
    d = os.path.join(data, "Toy")
    split = pd.read_csv(os.path.join(d, "split.csv"), dtype=str, keep_default_na=False)
    ev = split[split.split.isin(ca.EVAL_SPLITS)] if rows is None else rows
    ev = ev.iloc[drop:]
    pd.DataFrame({"label_uid": ev.label_uid, "split": ev.split, "true": ev.label_class, "p_correct": 0.4,
                  "p_incorrect": 0.6}).to_csv(os.path.join(d, f"predictions_{tag}.csv"), index=False)
    for n in (f"metrics_{tag}.json", f"report_{tag}.txt"):
        open(os.path.join(d, n), "w").write("{}")
    rec = os.path.join(d, "runs", tag)
    os.makedirs(rec, exist_ok=True)
    settings = {k: v for k, v in (ca.expected_settings(ca.variant_of_tag(tag)) or {}).items() if v is not None}
    json.dump({"seed": seed, "split_sha256": sha(os.path.join(d, "split.csv")), "epochs": 25, **settings, **(key or {})},
              open(os.path.join(rec, "run_key.json"), "w"))
    json.dump({"seed": seed}, open(os.path.join(rec, "training_run.json"), "w"))


def test_check_run_is_the_smoke_criterion(ready):
    data = ready["data"]
    finish_run(data, "side_info_nohq", 42)
    assert ca.check_run(data, "Toy", "side_info_nohq", seed=42) == []
    rec = ca.run_record(data, "Toy", "side_info_nohq")
    assert rec["status"] == "ok" and rec["seed"] == 42 and set(rec["files_sha256"]) == {"run_key.json", "training_run.json"}
    finish_run(data, "baseline-rerun-s43", 42, drop=3)
    p = ca.check_run(data, "Toy", "baseline-rerun-s43", seed=ca.seed_of_tag("baseline-rerun-s43"))
    assert any("3 missing" in x for x in p) and any("seed 42, expected 43" in x for x in p)
    split = pd.read_csv(os.path.join(data, "Toy", "split.csv"), dtype=str, keep_default_na=False)
    finish_run(data, "with-train", 42, rows=split)   # train rows scored too: extra rows
    assert any("extra" in x for x in ca.check_run(data, "Toy", "with-train"))
    assert ca.check_run(data, "Toy", "never-ran") == ["Toy/never-ran: no predictions_never-ran.csv"]


def test_check_run_verifies_split_hash_arm_resolution_and_epochs(ready):
    data = ready["data"]
    finish_run(data, "view_heatmap-r448", 42)
    assert ca.check_run(data, "Toy", "view_heatmap-r448", seed=42, epochs=25) == []
    # the documented single-job form with RES=448 but TAG=view_heatmap: the 224 arm's tag on a 448 run
    finish_run(data, "view_heatmap", 42, key={"context_view_res": 448})
    assert any("context_view_res 448, view_heatmap needs 224" in p for p in ca.check_run(data, "Toy", "view_heatmap"))
    finish_run(data, "baseline-rerun-s43", 43, key={"context_arm": "side_info", "context_view_res": 224})
    assert any("context_arm side_info, baseline-rerun needs None" in p
               for p in ca.check_run(data, "Toy", "baseline-rerun-s43", seed=43))
    finish_run(data, "two_scale", 42, key={"epochs": 1})
    assert ca.check_run(data, "Toy", "two_scale") == []          # the smoke stage: epochs not checked
    assert any("epochs 1, expected 25" in p for p in ca.check_run(data, "Toy", "two_scale", epochs=25))
    finish_run(data, "side_info", 42, key={"split_sha256": "0" * 64})
    assert any("split hash differs" in p for p in ca.check_run(data, "Toy", "side_info"))


def test_run_record_reads_the_flat_layout_too(ready):
    d = os.path.join(ready["data"], "Toy")
    json.dump({"seed": 44, "split_sha256": "ab"}, open(os.path.join(d, "training_run_dinov3-s44.json"), "w"))
    rec = ca.run_record(ready["data"], "Toy", "dinov3-s44")
    assert rec["status"] == "ok" and rec["seed"] == 44 and rec["path"] == "Toy/training_run_dinov3-s44.json"
    assert ca.run_record(ready["data"], "Toy", "nothing")["status"] == "missing"
