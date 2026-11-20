"""analysis/round4.py: the gold-free mirror, the variants, the gold manifests and scoring, and collect (issue #74).
Synthetic fixtures only; no network, no GPU."""
import json
import os

import numpy as np
import pandas as pd
import pytest

from analysis import confident_learning as cl
from analysis import round4 as r4

M_PER_DEG = 111195.0
NO_ROOTS = ["--forbidden-root", "/nonexistent-root-for-tests"]


def row(uid, pano, cls="correct", split="train", agree=3, disagree=0, source="margin", severity="2", tags="[]"):
    city, lid = uid.rsplit(":", 1)
    return {"label_uid": uid, "city": city, "label_id": lid, "pano_id": pano, "label_type": "Obstacle",
            "label_class": cls, "class_source": source, "human_agree": str(agree), "human_disagree": str(disagree),
            "severity": severity, "tags": tags, "crop_path": f"{cls}/{city}_{lid}.webp", "status": "ok",
            "split": split}


def base_split():
    rows = [
        row("a:1", "P1"),                       # gold item, train
        row("a:2", "P1"),                       # same scene as a gold item
        row("a:3", "P2"),                       # 2 m from a:1 on another panorama -> merged
        row("a:4", "P3"),                       # 10 m away -> kept
        row("b:1", "P1"),                       # same label_id / pano string, other city -> kept
        row("a:5", "P4", "incorrect", agree=0, disagree=3),   # same panorama as a gold item NOT in the split
        row("a:6", "P5", "incorrect", agree=0, disagree=2),
        row("a:7", "P6", "incorrect", agree=0, disagree=4),
        row("a:8", "P9", split="val"),          # gold item in val: its own scene has no train rows
        row("a:9", "P1", split="test"),         # eval row on a gold panorama: never touched
        row("a:10", "P7", split="test_heldout"),
    ]
    return pd.DataFrame(rows)


def coords():
    lat0, lon0 = 47.0, -122.0
    pts = {"a:1": 0, "a:2": 0, "a:3": 2, "a:4": 10, "a:5": 200, "a:6": 400, "a:7": 600, "a:8": 800, "a:9": 0,
           "a:10": 1000, "a:11": 200, "a:12": 5000}
    out = {}
    for uid, m in pts.items():
        city, lid = uid.split(":")
        out.setdefault(city, []).append({"label_id": lid, "latitude": lat0 + m / M_PER_DEG, "longitude": lon0})
    out.setdefault("b", []).append({"label_id": "1", "latitude": lat0 + 3000 / M_PER_DEG, "longitude": lon0})
    return out


def gold_uids(extra=()):
    rows = [{"label_uid": "a:1", "label_type": "Obstacle", "pano_id": "P1", "crop_path_r3": "correct/a_1.webp"},
            {"label_uid": "a:8", "label_type": "Obstacle", "pano_id": "P9", "crop_path_r3": ""},
            {"label_uid": "a:11", "label_type": "Obstacle", "pano_id": "P4", "crop_path_r3": ""},   # not in split
            {"label_uid": "a:12", "label_type": "Obstacle", "pano_id": "", "crop_path_r3": ""},     # matches nothing
            *extra]
    return pd.DataFrame(rows, columns=r4.GOLD_UID_COLUMNS)


@pytest.fixture
def layout(tmp_path):
    data, raw = tmp_path / "data", tmp_path / "raw"
    (data / "Obstacle" / "correct").mkdir(parents=True)
    (data / "Obstacle" / "incorrect").mkdir()
    (data / "Obstacle" / "correct" / "a_1.webp").write_bytes(b"x")
    raw.mkdir()
    for city, rows in coords().items():
        pd.DataFrame(rows).to_csv(raw / f"{city}.csv", index=False)
    base = base_split()
    cl.write_split(base, str(data / "Obstacle" / "split.csv"))
    (data / "Obstacle" / "generation_run.json").write_text("{}")
    (data / "deployment_priors.json").write_text("{}")
    uids = tmp_path / "gold_uids.csv"
    gold_uids().to_csv(uids, index=False)
    return {"tmp": tmp_path, "data": data, "raw": raw, "uids": uids, "r4": tmp_path / "scratch" / "r4", "base": base}


def run_prep(lay, *extra):
    return r4.main(["prep", "--data", str(lay["data"]), "--r4", str(lay["r4"]), "--gold-uids", str(lay["uids"]),
                    "--types", "Obstacle", *NO_ROOTS, *extra])


def test_gold_uid_file_has_exactly_four_columns_and_no_verdict(tmp_path):
    g = pd.DataFrame({"label_uid": ["a:1"], "label_type": ["Obstacle"], "pano_id": ["p"], "crop_path_r3": ["c.webp"],
                      "crop_status_r3": ["ok"], "expert": ["incorrect"], "stratum": ["lone"], "round": ["2"]})
    src = tmp_path / "gold.csv"
    g.to_csv(src, index=False)
    out = tmp_path / "out" / "gold_uids.csv"
    assert r4.main(["gold-uids", "--gold", str(src), "--out", str(out)]) == 0
    assert out.read_text().splitlines()[0] == "label_uid,label_type,pano_id,crop_path_r3"
    assert oct(os.stat(out).st_mode & 0o777) == "0o600"


def test_prep_drops_gold_items_and_their_scenes(layout):
    assert run_prep(layout, "--rawlabels", str(layout["raw"])) == 0
    out = cl.read_split(str(layout["r4"] / "Obstacle" / "split.csv")).set_index("label_uid")["split"]
    assert out["a:1"] == cl.TRAIN_DROPPED                      # gold_item
    assert out["a:2"] == cl.TRAIN_DROPPED                      # same panorama
    assert out["a:3"] == cl.TRAIN_DROPPED                      # 2 m away (3 m merge)
    assert out["a:4"] == "train"                               # 10 m away
    assert out["b:1"] == "train"                               # same label_id and pano string, other city
    assert out["a:5"] == cl.TRAIN_DROPPED                      # shares a panorama with a gold item not in the split
    assert out["a:6"] == "train" and out["a:7"] == "train"
    prep = json.loads((layout["r4"] / "Obstacle" / "r4_prep.json").read_text())
    assert prep["dropped"] == {"gold_item": 1, "gold_scene": 3}
    assert prep["dropped_by_reason_and_class"] == {"gold_item/correct": 1, "gold_scene/correct": 2,
                                                   "gold_scene/incorrect": 1}
    assert prep["gold_items_not_in_split"] == 2 and prep["gold_items_without_panorama"] == 1
    assert prep["gold_scene_panorama_only"] == 2
    assert "3 m merge" in prep["grouping"]


def test_prep_leaves_eval_rows_byte_identical_and_links_crops(layout):
    run_prep(layout, "--rawlabels", str(layout["raw"]))
    base, out = layout["base"], cl.read_split(str(layout["r4"] / "Obstacle" / "split.csv"))
    ev = ~base["split"].isin(["train"])
    a = base[ev].sort_values("label_uid").reset_index(drop=True)
    b = out[out["label_uid"].isin(a["label_uid"])].sort_values("label_uid").reset_index(drop=True)
    assert a.equals(b[a.columns])
    for name in ("correct", "incorrect"):
        link = layout["r4"] / "Obstacle" / name
        assert link.is_symlink() and os.path.realpath(link) == os.path.realpath(layout["data"] / "Obstacle" / name)
    assert (layout["r4"] / "deployment_priors.json").exists()
    assert (layout["r4"] / "Obstacle" / "generation_run.json").exists()


def test_prep_without_rawlabels_groups_by_panorama_only(layout):
    run_prep(layout)
    out = cl.read_split(str(layout["r4"] / "Obstacle" / "split.csv")).set_index("label_uid")["split"]
    assert out["a:3"] == "train"                               # no 3 m merge
    prep = json.loads((layout["r4"] / "Obstacle" / "r4_prep.json").read_text())
    assert prep["grouping"].startswith("panorama only")


def test_prep_round3_reference(layout):
    (layout["data"] / "Obstacle" / "metrics.json").write_text("{}")
    (layout["data"] / "Obstacle" / "predictions.csv").write_text("label_uid\n")
    run_prep(layout, "--round3-ref")
    t = layout["r4"] / "Obstacle"
    assert (t / "metrics_round3.json").exists() and (t / "predictions_round3.csv").exists()
    r3 = cl.read_split(str(t / "cl" / "round3" / "split.csv"))
    assert (r3["split"] == "train").sum() == (layout["base"]["split"] == "train").sum()


@pytest.mark.parametrize("where", ["tmp", "makelab", "data"])
def test_prep_refuses_bad_locations(layout, where):
    data = str(layout["data"])
    if where == "tmp":
        with pytest.raises(SystemExit, match="under /tmp"):
            r4.refuse_location("/tmp/x/r4", data)               # the default roots
    elif where == "makelab":
        with pytest.raises(SystemExit, match="under /gscratch/makelab"):
            r4.refuse_location("/gscratch/makelab/jfroehli/r4", data)
    else:
        with pytest.raises(SystemExit, match="is DATA"):
            r4.main(["prep", "--data", data, "--r4", data, "--gold-uids", str(layout["uids"]), "--types", "Obstacle",
                     *NO_ROOTS])


def test_tmp_refusal_through_the_parameter(layout):
    root = str(layout["tmp"] / "scratch")
    with pytest.raises(SystemExit, match="under"):
        r4.main(["prep", "--data", str(layout["data"]), "--r4", str(layout["r4"]), "--gold-uids",
                 str(layout["uids"]), "--types", "Obstacle", "--forbidden-root", root])


def test_pseudo_row_without_rawlabels_export_falls_back_to_panorama(layout):
    extra = {"label_uid": "z:5", "label_type": "Obstacle", "pano_id": "P6", "crop_path_r3": ""}
    gold_uids([extra]).to_csv(layout["uids"], index=False)
    run_prep(layout, "--rawlabels", str(layout["raw"]))       # no z.csv export: not a crash
    prep = json.loads((layout["r4"] / "Obstacle" / "r4_prep.json").read_text())
    assert prep["gold_pseudo_rows_without_rawlabels_export"] == 1
    out = cl.read_split(str(layout["r4"] / "Obstacle" / "split.csv")).set_index("label_uid")["split"]
    assert out["a:7"] == "train"                               # P6 in city z is another scene than P6 in city a


def test_train_row_without_rawlabels_export_raises(layout):
    base = layout["base"]
    base = pd.concat([base, pd.DataFrame([row("q:1", "PQ")])], ignore_index=True)
    cl.write_split(base, str(layout["data"] / "Obstacle" / "split.csv"))
    with pytest.raises(ValueError, match="no rawLabels export"):
        run_prep(layout, "--rawlabels", str(layout["raw"]))


def test_scene_ceiling_stops_a_transitive_chain(layout):
    # 40 train rows 2 m apart in a chain from the gold item: the 3 m merge would drop all of them.
    chain = [row(f"c:{i}", f"C{i}") for i in range(40)]
    base = pd.concat([pd.DataFrame(chain), pd.DataFrame([row("c:100", "C0", split="val")])], ignore_index=True)
    pd.DataFrame([{"label_id": str(i), "latitude": 47 + 2 * i / M_PER_DEG, "longitude": -122.0} for i in range(40)]
                 + [{"label_id": "100", "latitude": 47.0, "longitude": -122.0}]).to_csv(layout["raw"] / "c.csv",
                                                                                       index=False)
    gold = pd.DataFrame([{"label_uid": "c:100", "label_type": "Obstacle", "pano_id": "C0", "crop_path_r3": ""}])
    with pytest.raises(ValueError, match="ceiling") as e:
        r4.gold_free(base, gold, str(layout["raw"]), 3.0)
    assert "'rows': 40" in str(e.value)
    v, stats = r4.gold_free(base, gold, str(layout["raw"]), 3.0, ceiling=False)
    assert stats["dropped"]["gold_scene"] == 40 and stats["gold_scene_panorama_only"] == 1


def test_cl_drop_on_the_gold_free_base(layout):
    run_prep(layout, "--rawlabels", str(layout["raw"]))
    base = cl.read_split(str(layout["r4"] / "Obstacle" / "split.csv"))
    flagged = {"a:1", "a:6"}                                   # a:1 is a gold row already dropped
    v = r4.cl_drop(base, flagged)
    assert v.set_index("label_uid").loc["a:6", "split"] == cl.TRAIN_DROPPED
    assert (v["split"] == cl.TRAIN_DROPPED).sum() == (base["split"] == cl.TRAIN_DROPPED).sum() + 1
    assert v.equals(cl.cl_variants(base, flagged)["cl-drop"])
    with pytest.raises(ValueError):
        r4.check_flags_match({"source_sha256": "x"}, "y")


def rb_base(n_cor=60, n_inc=40):
    rows = [row(f"r:{i}", f"P{i}") for i in range(n_cor)]
    rows += [row(f"r:{n_cor + i}", f"Q{i}", "incorrect", agree=0, disagree=2 + (i % 3)) for i in range(n_inc)]
    rows += [row("r:999", "E1", "incorrect", agree=1, disagree=0, source="expert")]
    rows += [row("r:1000", "V1", split="test")]
    return pd.DataFrame(rows)


def test_rebalanced_arms_match_in_size_and_differ_by_seed():
    base = rb_base()
    ratio = r4.base_ratio(base)
    sym = r4.symmetric_rb(base, 42)
    c = r4.train_counts(sym)
    assert abs(c["correct"] - ratio * c["incorrect"]) <= 1
    assert r4.symmetric_rb(base, 42).equals(sym)              # deterministic
    mat = r4.matched_rb(base, sym, 42)
    assert r4.train_counts(mat) == c
    assert mat.set_index("label_uid").loc["r:999", "split"] == "train"   # expert-sourced row spared
    mat43 = r4.matched_rb(base, r4.symmetric_rb(base, 43), 43)
    dropped = lambda d: set(d.loc[d["split"] == cl.TRAIN_DROPPED, "label_uid"])  # noqa: E731
    assert dropped(mat43) != dropped(mat)
    assert dropped(r4.symmetric_rb(base, 43)) != dropped(sym)


def test_matched_rb_ignores_the_vote_margin():
    base = rb_base()
    sym = r4.symmetric_rb(base, 42)
    shuffled = base.copy()
    inc = (shuffled["label_class"] == "incorrect") & (shuffled["class_source"] != "expert")
    vals = shuffled.loc[inc, "human_disagree"].to_numpy()
    shuffled.loc[inc, "human_disagree"] = np.roll(vals, 7)    # same multiset: symmetric drops as many rows
    sym2 = r4.symmetric_rb(shuffled, 42)
    assert r4.train_counts(sym2) == r4.train_counts(sym)
    a, b = r4.matched_rb(base, sym, 42), r4.matched_rb(shuffled, sym2, 42)
    assert a["split"].equals(b["split"])


def test_variant_names_follow_the_seed_convention():
    assert r4.variant_name("matched-rb", 42) == "matched-rb"
    assert r4.variant_name("matched-rb", 43) == "matched-rb-s43"
    assert cl.SEED_RUN_RE.match("metrics_matched-rb-s43.json")["variant"] == "matched-rb"


def test_slice_train_counts_per_variant(tmp_path):
    base = pd.DataFrame([row(f"s:{i}", f"P{i}", severity=str(1 + i % 3)) for i in range(30)])
    d = tmp_path / "Obstacle"
    cl.write_split(base, str(d / "split.csv"))
    v = base.copy()
    v.loc[v["severity"] == "3", "split"] = cl.TRAIN_DROPPED
    cl.write_split(v, str(d / "cl" / "cl-drop" / "split.csv"))
    cl.write_split(base, str(d / "cl" / "round3" / "split.csv"))
    res = r4.slice_train(str(tmp_path), ["Obstacle"], ["round3", "baseline", "cl-drop"])
    sev = res["Obstacle"]["severity"]
    assert sev["3"]["cl-drop"]["train_n"] == 0 and sev["3"]["baseline"]["train_n"] == 10
    assert sev["3"]["round3"]["train_n"] == 10 and sev["3"]["baseline"]["needs_data"]
    assert "| 3 |" in r4.slice_train_markdown(res)


def test_gold_manifest_modes(tmp_path):
    split_root, crops = tmp_path / "data", tmp_path / "crops"
    (split_root / "Obstacle" / "correct").mkdir(parents=True)
    (split_root / "Obstacle" / "correct" / "a_1.webp").write_bytes(b"x")
    (crops / "crops").mkdir(parents=True)
    (crops / "crops" / "a_11.jpg").write_bytes(b"x")
    pd.DataFrame({"label_uid": ["a:11", "a:1"], "crop_path": ["crops/a_11.jpg", "crops/missing.jpg"],
                  "status": ["ok", "ok"]}).to_csv(crops / "manifest.csv", index=False)
    uids = gold_uids()
    m, side = r4.gold_manifest(uids, "Obstacle", "r3crops", split_root=str(split_root))
    assert m["label_uid"].tolist() == ["a:1"] and side["n_no_crop"] == 3
    assert list(m.columns) == r4.MANIFEST_COLUMNS and set(m["split"]) == {"gold"}
    u, side = r4.gold_manifest(uids, "Obstacle", "uniform", crops_root=str(crops))
    assert u["label_uid"].tolist() == ["a:11"] and side["n_no_crop"] == 3   # a:1's round-3 crop never used
    assert os.path.isabs(u["crop_path"].iloc[0])


def make_ckpt(path, weights=b"w1"):
    os.makedirs(path, exist_ok=True)
    for f in r4.WEIGHT_FILES:
        with open(os.path.join(path, f), "wb") as fh:
            fh.write(weights if f == "model.safetensors" else b"{}")
    return str(path)


def test_score_gold_skip_rerun_and_empty(tmp_path):
    calls = []

    def infer(man, ckpt, splits):
        calls.append(ckpt)
        return pd.DataFrame({"label_uid": man["label_uid"], "split": "gold", "true": "", "p_correct": 0.3,
                             "p_incorrect": 0.7})

    man = tmp_path / "Obstacle.csv"
    pd.DataFrame({"label_uid": ["a:1"], "label_class": [""], "split": ["gold"], "crop_path": ["/x.webp"]}).to_csv(
        man, index=False)
    ck = make_ckpt(tmp_path / "best")
    out = str(tmp_path / "pred.csv")
    assert r4.score_gold(str(man), ck, "t", out, infer=infer) == "scored"
    assert r4.score_gold(str(man), ck, "t", out, infer=infer) == "skipped" and len(calls) == 1
    make_ckpt(tmp_path / "best", b"w2")
    assert r4.score_gold(str(man), ck, "t", out, infer=infer) == "scored" and len(calls) == 2
    prov = json.loads(open(r4.provenance_path(out)).read())
    assert prov["model_safetensors_sha256"] == r4.sha256_file(os.path.join(ck, "model.safetensors"))
    empty = tmp_path / "Empty.csv"
    pd.DataFrame(columns=r4.MANIFEST_COLUMNS).to_csv(empty, index=False)
    assert r4.score_gold(str(empty), ck, "t", str(tmp_path / "e.csv"), infer=infer) == "empty"


def test_round3_checkpoint_resolves_next_to_trainer_state(tmp_path):
    d = tmp_path / "dinov2"
    for step in (10, 20):
        c = make_ckpt(d / f"checkpoint-{step}")
        with open(os.path.join(c, "trainer_state.json"), "w") as f:
            json.dump({"best_model_checkpoint": "retrain-2026-10/Obstacle/dinov2/checkpoint-10"}, f)
    assert r4.resolve_round3_checkpoint(str(d)) == str(d / "checkpoint-10")
    os.remove(d / "checkpoint-10" / "preprocessor_config.json")
    with pytest.raises(FileNotFoundError, match="preprocessor_config.json"):
        r4.resolve_round3_checkpoint(str(d))


def collect_layout(tmp_path, tags=("baseline", "baseline-s43")):
    r, runs = tmp_path / "r4", tmp_path / "runs"
    t = r / "Obstacle"
    t.mkdir(parents=True)
    for tag in tags:
        (t / f"metrics_{tag}.json").write_text(json.dumps({"tag": tag}))
        (t / f"predictions_{tag}.csv").write_text(f"label_uid\n{tag}\n")
        make_ckpt(runs / "Obstacle" / tag / "best")
        (runs / "Obstacle" / tag / "training_run.json").write_text("{}")
    return r, runs


def collect(r, runs, keep, *extra, expect=("baseline:42,43",)):
    return r4.main(["collect", "--r4", str(r), "--runs", str(runs), "--keep", str(keep), "--types", "Obstacle",
                    "--expect", *expect, *extra])


def test_collect_names_missing_runs(tmp_path, capsys):
    r, runs = collect_layout(tmp_path, ("baseline",))
    assert collect(r, runs, tmp_path / "keep") != 0
    assert "Obstacle/baseline-s43" in capsys.readouterr().err
    man = json.loads((r / "round4_manifest.json").read_text())
    assert man["missing"] == ["Obstacle/baseline-s43"]


def test_collect_replaces_a_stale_untagged_baseline(tmp_path):
    r, runs = collect_layout(tmp_path)
    (r / "Obstacle" / "metrics.json").write_text('{"stale": true}')
    assert collect(r, runs, tmp_path / "keep") == 0
    assert json.loads((r / "Obstacle" / "metrics.json").read_text()) == {"tag": "baseline"}
    man = json.loads((r / "round4_manifest.json").read_text())
    rec = man["untagged_baseline_copies"]["Obstacle"]["metrics.json"]
    assert rec["replaced_different"] and rec["previous_sha256"] != rec["source_sha256"]
    assert (tmp_path / "keep" / "data" / "Obstacle" / "metrics_baseline.json").exists()
    assert (tmp_path / "keep" / "runs" / "Obstacle" / "baseline" / "training_run.json").exists()
    assert not list((tmp_path / "keep").rglob("model.safetensors"))      # no weights to makelab


def test_collect_refuses_past_the_keep_budget(tmp_path, capsys):
    r, runs = collect_layout(tmp_path)
    assert collect(r, runs, tmp_path / "keep", "--max-files", "2") != 0
    assert "refusing to copy" in capsys.readouterr().err
    assert not (tmp_path / "keep").exists()


def test_collect_keeps_gold_files_private(tmp_path):
    r, runs = collect_layout(tmp_path)
    (r / "Obstacle" / "gold_predictions_r3crops_baseline.csv").write_text("label_uid\n")
    uids = tmp_path / "gold_uids.csv"
    uids.write_text("label_uid\n")
    assert collect(r, runs, tmp_path / "keep", "--gold-uids", str(uids)) == 0
    for p in (tmp_path / "keep").rglob("gold*"):
        assert os.stat(p).st_mode & 0o077 == 0, p


def test_parse_expect_and_storage():
    assert r4.parse_expect(["baseline:42"], ["A", "B"]) == [("A", "baseline", 42, "baseline"),
                                                           ("B", "baseline", 42, "baseline")]
    text = "│ Total:  │ 1000GB / 1024GB  │ 999000 / 1000000 files │"
    assert r4.parse_hyakstorage(text) == (1000.0, 1024.0, 999000, 1000000)


def test_verify_smoke_stops_at_the_first_failure(tmp_path):
    res = r4.verify_smoke(str(tmp_path))
    assert len(res) == 1 and not res[0][1] and "r4_prep.json" in res[0][0]


def test_adjudicator_crops_all_flag():
    from analysis import adjudicator as adj
    a = adj.build_parser().parse_args(["crops", "--gold", "g.csv", "--rawlabels", "r", "--out-dir", "o", "--all"])
    assert a.all is True
    assert adj.build_parser().parse_args(["crops", "--gold", "g", "--rawlabels", "r", "--out-dir", "o"]).all is False


# ---------------------------------------------------------------------------------------------------------------------
# Review round 1: B2 (collect records gold predictions) and the surviving mutants of S8
# ---------------------------------------------------------------------------------------------------------------------

def test_collect_records_missing_gold_predictions(tmp_path, capsys):
    r, runs = collect_layout(tmp_path)
    (r / "gold" / "r3crops").mkdir(parents=True)
    (r / "gold" / "r3crops" / "Obstacle.csv").write_text("label_uid\n")
    (r / "Obstacle" / "gold_predictions_r3crops_baseline.csv").write_text("label_uid,p_correct,p_incorrect\n")
    assert collect(r, runs, tmp_path / "keep") == 4                      # every run present, gold scoring not
    err = capsys.readouterr().err
    assert "Obstacle/baseline-s43/r3crops" in err and "Obstacle/round3/r3crops" in err
    man = json.loads((r / "round4_manifest.json").read_text())
    by_tag = {x["tag"]: x for x in man["runs"]}
    assert by_tag["baseline"]["gold_predictions_sha256"]["r3crops"] is not None
    assert by_tag["baseline-s43"]["gold_predictions_sha256"] == {"r3crops": None}
    assert man["missing"] == [] and "Obstacle/round3/r3crops" in man["missing_gold_predictions"]
    for tag in ("baseline-s43", "round3"):
        (r / "Obstacle" / f"gold_predictions_r3crops_{tag}.csv").write_text("label_uid\n")
    assert collect(r, runs, tmp_path / "keep") == 0


def test_score_job_time_covers_the_measured_rate():
    with open(os.path.join(os.path.dirname(r4.__file__), "round4_score.slurm")) as f:
        text = f.read()
    hours = int(text.split("#SBATCH --time=", 1)[1].split(":", 1)[0])
    assert hours * 60 >= 27 * 5.2         # 27 pilot checkpoints at the mean measured 5.2 min each (job 43405831)


def test_matched_rb_spares_expert_rows_for_every_seed():
    rows = [row(f"r:{i}", f"P{i}") for i in range(30)]
    rows += [row(f"r:{100 + i}", f"Q{i}", "incorrect", agree=0, disagree=2) for i in range(10)]   # symmetric drops
    rows += [row(f"r:{200 + i}", f"R{i}", "incorrect", agree=0, disagree=5) for i in range(10)]   # symmetric keeps
    rows += [row("r:999", "E1", "incorrect", agree=1, disagree=0, source="expert")]
    base = pd.DataFrame(rows)
    for seed in range(42, 52):
        mat = r4.matched_rb(base, r4.symmetric_rb(base, seed), seed)
        assert mat.set_index("label_uid").loc["r:999", "split"] == "train", seed


def test_rebalance_keeps_the_lowest_ranks():
    d = pd.DataFrame({"label_uid": [f"a:{i}" for i in range(20)], "split": "train",
                      "label_class": ["correct"] * 16 + ["incorrect"] * 4})
    out = r4.rebalance(d, 1.5, 42)                                    # keep 6 of 16 correct rows
    cor = d[d["label_class"] == "correct"]
    ranks = r4._rank(42, cor["label_uid"]).sort_values(kind="mergesort")
    kept = set(out.loc[(out["split"] == "train") & (out["label_class"] == "correct"), "label_uid"])
    assert kept == set(cor.loc[ranks.index[:6], "label_uid"])


def test_uniform_manifest_never_falls_back_to_the_round3_crop(tmp_path, monkeypatch):
    # The round-3 crop of a:1 exists relative to the working directory, so any fallback to it would be found.
    r3 = tmp_path / "data" / "Obstacle"
    (r3 / "correct").mkdir(parents=True)
    (r3 / "correct" / "a_1.webp").write_bytes(b"x")
    monkeypatch.chdir(r3)
    crops = tmp_path / "crops"
    crops.mkdir()
    pd.DataFrame({"label_uid": ["a:11"], "crop_path": ["missing.jpg"], "status": ["ok"]}).to_csv(
        crops / "manifest.csv", index=False)
    u, side = r4.gold_manifest(gold_uids(), "Obstacle", "uniform", split_root=str(tmp_path / "data"),
                               crops_root=str(crops))
    assert len(u) == 0 and side["n_no_crop"] == 4


def test_score_gold_reruns_when_the_manifest_changes(tmp_path):
    calls = []

    def infer(man, ckpt, splits):
        calls.append(len(man))
        return pd.DataFrame({"label_uid": man["label_uid"], "split": "gold", "true": "", "p_correct": 0.3,
                             "p_incorrect": 0.7})

    man = tmp_path / "Obstacle.csv"
    frame = pd.DataFrame({"label_uid": ["a:1"], "label_class": [""], "split": ["gold"], "crop_path": ["/x.webp"]})
    frame.to_csv(man, index=False)
    ck = make_ckpt(tmp_path / "best")
    out = str(tmp_path / "pred.csv")
    assert r4.score_gold(str(man), ck, "t", out, infer=infer) == "scored"
    pd.concat([frame, frame.assign(label_uid="a:2")]).to_csv(man, index=False)    # same weights, new manifest
    assert r4.score_gold(str(man), ck, "t", out, infer=infer) == "scored" and calls == [1, 2]
