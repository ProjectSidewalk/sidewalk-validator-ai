"""calibration/per_city.py (issue #15): per-city aggregation, Wilson intervals, private-city masking, min-n."""
import json

import numpy as np
import pandas as pd
import pytest

from calibration import per_city as pc

CITIES = {"seattle-wa": "public", "st-louis-mo": "public", "springfield-il": "public",
          "springfield-ma": "private", "hidden": "private"}


def _frame():
    """Toy labels_oof rows. seattle: 4 gated Agree (3 right), 2 gated Disagree (1 right), 1 Agree below the gate,
    1 unlabelled gated Agree; st_louis: 2 gated Agree (1 tie); hidden and springfield: 1 gated Agree each
    (private); validation: dropped."""
    rows = [
        ("sidewalk_seattle", "CurbRamp", "Agree", 0.95, True, 1.0),
        ("sidewalk_seattle", "CurbRamp", "Agree", 0.95, True, 1.0),
        ("sidewalk_seattle", "CurbRamp", "Agree", 0.97, True, 1.0),
        ("sidewalk_seattle", "CurbRamp", "Agree", 0.97, True, 0.0),
        ("sidewalk_seattle", "CurbRamp", "Disagree", 0.93, True, 1.0),
        ("sidewalk_seattle", "CurbRamp", "Disagree", 0.93, True, 0.0),
        ("sidewalk_seattle", "CurbRamp", "Agree", 0.91, True, 0.0),     # below the gate
        ("sidewalk_seattle", "CurbRamp", "Agree", 0.99, False, np.nan),  # not labelled
        ("sidewalk_st_louis", "CurbRamp", "Agree", 0.96, True, 1.0),
        ("sidewalk_st_louis", "CurbRamp", "Agree", 0.96, True, 0.5),     # a tie counts half
        ("sidewalk_hidden", "CurbRamp", "Agree", 0.94, True, 1.0),
        ("sidewalk_springfield", "CurbRamp", "Agree", 0.94, True, 0.0),
        ("sidewalk_validation", "CurbRamp", "Agree", 0.99, True, 0.0),
    ]
    df = pd.DataFrame(rows, columns=["schema", "label_type", "ai_result", "ai_accuracy", "labelled", "correct"])
    df["label_id"] = range(len(df))
    return df


def _load(tmp_path, df=None):
    path = tmp_path / "labels_oof.csv"
    (df if df is not None else _frame()).to_csv(path, index=False)
    return pc.load_labels(path)


@pytest.mark.parametrize("p, n, lo, hi", [
    (0.8, 10, 0.4902, 0.9433),   # textbook Wilson 95% for 8/10
    (0.0, 10, 0.0, 0.2775),      # 0/10
    (1.0, 10, 0.7225, 1.0),      # 10/10
    (0.5, 100, 0.4038, 0.5962),  # 50/100
])
def test_wilson_interval_known_values(p, n, lo, hi):
    got_lo, got_hi = pc.wilson_interval(p, n)
    assert float(got_lo) == pytest.approx(lo, abs=1e-4)
    assert float(got_hi) == pytest.approx(hi, abs=1e-4)


def test_wilson_interval_empty():
    lo, hi = pc.wilson_interval(0.0, 0)
    assert (float(lo), float(hi)) == (0.0, 0.0)


def test_private_and_ambiguous_cities_are_pooled(tmp_path):
    df = pc.assign_display_city(_load(tmp_path), CITIES)
    names = dict(zip(df.schema, df.city))
    assert names["sidewalk_seattle"] == "seattle-wa"
    assert names["sidewalk_st_louis"] == "st-louis-mo"
    # "hidden" is private; "springfield" is a prefix of springfield-il and springfield-ma, so it is never named
    # after the public one
    assert names["sidewalk_hidden"] == names["sidewalk_springfield"] == "other (2 private deployments)"
    assert "sidewalk_validation" not in names


@pytest.mark.parametrize("schema_city, cities, expected", [
    ("seattle", {"seattle-wa": "public"}, "seattle-wa"),            # region code
    ("rancagua", {"rancagua-chile": "public"}, "rancagua-chile"),   # country suffix
    ("newport_ky", {"newport-ky": "public"}, "newport-ky"),         # exact
    # unlisted schema whose name merely starts a listed id: never named after it
    ("oak", {"oak-park-gardens": "public"}, None),
    ("oak", {"oak-park": "public"}, None),
    ("springfield", {"springfield-il": "public", "springfield-ma": "private"}, None),   # ambiguous
    ("nowhere", {"seattle-wa": "public"}, None),
])
def test_match_city_id(schema_city, cities, expected):
    assert pc.match_city_id(schema_city, cities) == expected


def test_unlisted_schema_is_pooled_not_named():
    df = pd.DataFrame({"schema": ["sidewalk_oak", "sidewalk_seattle"]})
    out = pc.assign_display_city(df, {"oak-park": "public", "seattle-wa": "public"})
    assert list(out.city) == ["other (1 private deployment)", "seattle-wa"]


def test_pooled_label_counts_only_deployments_in_the_mask():
    df = pd.DataFrame({"schema": ["sidewalk_hidden", "sidewalk_quiet", "sidewalk_seattle"]})
    cities = {"hidden": "private", "quiet": "private", "seattle-wa": "public"}
    assert set(pc.assign_display_city(df, cities).city) == {"other (2 private deployments)", "seattle-wa"}
    masked = pc.assign_display_city(df, cities, count_mask=[True, False, True])
    assert set(masked.city) == {"other (1 private deployment)", "seattle-wa"}


def test_without_cities_json_nothing_is_named(tmp_path):
    df = pc.assign_display_city(_load(tmp_path), {})
    assert set(df.city) == {"other (4 private deployments)"}


def test_per_city_strata_values(tmp_path):
    df = pc.assign_display_city(_load(tmp_path), CITIES)
    s = pc.per_city_strata(df).set_index(["city", "verdict"])
    a = s.loc[("seattle-wa", "Agree")]
    assert a.n == 4                                   # the 0.91 row and the unlabelled row are excluded
    assert a.observed == pytest.approx(0.75)
    assert a.served == pytest.approx(0.96)
    assert a.gap == pytest.approx(0.21)
    lo, hi = pc.wilson_interval(0.75, 4)
    assert (a.ci_lo, a.ci_hi) == (pytest.approx(float(lo)), pytest.approx(float(hi)))
    assert s.loc[("seattle-wa", "Disagree")].observed == pytest.approx(0.5)
    assert s.loc[("st-louis-mo", "Agree")].observed == pytest.approx(0.75)
    other = s.loc[("other (2 private deployments)", "Agree")]
    assert (other.n, other.observed) == (2, pytest.approx(0.5))
    assert s.loc[(pc.ALL_CITIES, "Agree")].n == 8
    # the gap is per verdict, never pooled: Disagree served 0.93, observed 0.5
    assert s.loc[("seattle-wa", "Disagree")].gap == pytest.approx(0.43)


def test_min_n_filters_rows_and_below_gate(tmp_path):
    df = pc.assign_display_city(_load(tmp_path), CITIES)
    strata = pc.per_city_strata(df)
    md = pc.markdown_tables(strata, min_n=3)
    curb = md.split("### CurbRamp")[1].split("###")[0]
    assert "| seattle-wa | 4 | 0.960 | 0.750" in curb
    assert "st-louis-mo" not in curb and "other (" not in curb   # 2 gated votes < 3
    assert "2 more cities have fewer than 3 gated votes" in curb
    assert "| seattle-wa | 4 | 0.960 | 0.750 (0.301–0.954) | +0.210 | 2 | — | — | — |" in curb
    assert "calibration error" not in md

    b = pc.below_gate(strata[strata.city != pc.ALL_CITIES], min_n=2)
    assert list(b.city) == ["other (2 private deployments)", "seattle-wa", "st-louis-mo"]
    assert b.shortfall.iloc[0] == pytest.approx(0.42)
    assert list(pc.below_gate(strata[strata.city != pc.ALL_CITIES], min_n=3).city) == ["seattle-wa"]


def test_city_gate_uses_the_wilson_lower_bound():
    d = pd.DataFrame({"city": "a", "label_type": "T", "ai_result": "Agree", "labelled": True,
                      "ai_accuracy": [0.93] * 20 + [0.97] * 80, "correct": [0] * 20 + [1] * 80})
    # 80/80 at 0.97: lower bound 0.954 >= 0.92
    assert pc.city_gate(d, "a", "T") == (0.97, 0.8)
    assert pc.city_gate(d, "a", "T", min_n=81) == (None, 0.0)   # 0.97 keeps only 80 votes
    # 30/30 has a point estimate of 1.0 but a lower bound of 0.886 < 0.92: not enough
    small = d.iloc[50:].iloc[:30].assign(ai_accuracy=0.97)
    assert pc.city_gate(small, "a", "T", min_n=10) == (None, 0.0)


def test_below_gate_markdown_pooled_row_and_private_counts(tmp_path):
    df = pc.assign_display_city(_load(tmp_path), CITIES)
    strata = pc.per_city_strata(df)
    md = pc.markdown_below_gate(strata, min_n=2, df=df)
    assert "| other (2 private deployments) | CurbRamp | 2 |" in md
    assert md.split("other (2 private deployments) | CurbRamp")[1].split("\n")[0].endswith("n/a (pooled) |")
    assert "Agree strata with n >= 2: 2 named public city x type strata and 1 pooled private rows" in md
    # one deployment at a time, the private strata have 1 vote each (< 2): nothing counted, and nothing named
    assert "0 of 0 deployment x type Agree strata" in md
    assert "hidden" not in md and "springfield" not in md
    pb = pc.private_below_gate(df, min_n=1)
    assert pb == {"strata": 2, "below": 1, "ci_below": 1, "votes_below": 1, "votes_total": 8}   # 0/1: upper 0.79


def test_main_end_to_end(tmp_path):
    labels = tmp_path / "labels_oof.csv"
    _frame().to_csv(labels, index=False)
    cities = tmp_path / "cities.json"
    cities.write_text(json.dumps({"status": "OK", "cities": [{"city_id": k, "visibility": v}
                                                             for k, v in CITIES.items()]}))
    out = tmp_path / "out"
    pc.main(["--labels", str(labels), "--cities-json", str(cities), "--min-n", "1", "--out-dir", str(out)])
    md = (out / "per_city.md").read_text()
    for lt in pc.LABEL_TYPES:
        assert f"### {lt}" in md
        assert (out / f"per_city_{lt.lower()}.png").exists()
    assert "hidden" not in md and "sidewalk_" not in md
    csv = pd.read_csv(out / "per_city.csv")
    assert not csv.city.str.contains("hidden|sidewalk_").any()
    assert set(csv.verdict) == {"Agree", "Disagree"}
