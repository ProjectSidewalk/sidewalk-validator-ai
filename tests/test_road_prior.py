"""Tests for analysis/road_prior.py (issue #45). CPU only, no network, no optional dependencies."""
import doctest
import json

import numpy as np
import pandas as pd
import pytest

from analysis import road_prior as rp


def test_doctests():
    res = doctest.testmod(rp)
    assert res.failed == 0 and res.attempted > 0


def _streets():
    # a plus-shaped junction at (0, 0) with four arms, one arm split mid-way (a degree-2 node at lat 0.002)
    return pd.DataFrame({
        "street_edge_id": [1, 2, 3, 4, 5],
        "way_type": ["residential", "residential", "service", "primary", "residential"],
        "start_point": ["0,0", "0,0", "0,0", "0,0", "0,0.001"],
        "end_point": ["0,0.001", "0.001,0", "-0.001,0", "0,-0.001", "0,0.002"]})


def test_street_nodes_degrees():
    nodes = rp.street_nodes(_streets()).set_index(["lat", "lng"]).degree
    assert nodes[(0.0, 0.0)] == 4
    assert nodes[(0.001, 0.0)] == 2  # where edge 1 meets edge 5
    assert nodes[(0.002, 0.0)] == 1  # dead end
    s = rp.intersection_summary(rp.street_nodes(_streets()))
    assert s == {"nodes": 6, "deg_ge2": 2, "deg_ge3": 1, "dead_end_share": 4 / 6}


def test_street_features_distances_and_way_type():
    lab = pd.DataFrame({"latitude": [0.0009, 0.0], "longitude": [0.0, 0.0002], "street_edge_id": [1, 3]})
    out = rp.street_features(lab, _streets())
    # first label is 0.0001 deg (11.1 m) from the degree-2 node, 100 m from the junction
    assert out.d_intersection_m.round(1).tolist() == [11.1, 22.2]
    assert out.d_junction3_m.round(0).tolist() == [100.0, 22.0]
    assert out.own_way_type.tolist() == ["residential", "service"]
    assert out.own_way_is_service.tolist() == [0.0, 1.0]
    assert out.d_deadend_m.notna().all() and (out.d_deadend_m >= 0).all()


def test_nearest_distance_synthetic_points():
    nodes = np.array([[0.0, 0.0], [100.0, 0.0]])
    pts = np.array([[10.0, 0.0], [60.0, 0.0], [100.0, 30.0]])
    assert rp.nearest_distance(pts, nodes).tolist() == [10.0, 40.0, 30.0]
    assert np.isnan(rp.nearest_distance(pts, np.empty((0, 2)))).all()


def test_human_majority_excludes_ai_and_drops_ties():
    def v(*pairs):
        return json.dumps([{"user_id": str(i), "validation": val, "validator_type": t}
                           for i, (val, t) in enumerate(pairs)])
    # the AI's Disagree would break the tie if it were counted
    assert rp.human_votes(v(("Agree", "Human"), ("Disagree", "Human"), ("Disagree", "AI"))) == (1, 1, 0, None,
                                                                                                "Disagree")
    assert rp.human_votes(v(("Disagree", "Human"), ("Disagree", "Human"), ("Agree", "Human"), ("Unsure", "Human"))
                          )[3] == "Disagree"
    assert rp.human_votes(v(("Agree", "AI"))) == (0, 0, 0, None, "Agree")
    assert rp.human_votes(float("nan")) == (0, 0, 0, None, None)
    d = pd.DataFrame({"label_type": ["CurbRamp"] * 3, "h_n": [3, 4, 2], "h_maj": ["Agree", None, "Disagree"]})
    s = rp.analysis_set(d, "CurbRamp", min_votes=3)
    assert len(s) == 1 and s.y.tolist() == [0]


def test_driveway_counts_within_radius():
    ways = [np.array([[0.0, 15.0], [0.0, 100.0]]),     # nearest point 15 m away
            np.array([[40.0, -50.0], [40.0, 50.0]]),   # 40 m away
            np.array([[-200.0, 0.0], [-300.0, 0.0]])]  # 200 m away
    pts = np.array([[0.0, 0.0]])
    assert rp.count_ways_within(pts, ways, 20).tolist() == [1]
    assert rp.count_ways_within(pts, ways, 50).tolist() == [2]
    assert rp.count_ways_within(pts, [], 50).tolist() == [0]


def test_osm_features_ignore_non_driveway_service():
    lab = pd.DataFrame({"latitude": [0.0], "longitude": [0.0], "osm_way_id": [99]})
    cr = pd.DataFrame(columns=["osmid", "lat", "lng"])
    sv = pd.DataFrame({"osmid": [1, 2], "service": ["parking_aisle", "driveway"],
                       "coords": ["[[0.0, 0.0], [0.0001, 0.0]]", "[[0.0, 0.0003], [0.0001, 0.0003]]"]})
    out = rp.osm_features(lab, cr, sv)
    assert np.isnan(out.d_crossing_m[0])
    assert out.own_way_is_driveway[0] == 0
    assert out.n_driveways_20m[0] == 0 and out.n_driveways_50m[0] == 1  # 33 m away; the parking aisle is ignored


def test_public_name_filter(tmp_path):
    p = tmp_path / "cities.json"
    p.write_text(json.dumps({"cities": [
        {"city_id": "seattle-wa", "url": "https://s/", "visibility": "public", "country_id": "usa"},
        {"city_id": "secret-xx", "url": "https://x", "visibility": "private", "country_id": "usa"}]}))
    assert rp.public_cities(p) == {"seattle-wa": "https://s"}
    cities = rp.load_cities(p)
    df = pd.DataFrame({"city": ["seattle-wa", "secret-xx", "unknown"], "v": [1, 2, 3]})
    assert rp.public_only(df, cities).city.tolist() == ["seattle-wa"]
    assert rp.display_city("secret-xx", cities) == "other (private)"


def test_production_assessments_key_on_city_and_label_id():
    oof = pd.DataFrame({"schema": ["sidewalk_seattle", "sidewalk_amsterdam"], "label_id": [7, 7],
                        "label_type": ["CurbRamp", "CurbRamp"], "ai_result": ["Agree", "Disagree"],
                        "ai_accuracy": [0.9, 0.6], "oof_accuracy": [0.95, 0.5],
                        "new_accuracy": [0.95, 0.5]})
    cities = {"seattle-wa": "public", "amsterdam": "public"}
    out = rp.production_assessments(oof, cities)
    assert sorted(out.label_uid) == ["amsterdam:7", "seattle-wa:7"]


def test_stratification_arithmetic():
    df = pd.DataFrame({
        "label_type": ["CurbRamp"] * 6,
        "ai_result": ["Disagree", "Disagree", "Agree", "Agree", "Agree", "Agree"],
        "h_n": [3, 3, 3, 3, 3, 1],
        "h_maj": ["Disagree", "Agree", "Agree", "Disagree", "Agree", "Disagree"],
        "expert1": [None, None, None, None, "Disagree", None],
        "d_intersection_m": [50.0, 10.0, 10.0, 30.0, 20.0, 100.0],
        "d_crossing_m": [50.0, np.nan, 5.0, 30.0, 30.0, 100.0]})
    pools = rp.production_pools(df, 3, ["expert1"])
    t = rp.far_shares(df, pools).set_index("pool")
    assert t.loc["AI Disagree (all)", "n"] == 2
    assert t.loc["AI Disagree (all)", "far_int_25m"] == 0.5
    assert t.loc["AI Disagree (all)", "n_osm"] == 1
    assert t.loc["AI Agree (all)", "n"] == 4
    assert t.loc["AI Agree (all)", "far_int_15m"] == 0.75
    assert t.loc["AI Agree, human majority Disagree (contested)", "n"] == 1  # h_n 1 does not count
    assert t.loc["AI Agree, human majority Disagree (contested)", "far_both_25m"] == 1.0
    assert t.loc["AI Agree, expert Disagree", "n"] == 1


def test_packet_stratified_and_blinded():
    rng = np.random.default_rng(0)
    n = 400
    df = pd.DataFrame({"label_type": "CurbRamp", "city": rng.choice(["a-wa", "b-wa", "c-wa"], n),
                       "label_id": np.arange(n), "ai_result": rng.choice(["Agree", "Disagree"], n),
                       "h_n": 3, "h_maj": rng.choice(["Agree", "Disagree"], n),
                       "d_intersection_m": rng.uniform(0, 80, n), "d_crossing_m": np.nan,
                       "oof_accuracy": 0.9, "h_agree": 2, "h_dis": 1, "pano_url": "https://gsv"})
    df["label_uid"] = df.city + ":" + df.label_id.astype(str)
    urls = {"a-wa": "https://a", "b-wa": "https://b", "c-wa": "https://c"}
    packet, key = rp.build_packet(df, urls, seed=123, n_total=40)
    assert len(packet) == 40 and packet.item.tolist() == list(range(1, 41))
    assert set(packet.columns) == set(rp.PACKET_COLUMNS)  # no verdicts or distances on the rater's sheet
    assert key.distance_bucket.value_counts().tolist() == [10, 10, 10, 10]
    assert key.pool.value_counts().tolist() == [20, 20]  # AI Disagree and overturned Agree sampled as halves
    assert set(key.reason) <= {"ai_disagree", "agree_overturned_crowd", "agree_overturned_expert"}
    assert not {"label", "label_uid", "ps_label_page", "city"} & set(packet.columns)  # no city, no share page
    assert packet.gsv_link.eq("https://gsv").all()  # the link is the label's own pano, never a share page
    assert not packet.gsv_link.str.contains("/label/").any()
    first = key.set_index("item").loc[1]
    assert first.ps_label_page.startswith(urls[first.label_uid.split(":")[0]] + "/label/")
    again, _ = rp.build_packet(df, urls, seed=123, n_total=40)
    assert again.equals(packet)
    _, other_key = rp.build_packet(df, urls, seed=124, n_total=40)
    assert not other_key.label_uid.tolist() == key.label_uid.tolist()


def test_loco_predict_never_sees_the_held_out_city():
    # city c has the opposite relation and outnumbers a + b 5:1, so a fit that saw c would learn c's direction
    d = pd.DataFrame({"city": ["a"] * 20 + ["b"] * 20 + ["c"] * 200,
                      "d_x": np.r_[np.tile(np.r_[np.full(10, 2.0), np.full(10, 50.0)], 2),
                                   np.full(100, 2.0), np.full(100, 50.0)]})
    d["y"] = np.r_[np.tile(np.repeat([0, 1], 10), 2), np.repeat([1, 0], 100)]
    p = rp.loco_predict(d, ["d_x"])
    c = d.city == "c"
    assert p[c & (d.d_x == 50)].min() > p[c & (d.d_x == 2)].max()  # learned from a and b, wrong for c


def test_inside_git_checkout(tmp_path):
    assert rp.inside_git_checkout(__file__)
    assert not rp.inside_git_checkout(tmp_path / "nowhere" / "x.csv")


def test_overpass_tiles_cover_every_label():
    lat = np.array([47.60, 47.61, 47.95])
    lng = np.array([-122.33, -122.32, -122.05])
    tiles = rp.label_tiles(lat, lng)
    assert len(tiles) == 2
    for la, ln in zip(lat, lng):
        assert any(s <= la <= n and w <= ln <= e for s, w, n, e in (rp.tile_bbox(i, j) for i, j in tiles))


def test_overpass_post_falls_back_to_the_next_mirror():
    class R:
        def __init__(self, code, body):
            self.status_code, self._body, self.text = code, body, json.dumps(body)

        def json(self):
            return self._body

    class S:
        def __init__(self):
            self.calls = []

        def post(self, url, **kw):
            self.calls.append(url)
            return R(502, {}) if "bad" in url else R(200, {"elements": [{"type": "node", "id": 1}]})

    s = S()
    el, url = rp.overpass_post("q", ["https://bad/api", "https://good/api/interpreter"], retries=2, session=s,
                               base_sleep=0)
    assert url == "https://good/api/interpreter" and el == [{"type": "node", "id": 1}]
    assert s.calls == ["https://bad/api/interpreter"] * 2 + ["https://good/api/interpreter"]
    with pytest.raises(RuntimeError, match="HTTP 502"):
        rp.overpass_post("q", ["https://bad/api"], retries=1, session=S(), base_sleep=0)


def test_disagree_context_table_separates_context_within_confidence():
    d = pd.DataFrame({"label_type": "CurbRamp", "ai_result": "Disagree", "h_n": 3,
                      "h_maj": ["Disagree", "Agree"] * 6, "oof_accuracy": [0.1] * 4 + [0.4] * 4 + [0.7] * 4,
                      "d_intersection_m": [50.0, 5.0] * 6, "d_crossing_m": [50.0, 5.0] * 6})
    t = rp.disagree_context_table(d, n_conf_bins=3)
    per_bin = t[t.oof_accuracy_bin != "all"]
    assert set(per_bin.context) == {"far_both", "near_both"}
    assert (per_bin[per_bin.context == "far_both"].confirm_rate == 1.0).all()
    assert (per_bin[per_bin.context == "near_both"].confirm_rate == 0.0).all()
    assert t[t.oof_accuracy_bin == "all"].n.sum() == len(d)


def _cast(n=600, seed=0):
    """Synthetic cast CurbRamp Disagrees in three cities where near-both labels are confirmed less often."""
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({"label_type": "CurbRamp", "ai_vote": "Disagree", "ai_result": "Disagree",
                      "city": rng.choice(["a-wa", "b-wa", "c-wa"], n), "oof_accuracy": rng.uniform(0.2, 0.7, n),
                      "d_intersection_m": rng.choice([5.0, 50.0], n), "d_crossing_m": rng.choice([5.0, 50.0], n),
                      "h_n": 3, "d_deadend_m": 100.0, "n_driveways_50m": 0})
    near = (d.d_intersection_m < 25) & (d.d_crossing_m < 25)
    p = np.where(near, 0.2, 0.7)
    d["h_maj"] = np.where(rng.random(n) < p, "Disagree", "Agree")
    return d


def test_logit_fit_matches_sklearn_unpenalised():
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(1)
    X = rng.normal(size=(2000, 2))
    y = (rng.random(2000) < 1 / (1 + np.exp(-(X @ [1.0, -0.5])))).astype(float)
    ours = rp.logit_fit(np.column_stack([np.ones(2000), X]), y, ["c", "a", "b"])
    sk = LogisticRegression(C=1e9, max_iter=5000).fit(X, y)
    assert np.allclose(ours.coef[["a", "b"]].values, sk.coef_[0], atol=1e-3)


def test_city_fe_recovers_negative_near_both_effect():
    v = rp.cast_disagrees(_cast(3000))
    fe = rp.city_fe_models(v[v.validated], thresholds=(25,)).set_index(["spec", "term"])
    assert fe.loc[("context", "near_both"), "coef"] < -1.0 and fe.loc[("context", "near_both"), "p"] < 1e-6


def test_cast_disagrees_falls_back_to_new_accuracy():
    d = _cast(20).assign(new_accuracy=0.5)
    d.loc[:4, "oof_accuracy"] = np.nan
    c = rp.cast_disagrees(d)
    assert len(c) == 20 and (c.conf_accuracy.iloc[:5] == 0.5).all()


def test_cast_disagrees_keeps_only_cast_votes():
    d = _cast(50)
    d.loc[:9, "ai_vote"] = "Unsure"
    assert len(rp.cast_disagrees(d)) == 40


def test_routing_report_runs_and_names_only_public_cities():
    d = _cast(1200)
    d.loc[d.city == "c-wa", "city"] = "secret-city"
    cities = {"a-wa": {"visibility": "public"}, "b-wa": {"visibility": "public"},
              "secret-city": {"visibility": "private"}}
    md, tables = rp.routing_report(d, cities, n_boot=20, min_city_n=10)
    assert "secret-city" not in md
    assert set(tables["routing_by_city"].city) == {"a-wa", "b-wa"}
    assert {"quartile_context_minvotes1", "quartile_context_minvotes3", "city_fe", "coarse_lift"} <= set(tables)


def test_city_fe_drops_single_outcome_cities():
    v = rp.cast_disagrees(_cast(3000))
    v = v[v.validated].copy()
    extra = v.head(5).assign(city="tiny-wa", y=1)
    fe = rp.city_fe_models(pd.concat([v, extra]), thresholds=(25,))
    assert (fe.cities_dropped == 1).all() and (fe.cities == 3).all()


def test_road_context_and_dummies_agree_at_the_threshold():
    d = pd.DataFrame({"d_intersection_m": [24.999, 25.0, 25.0], "d_crossing_m": [24.999, 24.999, 25.0]})
    assert rp.road_context(d, 25).tolist() == ["near_both", "far_one", "far_both"]
    dm = rp.context_dummies(d, 25)
    assert dm.near_both.tolist() == [1.0, 0.0, 0.0] and dm.far_one.tolist() == [0.0, 1.0, 0.0]


def test_expert_votes_come_from_every_label_not_only_majority_ones(tmp_path):
    v = [{"user_id": "u-exp", "validation": "Disagree", "validator_type": "Human"},
         {"user_id": "ai", "validation": "Agree", "validator_type": "AI"}]
    pd.DataFrame({"label_id": [1, 2], "label_type": "CurbRamp", "latitude": 47.6, "longitude": -122.3,
                  "validations": [json.dumps(v), json.dumps(v[1:])]}).to_csv(tmp_path / "x-wa.csv", index=False)
    lab = rp.load_labels(tmp_path / "x-wa.csv", "x-wa", ["u-exp", "u-other"])
    # a lone expert Disagree, which ai_vs_humans.csv (>= 2 human votes, strict majority) never holds
    assert lab.expert1.iloc[0] == "Disagree" and lab.expert1.iloc[1:].isna().all() and lab.expert2.isna().all()
    assert lab.h_n.tolist() == [1, 0] and lab.ai_vote.tolist() == ["Agree", "Agree"]


def test_city_bootstrap_is_wider_than_label_bootstrap_when_gain_varies_by_city():
    rng = np.random.default_rng(0)
    g = np.repeat(list("abcdef"), 200)
    y = rng.integers(0, 2, len(g))
    gain = np.where(np.isin(g, ["a", "b", "c"]), 0.3, -0.25)  # helps half the cities, hurts the rest
    p1 = np.clip(0.5 + gain * (y - 0.5), 0.01, 0.99)
    (alo, ahi), _ = rp.paired_delta_ci(y, np.full(len(g), 0.5), p1, n_boot=200)
    (clo, chi), _ = rp.city_delta_ci(y, np.full(len(g), 0.5), p1, g, n_boot=200)
    assert chi - clo > ahi - alo


def test_paired_bootstrap_resamples_both_models_together():
    # identical predictions: a paired bootstrap gives exactly zero difference, an unpaired one would not
    rng = np.random.default_rng(1)
    y, p = rng.integers(0, 2, 300), rng.random(300)
    (alo, ahi), (llo, lhi) = rp.paired_delta_ci(y, p, p, n_boot=50)
    (clo, chi), _ = rp.city_delta_ci(y, p, p, np.repeat(list("abc"), 100), n_boot=50)
    assert alo == ahi == llo == lhi == clo == chi == 0
