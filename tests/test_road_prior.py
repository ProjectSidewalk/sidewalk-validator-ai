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
                        "ai_accuracy": [0.9, 0.6], "oof_accuracy": [0.95, 0.5]})
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
    assert set(key.reason) <= {"ai_disagree", "agree_overturned_crowd", "agree_overturned_expert"}
    assert packet.ps_label_page.iloc[0].startswith(urls[packet.label.iloc[0].split(":")[0]] + "/label/")
    again, _ = rp.build_packet(df, urls, seed=123, n_total=40)
    assert again.equals(packet)
    other, _ = rp.build_packet(df, urls, seed=124, n_total=40)
    assert not other.label.tolist() == packet.label.tolist()


def test_loco_predict_never_sees_the_held_out_city():
    # city c has the opposite relation; a model that saw c would score it well
    d = pd.DataFrame({"city": ["a"] * 40 + ["b"] * 40 + ["c"] * 40,
                      "d_x": np.tile(np.r_[np.full(20, 2.0), np.full(20, 50.0)], 3)})
    d["y"] = np.r_[np.repeat([0, 1], 20), np.repeat([0, 1], 20), np.repeat([1, 0], 20)]
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
