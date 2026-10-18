"""The shared map builder: format, monotonicity, and agreement with the API's own lookup loop."""
import json

import numpy as np
import pytest

from calibration import _isotonic_map as im


def _noisy_curve(n=5000, seed=0):
    rng = np.random.default_rng(seed)
    c = rng.uniform(0.5, 1.0, n)
    y = (rng.uniform(size=n) < 0.3 + 0.65 * (c - 0.5) / 0.5).astype(int)
    return c, y


def test_map_is_monotone_and_in_api_format():
    c, y = _noisy_curve()
    api_map = im.build_api_map({"correct": im.fit_isotonic(c, y), "incorrect": None})
    im.validate_api_map(api_map)
    assert list(api_map) == ["correct", "incorrect"]
    keys = list(api_map["correct"])
    assert keys[0] == "0.5" and keys[-1] == "1.0" and "0.9999" in keys and "0.8475" in keys
    assert all(len(k.split(".")[1]) <= 6 for k in keys)
    assert set(api_map["incorrect"].values()) == {0.0}       # no data -> never passes the gate
    json.loads(json.dumps(api_map))                           # plain JSON types


def test_validate_rejects_bad_maps():
    good = {"correct": {"0.5": 0.1, "0.9": 0.2}, "incorrect": {"0.5": 0.3}}
    im.validate_api_map(good)
    for bad in [{"correct": {"0.9": 0.2, "0.5": 0.1}, "incorrect": {"0.5": 0.3}},   # descending keys
                {"correct": {"0.5": 0.3, "0.9": 0.2}, "incorrect": {"0.5": 0.3}},   # decreasing values
                {"correct": {"0.5": 0.12345}, "incorrect": {"0.5": 0.3}},           # > 4 decimals
                {"correct": {"0.5": 0.1}}]:                                          # missing class
        with pytest.raises(ValueError):
            im.validate_api_map(bad)


def test_api_lookup_round_trip():
    c, y = _noisy_curve()
    curve = im.fit_isotonic(c, y)
    class_map = im.curve_to_class_map(curve)
    def floor4(v):
        return np.floor(v * 1e4 + 1e-9) / 1e4
    # On grid points the dict serves exactly the guarded curve, rounded down.
    for t in im.GRID:
        assert im.api_lookup(class_map, t) == pytest.approx(floor4(float(im.curve_value(curve, t, True)[0])), abs=1e-12)
    # Anywhere else it serves the curve at the grid point at or below c, never above the curve at c.
    rng = np.random.default_rng(1)
    for q in rng.uniform(0.5, 1.0, 500):
        floor = im.GRID[im.GRID <= q].max()
        served = im.api_lookup(class_map, q)
        assert served == pytest.approx(floor4(float(im.curve_value(curve, floor, True)[0])), abs=1e-12)
        assert served <= float(im.curve_value(curve, q, True)[0])    # never above the curve, no tolerance
        assert served <= float(im.curve_value(curve, q)[0])
    # The vectorised lookup agrees with the copied loop, including below the first threshold.
    qs = np.r_[0.3, rng.uniform(0.5, 1.0, 200), 1.0]
    assert np.allclose(im.api_lookup_many(class_map, qs), [im.api_lookup(class_map, q) for q in qs])
    assert im.api_lookup(class_map, 0.3) == 0


def test_wilson_and_thin_block_guard():
    # hand-checked Wilson bound
    assert float(im.wilson_lower(0.95, 100)) == pytest.approx(0.9008, abs=1e-4)   # z=1.645, by hand
    assert float(im.wilson_lower(0.5, 0)) == 0.0
    # a thin block (40 < 300 labels, all correct) is served at its Wilson lower bound, not 1.0
    curve = im.fit_isotonic(np.full(40, 0.6), np.ones(40))
    assert im.curve_to_class_map(curve, conservative=False)["0.6"] == 1.0
    assert im.curve_to_class_map(curve)["0.6"] == pytest.approx(round(float(im.wilson_lower(1.0, 40)), 4))
    assert curve.thin().tolist() == [True]
    # a big block (>= 300) is served at its mean: no shift
    big = im.fit_isotonic(np.full(1000, 0.7), np.r_[np.ones(950), np.zeros(50)])
    assert im.curve_to_class_map(big)["0.7"] == 0.95 and not big.thin().any()
    # a thin, perfect block after a big good block keeps the running maximum (the big block's mean)
    c = np.r_[np.full(5000, 0.7), np.full(3, 0.99)]
    y = np.r_[np.ones(4700), np.zeros(300), np.ones(3)]
    m = im.curve_to_class_map(im.fit_isotonic(c, y))
    im.validate_api_map({"correct": m, "incorrect": m})
    assert m["0.7"] == 0.94 and m["0.99"] == 0.94


def test_fractional_outcomes():
    # vote shares (1-1 -> 0.5, 2-1 -> 0.667) are fitted as means
    curve = im.fit_isotonic([0.9, 0.9, 0.9], [0.5, 2 / 3, 1.0])
    assert float(curve.value(0.9)[0]) == pytest.approx((0.5 + 2 / 3 + 1) / 3)


def test_reliability_and_ece():
    conf = np.array([0.55, 0.55, 0.95, 0.95, 1.0])
    correct = np.array([1, 0, 1, 1, 1])
    tab = im.reliability_table(conf, correct, predicted=np.full(5, 0.75))
    assert tab.n.tolist() == [2, 2, 1]                        # 1.0 falls in the closed last bin
    assert tab.observed.tolist() == [0.5, 1.0, 1.0]
    assert im.expected_calibration_error(tab) == pytest.approx((2 * 0.25 + 2 * 0.25 + 0.25) / 5)
    g = im.gate_summary([0.5, 0.95, 0.93, 0.1], [0, 1, 0, 1])
    assert g == {"coverage": 0.5, "n_pass": 2, "precision": 0.5}


def test_unsupported_low_confidence_is_zero_and_range_recorded():
    iso = im.fit_isotonic([0.8, 0.9, 0.95, 1.0], [1, 1, 1, 1])
    class_map = im.curve_to_class_map(iso)
    assert im.api_lookup(class_map, 0.6) == 0.0
    assert im.curve_value(iso, 0.85)[0] == 1.0 and 0 < im.api_lookup(class_map, 0.85) < 1.0  # n=4: thin-block guard
    assert im.fit_range(iso) == {"n": 4, "min_confidence": 0.8, "max_confidence": 1.0}
    capped = im.cap_class_map(im.curve_to_class_map(iso, conservative=False))
    assert max(capped.values()) == im.HOLD_BACK_CAP < im.GATE
    im.validate_api_map({"correct": capped, "incorrect": class_map})


def test_predict_verdicts_tie_goes_to_correct():
    verdict, conf = im.predict_verdicts([0.5, 0.3, 0.9], [0.5, 0.7, 0.1])
    assert verdict.tolist() == ["correct", "incorrect", "correct"]
    assert conf.tolist() == [0.5, 0.7, 0.9]


def test_reliability_table_zero_weight_bin():
    tab = im.reliability_table([0.55, 0.95], [1, 0], weight=[0.0, 1.0])
    assert tab.n.tolist() == [1] and tab.observed.tolist() == [0.0]


def test_rounding_never_lifts_onto_the_gate():
    # a big block whose mean is 0.91996 must be served as 0.9199, not rounded up to 0.92
    n = 100_000
    y = np.r_[np.ones(91_996), np.zeros(n - 91_996)]
    m = im.curve_to_class_map(im.fit_isotonic(np.full(n, 0.9), y))
    assert m["0.9"] == 0.9199 and im.api_lookup(m, 0.95) < im.GATE


def test_decreasing_outcome_gives_a_flat_map():
    c = np.linspace(0.5, 1.0, 2000)
    y = (c < 0.75).astype(float)            # right at low confidence, wrong at high: isotonic must pool to flat
    m = im.curve_to_class_map(im.fit_isotonic(c, y))
    im.validate_api_map({"correct": m, "incorrect": m})
    vals = [v for k, v in m.items() if float(k) >= c.min()]
    assert len(set(vals)) == 1
    assert vals[0] == pytest.approx(y.mean(), abs=1e-3)   # the pooled mean, ~0.5


def test_thin_guard_uses_kish_effective_n():
    # 100 labels of weight 5: weight sum 500 >= 300, but Kish n = 100 < 300 -> thin
    curve = im.fit_isotonic(np.full(100, 0.8), np.ones(100), weight=np.full(100, 5.0))
    assert curve.block_n[0] == pytest.approx(100) and curve.thin().tolist() == [True]
    # exactly 300 labels is NOT thin (the guard is "fewer than 300")
    assert im.fit_isotonic(np.full(300, 0.8), np.ones(300)).thin().tolist() == [False]
    assert im.fit_isotonic(np.full(299, 0.8), np.ones(299)).thin().tolist() == [True]


def test_gate_summary_boundary():
    assert im.gate_summary([0.92, 0.9199], [1, 0]) == {"coverage": 0.5, "n_pass": 1, "precision": 1.0}


def test_grid_is_fine_near_one():
    keys = {im.threshold_key(t) for t in im.GRID}
    # every 1e-6 from 0.99998 to 1.0: 21 keys, the finest a 6-decimal key allows
    assert {im.threshold_key(i / 1_000_000) for i in range(999_980, 1_000_001)} <= keys
    assert "0.999987" in keys and "0.999996" in keys and len(im.GRID) == 231
    # a gating block that starts between coarse grid points is served from its own start, not the next coarse point
    c = np.r_[np.full(1000, 0.99990), np.full(1000, 0.9999866)]
    y = np.r_[np.full(1000, 0.8), np.full(1000, 0.95)]
    m = im.curve_to_class_map(im.fit_isotonic(c, y))
    assert im.api_lookup(m, 0.9999875) == 0.95   # with only 0.99998/0.99999 this would serve 0.8


def test_zero_below_fitted_range():
    curve = im.fit_isotonic(np.full(500, 0.9), np.ones(500))
    m = im.curve_to_class_map(curve)
    assert im.api_lookup(m, 0.8) == 0.0 and im.api_lookup(m, 0.95) == 1.0
    assert float(curve.value(0.8)[0]) == 0.0
