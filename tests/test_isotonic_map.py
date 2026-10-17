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
    iso = im.fit_isotonic(c, y)
    class_map = im.curve_to_class_map(iso)
    # On grid points the dict serves exactly the (rounded) curve.
    for t in im.GRID:
        assert im.api_lookup(class_map, t) == pytest.approx(round(float(im.curve_value(iso, t)[0]), 4), abs=1e-12)
    # Anywhere else it serves the curve at the grid point at or below c, which never exceeds the curve at c.
    rng = np.random.default_rng(1)
    for q in rng.uniform(0.5, 1.0, 500):
        floor = im.GRID[im.GRID <= q].max()
        served = im.api_lookup(class_map, q)
        assert served == pytest.approx(round(float(im.curve_value(iso, floor)[0]), 4), abs=1e-12)
        assert served <= float(im.curve_value(iso, q)[0]) + 5e-5
    # The vectorised lookup agrees with the copied loop, including below the first threshold.
    qs = np.r_[0.3, rng.uniform(0.5, 1.0, 200), 1.0]
    assert np.allclose(im.api_lookup_many(class_map, qs), [im.api_lookup(class_map, q) for q in qs])
    assert im.api_lookup(class_map, 0.3) == 0


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
    assert im.api_lookup(class_map, 0.6) == 0.0 and im.api_lookup(class_map, 0.85) == 1.0
    assert im.fit_range(iso) == {"n": 4, "min_confidence": 0.8, "max_confidence": 1.0}
    capped = im.cap_class_map(class_map)
    assert max(capped.values()) == im.HOLD_BACK_CAP < im.GATE
    im.validate_api_map({"correct": capped, "incorrect": class_map})


def test_predict_verdicts_tie_goes_to_correct():
    verdict, conf = im.predict_verdicts([0.5, 0.3, 0.9], [0.5, 0.7, 0.1])
    assert verdict.tolist() == ["correct", "incorrect", "correct"]
    assert conf.tolist() == [0.5, 0.7, 0.9]


def test_reliability_table_zero_weight_bin():
    tab = im.reliability_table([0.55, 0.95], [1, 0], weight=[0.0, 1.0])
    assert tab.n.tolist() == [1] and tab.observed.tolist() == [0.0]
