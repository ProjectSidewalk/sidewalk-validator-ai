"""Shared pieces for turning (confidence, correct?) pairs into the accuracy map sidewalk-ai-api serves.

The serving format, read by https://github.com/ProjectSidewalk/sidewalk-ai-api/blob/main/main.py, is one JSON file per
label type::

    {"correct":   {"0.5": 0.9412, "0.5025": 0.9412, ..., "1.0": 0.9931},
     "incorrect": {"0.5": 0.3120, ...}}

The outer key is the model's predicted class (``correct`` = the AI says Agree, ``incorrect`` = Disagree). For a
prediction with softmax confidence ``c`` for that class, the API walks the inner dict **in insertion order** and keeps
the value of the last threshold ``<= c``. So thresholds must be written in ascending order, and the values should be
non-decreasing, or a dip in the map becomes a dip in the served accuracy. SidewalkWebpage records an AI vote only when
the served value is ``>= 0.92``.

This module fits a monotone non-decreasing precision-vs-confidence curve with isotonic regression and samples it on a
fixed grid (``GRID``). Each map value is the lower end of a 90% Wilson interval for the isotonic block it falls in, so
**the served accuracy is a lower confidence bound on precision**, not a point estimate. Because the API returns the
value at the grid point at or below ``c``, and the curve is non-decreasing, the served value is never above the
fitted curve at ``c``.

Below the lowest confidence seen in the fit data the curve is 0.0, not a flat extrapolation, so a grid point with no
support can never clear the gate. Above the highest it is held flat (confidences top out at 1.0).

Example::

    verdict, confidence = predict_verdicts(p_correct, p_incorrect)
    iso = fit_isotonic(confidence, correct, weight)
    class_map = curve_to_class_map(iso)          # {"0.5": ..., ..., "1.0": ...}
    api_lookup(class_map, 0.973)                 # what the API would serve
"""
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

GATE = 0.92
"""SidewalkWebpage's ``ai-validation-min-accuracy``: an AI verdict becomes a vote only at or above this value."""

HOLD_BACK_CAP = 0.9199
"""Value a held-back class's map is capped at: just under the gate, so that verdict never becomes a vote."""

API_CLASSES = ("correct", "incorrect")

# 0.5 .. 1.0 in steps of 0.0025, plus extra points near 1.0 where most confidences sit: in production (Oct 2026) half
# of all verdicts have confidence above 0.99905 and 29% above 0.9999, spread down to the 6th decimal. The API serves
# the value at the grid point at or below c, so without these points everything in (0.9999, 1.0) would get the
# curve's value at 0.9999. Built from integers so the string keys are exact ("0.8475", never "0.8474999999999999").
_GRID_STEPS = [i / 10000 for i in range(5000, 10001, 25)]
_TOP_STEPS = [0.999, 0.9993, 0.9995, 0.9997, 0.9998, 0.9999,
              0.99992, 0.99994, 0.99996, 0.99997, 0.99998, 0.99999, 0.999995, 0.999998, 0.999999, 1.0]
GRID = np.array(sorted(set(_GRID_STEPS) | set(_TOP_STEPS)))

RELIABILITY_EDGES = np.array([0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.97, 0.98, 0.99, 0.995, 0.999, 0.9999, 1.0])
"""Default confidence bins for reliability tables. The last bin is closed on the right so 1.0 is included."""


def threshold_key(t):
    """Format a grid threshold as the map's string key: at most 6 decimals, no trailing zeros (``"0.5"``, ``"1.0"``)."""
    s = f"{t:.6f}".rstrip("0")
    return s + "0" if s.endswith(".") else s


def predict_verdicts(p_correct, p_incorrect):
    """The model's verdict and confidence, with the same tie rule as the API.

    sidewalk-ai-api takes ``logits.argmax(-1)``, which returns the first index on a tie; index 0 is ``correct``. So
    the verdict is ``incorrect`` only when ``p_incorrect > p_correct``. Confidence is the max softmax probability.

    Returns:
        ``(verdict, confidence)`` as numpy arrays (strings ``correct``/``incorrect``, floats).
    """
    p_correct = np.asarray(p_correct, dtype=float)
    p_incorrect = np.asarray(p_incorrect, dtype=float)
    verdict = np.where(p_incorrect > p_correct, "incorrect", "correct")
    return verdict, np.maximum(p_correct, p_incorrect)


MAP_Z = 1.645
"""z for the map values: each is the lower end of a two-sided 90% Wilson interval for its isotonic block."""


def wilson_lower(p, n, z=MAP_Z):
    """Lower end of the Wilson score interval for a mean ``p`` (may be a fractional outcome rate) on ``n`` examples.

    Returns 0 where ``n`` is 0. Example: ``wilson_lower(0.95, 100)`` is 0.9008 at the default 90% level.
    """
    p = np.asarray(p, dtype=float)
    n = np.asarray(n, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        z2n = z * z / n
        lo = (p + z2n / 2 - z * np.sqrt(p * (1 - p) / n + z2n / (4 * n))) / (1 + z2n)
    return np.where(n > 0, np.clip(lo, 0.0, 1.0), 0.0)


class IsotonicCurve:
    """A fitted non-decreasing step curve of P(verdict correct) against confidence, with its isotonic blocks.

    Attributes:
        block_start: ascending confidence at which each block starts (its smallest fitted confidence).
        block_mean: the block's (weighted) mean outcome, i.e. the isotonic fit.
        block_n: the block's size (Kish effective size when weighted), used for the confidence bound.
        n, x_min, x_max: number of examples and the fitted confidence range.

    The curve at ``c`` is the mean of the last block starting at or before ``c``, and 0 below ``x_min``: no
    extrapolation below the data, and no linear interpolation between blocks (the step value is never above it).
    """

    def __init__(self, confidence, correct, weight=None):
        confidence = np.asarray(confidence, dtype=float)
        correct = np.asarray(correct, dtype=float)
        weight = np.ones_like(confidence) if weight is None else np.asarray(weight, dtype=float)
        order = np.argsort(confidence, kind="mergesort")
        x, w = confidence[order], weight[order]
        iso = IsotonicRegression(increasing=True, out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(confidence, correct, sample_weight=weight)
        fitted = iso.predict(x)
        # a new block starts wherever the fitted value changes
        starts = np.r_[0, np.flatnonzero(np.diff(fitted) != 0) + 1]
        ends = np.r_[starts[1:], len(x)]
        self.block_start = x[starts]
        self.block_mean = fitted[starts]
        sw = np.array([w[a:b].sum() for a, b in zip(starts, ends)])
        sw2 = np.array([(w[a:b] ** 2).sum() for a, b in zip(starts, ends)])
        self.block_n = np.where(sw2 > 0, sw ** 2 / np.where(sw2 > 0, sw2, 1), 0.0)
        self.n, self.x_min, self.x_max = len(x), float(x[0]), float(x[-1])

    def _block(self, c):
        return np.searchsorted(self.block_start, np.atleast_1d(np.asarray(c, dtype=float)), side="right") - 1

    def value(self, c):
        """Point estimate (block mean) at confidence(s) ``c``; 0 below the fitted range."""
        b = self._block(c)
        return np.where(b >= 0, self.block_mean[np.clip(b, 0, None)], 0.0)

    def lower(self, c, z=MAP_Z):
        """Conservative value at ``c``: the running maximum over blocks of each block's Wilson lower bound.

        The running maximum keeps the result non-decreasing: under the monotone model, a lower bound for an
        earlier block is also a lower bound for every later one.
        """
        lb = np.maximum.accumulate(wilson_lower(self.block_mean, self.block_n, z))
        b = self._block(c)
        return np.where(b >= 0, lb[np.clip(b, 0, None)], 0.0)


def fit_isotonic(confidence, correct, weight=None):
    """Fit a non-decreasing curve of P(verdict correct) against confidence.

    Args:
        confidence: array of softmax confidences for the predicted class (0.5..1.0 for a binary model).
        correct: outcome per example in [0, 1]: 0/1, or a fractional share (e.g. the share of human votes that
            agree with the verdict); isotonic regression fits the mean either way.
        weight: optional per-example weights (e.g. prior-correction weights); defaults to 1.

    Returns:
        An ``IsotonicCurve``, or ``None`` when there is no data, which ``curve_to_class_map`` turns into an
        all-zero map (the verdict never gates).
    """
    if len(np.asarray(confidence)) == 0:
        return None
    return IsotonicCurve(confidence, correct, weight)


def fit_range(curve):
    """The confidence range a curve was fitted on, for reports: ``{"n", "min_confidence", "max_confidence"}``."""
    if curve is None:
        return {"n": 0, "min_confidence": None, "max_confidence": None}
    return {"n": int(curve.n), "min_confidence": curve.x_min, "max_confidence": curve.x_max}


def curve_value(curve, c, conservative=False):
    """The curve at confidence(s) ``c`` (block mean, or its lower bound if ``conservative``); 0 with no curve."""
    c = np.atleast_1d(np.asarray(c, dtype=float))
    if curve is None:
        return np.zeros_like(c)
    return curve.lower(c) if conservative else curve.value(c)


def curve_to_class_map(curve, grid=GRID, conservative=True):
    """Sample the curve on ``grid`` and return one class's inner dict, ascending, values rounded to 4 decimals.

    By default each value is the lower end of a 90% Wilson interval for its block, so **the served accuracy is a
    lower confidence bound on precision**, not a point estimate: with blocks of thousands this changes the value by
    well under 0.01, while a thin block at the low end of the confidence range cannot clear the gate on a lucky
    estimate. Rounding preserves monotonicity.
    """
    values = np.round(curve_value(curve, grid, conservative), 4)
    return {threshold_key(t): float(v) for t, v in zip(grid, values)}


def cap_class_map(class_map, cap=HOLD_BACK_CAP):
    """Cap every value of one class's map at ``cap`` (used to hold a class back from the gate); stays monotone."""
    return {k: min(v, cap) for k, v in class_map.items()}


def build_api_map(curves, grid=GRID, conservative=True):
    """Build the full API map from ``{"correct": curve_or_None, "incorrect": curve_or_None}``."""
    return {cls: curve_to_class_map(curves.get(cls), grid, conservative) for cls in API_CLASSES}


def api_lookup(class_map, confidence):
    """Serve ``confidence`` through one class's map exactly as sidewalk-ai-api's ``main.py`` does.

    Copied from the ``/process`` handler ("Find the highest mapping that's less than or equal to the confidence"):
    walk the dict in insertion order and keep the last value whose threshold is ``<= confidence``; 0 if none.
    """
    accuracy = 0
    for threshold, acc in class_map.items():
        if float(threshold) <= confidence:
            accuracy = acc
    return accuracy


def api_lookup_many(class_map, confidences):
    """Vectorised ``api_lookup`` for an ascending map (what ``validate_api_map`` checks). Returns a float array."""
    thresholds = np.array([float(k) for k in class_map], dtype=float)
    values = np.array(list(class_map.values()), dtype=float)
    idx = np.searchsorted(thresholds, np.asarray(confidences, dtype=float), side="right") - 1
    out = np.where(idx >= 0, values[np.clip(idx, 0, None)], 0.0)
    return out


def validate_api_map(api_map):
    """Raise ``ValueError`` unless ``api_map`` is in the strict serving format.

    Checks: exactly the keys ``correct`` and ``incorrect``; string thresholds that parse as floats in [0, 1] with at
    most 6 decimals, strictly ascending in insertion order; values in [0, 1] with at most 4 decimals, non-decreasing.
    """
    if set(api_map) != set(API_CLASSES):
        raise ValueError(f"map keys must be {API_CLASSES}, got {sorted(api_map)}")
    for cls, class_map in api_map.items():
        if not class_map:
            raise ValueError(f"{cls}: empty map")
        thresholds = []
        for k, v in class_map.items():
            if not isinstance(k, str):
                raise ValueError(f"{cls}: threshold {k!r} is not a string")
            t = float(k)
            if not 0.0 <= t <= 1.0 or ("." in k and len(k.split(".")[1]) > 6):
                raise ValueError(f"{cls}: bad threshold {k!r}")
            if not isinstance(v, (int, float)) or not 0.0 <= v <= 1.0 or round(v, 4) != v:
                raise ValueError(f"{cls}: bad value {v!r} at {k}")
            thresholds.append(t)
        if any(b <= a for a, b in zip(thresholds, thresholds[1:])):
            raise ValueError(f"{cls}: thresholds are not strictly ascending")
        values = list(class_map.values())
        if any(b < a for a, b in zip(values, values[1:])):
            raise ValueError(f"{cls}: values decrease")


def reliability_table(confidence, correct, predicted=None, weight=None, edges=RELIABILITY_EDGES):
    """Per-confidence-bin counts and rates, for reliability diagrams and ECE.

    Args:
        confidence: confidences of the predicted class.
        correct: 0/1 outcomes.
        predicted: optional predicted accuracy per example (e.g. the map's served value); its weighted mean per bin
            is reported as ``predicted``. Defaults to the confidence itself.
        weight: optional per-example weights.
        edges: bin edges; bins are ``[e_i, e_{i+1})`` except the last, which also includes its right edge.

    Returns:
        DataFrame with one row per non-empty bin: ``bin_lo, bin_hi, n, weight, mean_confidence, predicted,
        observed, se`` where ``observed`` is the weighted rate of correct verdicts and ``se`` its binomial standard
        error (using the effective sample size when weighted).
    """
    confidence = np.asarray(confidence, dtype=float)
    correct = np.asarray(correct, dtype=float)
    predicted = confidence if predicted is None else np.asarray(predicted, dtype=float)
    weight = np.ones_like(confidence) if weight is None else np.asarray(weight, dtype=float)
    edges = np.asarray(edges, dtype=float)
    idx = np.clip(np.searchsorted(edges, confidence, side="right") - 1, 0, len(edges) - 2)
    rows = []
    for b in range(len(edges) - 1):
        m = idx == b
        if not m.any():
            continue
        w = weight[m]
        if w.sum() <= 0:          # e.g. a class reweighted to a prior of 0: nothing to measure in this bin
            continue
        obs = np.average(correct[m], weights=w)
        n_eff = w.sum() ** 2 / (w ** 2).sum()
        rows.append({"bin_lo": edges[b], "bin_hi": edges[b + 1], "n": int(m.sum()), "weight": float(w.sum()),
                     "mean_confidence": float(np.average(confidence[m], weights=w)),
                     "predicted": float(np.average(predicted[m], weights=w)),
                     "observed": float(obs), "se": float(np.sqrt(obs * (1 - obs) / n_eff))})
    return pd.DataFrame(rows, columns=["bin_lo", "bin_hi", "n", "weight", "mean_confidence", "predicted",
                                       "observed", "se"])


def expected_calibration_error(table):
    """Weight-averaged ``|predicted - observed|`` over the bins of a ``reliability_table``."""
    if len(table) == 0:
        return float("nan")
    return float(np.average((table.predicted - table.observed).abs(), weights=table.weight))


def gate_summary(served, correct, weight=None, gate=GATE):
    """Coverage and precision of the verdicts a gate lets through.

    Args:
        served: served accuracy per example.
        correct: 0/1 outcome per example, or ``None`` when outcomes are unknown (precision is then NaN).
        weight: optional weights.

    Returns:
        dict with ``coverage`` (weighted fraction with served >= gate), ``n_pass`` and ``precision`` (weighted
        rate of correct verdicts among those passing; NaN if none pass).
    """
    served = np.asarray(served, dtype=float)
    weight = np.ones_like(served) if weight is None else np.asarray(weight, dtype=float)
    passed = served >= gate
    coverage = float(weight[passed].sum() / weight.sum()) if weight.sum() > 0 else float("nan")
    precision = float("nan")
    if correct is not None and passed.any():
        precision = float(np.average(np.asarray(correct, dtype=float)[passed], weights=weight[passed]))
    return {"coverage": coverage, "n_pass": int(passed.sum()), "precision": precision}
