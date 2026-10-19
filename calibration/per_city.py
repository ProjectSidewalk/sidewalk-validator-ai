"""Per-city precision of the AI's gated votes against the human majority, from production outcomes (issue #15).

Reads the ``labels_oof.csv`` that ``python -m calibration.calibrate_from_production`` writes (one row per assessed
label: ``schema, label_id, label_type, ai_result, ai_accuracy, new_accuracy, oof_accuracy, labelled, correct, ...``)
and asks whether the 0.92 gate (SidewalkWebpage's ``ai-validation-min-accuracy``, ``GATE`` in ``_isotonic_map``)
means the same thing in every city. For each city x label type, among the labelled rows (>= 2 human votes) whose
served accuracy is >= 0.92, it reports, for Agree and Disagree verdicts separately:

* ``n``: gated votes with a human outcome,
* ``served``: their mean served accuracy (what the map promised),
* ``observed``: their mean outcome (``correct``: 1 if the human majority matches the verdict, 0.5 for a tie, 0
  otherwise, under ``calibrate_from_production``'s default ``tie_half``), with a Wilson 95% interval,
* ``gap`` = served - observed, per verdict (positive: the map over-promised in that city). Agree and Disagree are
  never pooled: their gaps have opposite signs often enough that a pooled number hides both.

**Which served accuracy.** ``--served-column`` picks it: ``ai_accuracy`` (default) is what production served when the
vote was cast (the maps before sidewalk-ai-api PR #16); ``oof_accuracy`` is the out-of-fold value of the Phase 0 maps
now live (API 1.0.5), the honest estimate of what they serve; ``new_accuracy`` is the shipped Phase 0 map in-sample.
No Disagree map of the Phase 0 set reaches 0.92 (``summary.csv``: ``below gate`` for all five), so under them only
Agree verdicts vote; the few out-of-fold Disagree rows above 0.92 come from the fold maps, not the shipped one.

**Naming.** A city is named only if ``--cities-json`` (a saved ``/v3/api/cities`` response) lists it with
``visibility == "public"``; it is named by its ``city_id`` (``seattle-wa``), the id ``split.csv`` and
``split_data.py --holdout-cities`` use. Every other deployment, including any the file does not identify (see
``match_city_id``), is pooled into one row ``other (N private deployments)``, where N counts the pooled deployments
with at least one gated vote. Without ``--cities-json`` nothing is named. The internal ``sidewalk_validation`` schema
is dropped. Per private deployment, only a count of strata below the gate is reported, never a name.

Outputs under ``--out-dir``: ``per_city.md`` (one table per label type, the Agree strata below the gate, and the
run's settings), ``per_city.csv`` (the long table, one row per city x type x verdict, named cities only plus the
pooled private row) and ``per_city_<type>.png`` (served vs observed per city, one panel per verdict). The CSV holds
only aggregates, but read it before committing it anywhere.

Usage::

    python -m calibration.per_city --labels ~/Data/<pull>/phase0/labels_oof.csv \\
        --cities-json ~/Data/<pull>/cities.json --out-dir ~/Data/<pull>/per_city
    # the same under the Phase 0 maps now live (out-of-fold):
    python -m calibration.per_city --labels ~/Data/<pull>/phase0/labels_oof.csv \\
        --cities-json ~/Data/<pull>/cities.json --served-column oof_accuracy --out-dir ~/Data/<pull>/per_city_oof

Example (the interval)::

    >>> lo, hi = wilson_interval(0.8, 10)
    >>> round(float(lo), 4), round(float(hi), 4)
    (0.4902, 0.9433)
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from calibration import _isotonic_map as im
from calibration._plots import COLORS

LABEL_TYPES = ["CurbRamp", "NoCurbRamp", "Obstacle", "SurfaceProblem", "Crosswalk"]
VERDICTS = ("Agree", "Disagree")
EXCLUDED_SCHEMAS = {"sidewalk_validation"}  # internal test deployment
SERVED_COLUMNS = ("ai_accuracy", "oof_accuracy", "new_accuracy")
Z95 = 1.96
ALL_CITIES = "all cities"
STRATA_COLUMNS = ["city", "label_type", "verdict", "n", "served", "observed", "ci_lo", "ci_hi", "gap"]


# ---------------------------------------------------------------------------------------------------------------------
# Naming: public deployments only
# ---------------------------------------------------------------------------------------------------------------------

def load_cities(path):
    """``{city_id: visibility}`` from a saved ``/v3/api/cities`` response (``{"cities": [...]}`` or a bare list)."""
    with open(path) as f:
        cities = json.load(f)
    cities = cities.get("cities", cities) if isinstance(cities, dict) else cities
    return {c["city_id"]: c.get("visibility") for c in cities}


COUNTRY_SUFFIXES = {"brazil", "canada", "chile", "ecuador", "india", "mexico", "netherlands", "switzerland",
                    "taiwan"}
"""Region suffixes besides two-letter codes that a ``city_id`` may add to its schema's name (``rancagua-chile``)."""


def _is_region_suffix(token):
    return (len(token) == 2 and token.isalpha()) or token in COUNTRY_SUFFIXES


def match_city_id(schema_city, cities):
    """The ``city_id`` that identifies a schema's city part (``seattle``, ``st_louis``), or ``None``.

    The slug (``_`` -> ``-``) is identified only if it IS a listed ``city_id``, or if exactly one listed ``city_id``
    is the slug plus one region suffix: a two-letter code or a country in ``COUNTRY_SUFFIXES`` (``seattle`` ->
    ``seattle-wa``, ``st_louis`` -> ``st-louis-mo``). Anything else matches nothing and is pooled as private: a
    schema the listing does not contain never borrows the name of a listed city that merely starts with it
    (``springfield`` is not ``springfield-gardens``), and two candidates are ambiguous.

    >>> match_city_id("st_louis", {"st-louis-mo": "public"})
    'st-louis-mo'
    >>> match_city_id("springfield", {"springfield-gardens": "public"}) is None
    True
    >>> match_city_id("springfield", {"springfield-ma": "private", "springfield-il": "public"}) is None
    True
    """
    slug = schema_city.replace("_", "-")
    if slug in cities:
        return slug
    candidates = [cid for cid in cities
                  if cid.startswith(slug + "-") and _is_region_suffix(cid[len(slug) + 1:])]
    return candidates[0] if len(candidates) == 1 else None


def public_name(schema_city, cities):
    """The public ``city_id`` to show for a schema's city part, or ``None`` (private, unlisted or ambiguous)."""
    cid = match_city_id(schema_city, cities or {})
    return cid if cid is not None and cities[cid] == "public" else None


def private_label(n):
    """The display name of the pooled row."""
    return f"other ({n} private deployment{'s' if n != 1 else ''})"


def assign_display_city(df, cities, count_mask=None):
    """Add ``city``: the public ``city_id``, or the pooled private label, from ``schema``. Returns a copy.

    ``df`` must already exclude the internal schema. The pooled label counts the distinct private deployments among
    the rows where ``count_mask`` is true (``main`` passes the gated rows), or among all rows without it.
    """
    out = df.copy()
    schema_city = out["schema"].str.replace("sidewalk_", "", n=1, regex=False)
    names = {s: public_name(s, cities) for s in schema_city.unique()}
    counted = set(schema_city if count_mask is None else schema_city[np.asarray(count_mask, dtype=bool)])
    n_private = sum(names[s] is None for s in counted)
    out["city"] = schema_city.map(lambda s: names[s] if names[s] is not None else private_label(n_private))
    return out


# ---------------------------------------------------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------------------------------------------------

def wilson_interval(p, n, z=Z95):
    """Two-sided Wilson score interval ``(lo, hi)`` for a mean ``p`` (may be fractional) on ``n`` examples.

    The upper end is the lower end of the complementary rate, so both use ``_isotonic_map.wilson_lower``.
    ``(0, 0)`` where ``n`` is 0.
    """
    lo = im.wilson_lower(p, n, z)
    hi = np.where(np.asarray(n, dtype=float) > 0, 1.0 - im.wilson_lower(1.0 - np.asarray(p, dtype=float), n, z), 0.0)
    return lo, hi


def gated_rows(df, served_column="ai_accuracy", gate=im.GATE):
    """Labelled rows with an outcome whose served accuracy clears the gate."""
    labelled = df["labelled"].astype(bool) if "labelled" in df.columns else pd.Series(True, index=df.index)
    m = labelled & df["correct"].notna() & (df[served_column] >= gate)
    return df[m]


def _summarise(g, served_column):
    n = len(g)
    served = float(g[served_column].mean()) if n else np.nan
    observed = float(g["correct"].mean()) if n else np.nan
    lo, hi = wilson_interval(observed if n else 0.0, n)
    return {"n": n, "served": served, "observed": observed, "ci_lo": float(lo), "ci_hi": float(hi),
            "gap": served - observed}


def per_city_strata(df, served_column="ai_accuracy", gate=im.GATE):
    """One row per city x label type x verdict over the gated votes, plus an ``all cities`` row per type x verdict.

    Args:
        df: rows with ``city, label_type, ai_result, correct, labelled`` and ``served_column``.

    Returns:
        DataFrame with ``STRATA_COLUMNS``; strata with no gated votes are absent.
    """
    g = gated_rows(df, served_column, gate)
    rows = []
    for (city, lt, v), s in g.groupby(["city", "label_type", "ai_result"], sort=True):
        rows.append({"city": city, "label_type": lt, "verdict": v, **_summarise(s, served_column)})
    for (lt, v), s in g.groupby(["label_type", "ai_result"], sort=True):
        rows.append({"city": ALL_CITIES, "label_type": lt, "verdict": v, **_summarise(s, served_column)})
    return pd.DataFrame(rows, columns=STRATA_COLUMNS)


def below_gate(strata, min_n=50, gate=im.GATE, verdict="Agree"):
    """Strata of one verdict, n >= ``min_n``, whose observed precision is below the gate, worst first.

    Adds ``shortfall`` = gate - observed and ``ci_below`` (the whole Wilson interval is below the gate).
    """
    s = strata[(strata.verdict == verdict) & (strata.n >= min_n) & (strata.observed < gate)].copy()
    s["shortfall"] = gate - s.observed
    s["ci_below"] = s.ci_hi < gate
    return s.sort_values("shortfall", ascending=False).reset_index(drop=True)


def city_gate(df, city, label_type, served_column="ai_accuracy", gate=im.GATE, min_n=50, verdict="Agree"):
    """The lowest served-accuracy threshold at which one city's gated votes of a type clear the gate the way the
    Phase 0 hold-back rule does: the Wilson 95% LOWER bound of their precision is >= ``gate``, on >= ``min_n`` votes.

    Scans the distinct served values >= ``gate`` (the maps are step functions, so these are the only thresholds that
    change anything) and returns ``(threshold, kept)``: the smallest qualifying threshold and the share of the
    city's gated votes it keeps; ``(None, 0.0)`` if none qualifies. The threshold is chosen on the same rows it is
    measured on, so it is optimistic: read it as the size of the change a per-city gate would need, not as a value
    to ship. Meaningless for the pooled private row, which is not one deployment.

    >>> d = pd.DataFrame({"city": "a", "label_type": "T", "ai_result": "Agree", "labelled": True,
    ...                   "ai_accuracy": [0.93] * 20 + [0.97] * 80, "correct": [0] * 20 + [1] * 80})
    >>> city_gate(d, "a", "T")
    (0.97, 0.8)
    """
    g = gated_rows(df, served_column, gate)
    g = g[(g.city == city) & (g.label_type == label_type) & (g.ai_result == verdict)]
    if not len(g):
        return None, 0.0
    served = g[served_column].to_numpy(dtype=float)
    correct = g["correct"].to_numpy(dtype=float)
    for t in np.unique(served):
        keep = served >= t
        if keep.sum() < min_n:
            break
        if float(im.wilson_lower(correct[keep].mean(), keep.sum(), Z95)) >= gate:
            return float(t), float(keep.mean())
    return None, 0.0


def private_below_gate(df, served_column="ai_accuracy", gate=im.GATE, min_n=50, verdict="Agree"):
    """Counts only, never names: the pooled private deployments' strata taken one deployment at a time.

    Returns ``{"strata", "below", "ci_below", "votes_below", "votes_total"}``: the number of private deployment x
    type strata of ``verdict`` with >= ``min_n`` gated votes, how many have observed precision below ``gate``, how
    many have the whole Wilson 95% interval below it, the gated votes in the strata below, and all gated votes of
    ``verdict`` (every city) for the share.
    """
    g = gated_rows(df, served_column, gate)
    g = g[g.ai_result == verdict]
    priv = g[g.city.str.startswith("other (")]
    st = priv.groupby(["schema", "label_type"]).correct.agg(["size", "mean"])
    st = st[st["size"] >= min_n]
    below = st[st["mean"] < gate]
    _, hi = wilson_interval(below["mean"].to_numpy(), below["size"].to_numpy())
    return {"strata": int(len(st)), "below": int(len(below)), "ci_below": int((np.asarray(hi) < gate).sum()),
            "votes_below": int(below["size"].sum()), "votes_total": int(len(g))}


# ---------------------------------------------------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------------------------------------------------

def _city_order(strata, label_type, min_n):
    """Cities with at least one verdict at n >= min_n, by total gated votes (desc); pooled private row and the
    all-cities row last."""
    s = strata[(strata.label_type == label_type) & (strata.city != ALL_CITIES)]
    keep = s[s.n >= min_n].city.unique()
    totals = s[s.city.isin(keep)].groupby("city").n.sum()
    named = [c for c in totals.sort_values(ascending=False, kind="mergesort").index if not c.startswith("other (")]
    other = [c for c in totals.index if c.startswith("other (")]
    return named + other


def _cell(r, min_n):
    if r is None:
        return ["0", "—", "—", "—"]
    if r.n < min_n:
        return [f"{r.n:,}", "—", "—", "—"]
    return [f"{r.n:,}", f"{r.served:.3f}", f"{r.observed:.3f} ({r.ci_lo:.3f}–{r.ci_hi:.3f})", f"{r.gap:+.3f}"]


def markdown_tables(strata, min_n=50, gate=im.GATE):
    """One markdown table per label type, plus a footnote of strata omitted for n < ``min_n``."""
    lines = []
    for lt in LABEL_TYPES:
        cities = _city_order(strata, lt, min_n)
        lines += [f"### {lt}", "",
                  "| city | Agree n | Agree served | Agree observed (95% CI) | Agree gap "
                  "| Disagree n | Disagree served | Disagree observed (95% CI) | Disagree gap |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        s = strata[strata.label_type == lt].set_index(["city", "verdict"])
        for city in cities + [ALL_CITIES]:
            cells = []
            for v in VERDICTS:
                cells += _cell(s.loc[(city, v)] if (city, v) in s.index else None, min_n)
            name = f"**{city}**" if city == ALL_CITIES else city
            lines.append("| " + " | ".join([name] + cells) + " |")
        dropped = strata[(strata.label_type == lt) & (strata.city != ALL_CITIES) & ~strata.city.isin(cities)]
        if len(dropped):
            lines.append("")
            lines.append(f"{dropped.city.nunique()} more cities have fewer than {min_n} gated votes of either verdict "
                         f"({int(dropped.n.sum()):,} votes); they are counted in **{ALL_CITIES}**.")
        lines.append("")
    return "\n".join(lines)


def markdown_below_gate(strata, min_n=50, gate=im.GATE, df=None, served_column="ai_accuracy"):
    """The Agree strata (n >= ``min_n``) whose observed precision is below the gate, as a markdown table.

    With ``df`` (the rows behind ``strata``) it adds the per-city threshold whose Wilson lower bound reaches the
    gate (``city_gate``) and the share of that stratum's votes it would keep; n/a on the pooled private row. A
    summary line first counts the strata with n >= ``min_n``, named and pooled separately.
    """
    named_or_pooled = strata[(strata.city != ALL_CITIES) & (strata.verdict == "Agree") & (strata.n >= min_n)]
    pooled = named_or_pooled.city.str.startswith("other (")
    b = below_gate(strata[strata.city != ALL_CITIES], min_n, gate)
    b_pooled = b.city.str.startswith("other (")
    summary = (f"Agree strata with n >= {min_n}: {int((~pooled).sum())} named public city x type strata and "
               f"{int(pooled.sum())} pooled private rows. Below {gate}: {int((~b_pooled).sum())} named "
               f"({int(b[~b_pooled].ci_below.sum())} with the whole interval below, "
               f"{int(b[~b_pooled].n.sum()):,} votes) and {int(b_pooled.sum())} pooled.\n")
    if df is not None:
        pb = private_below_gate(df, served_column, gate, min_n)
        share = pb["votes_below"] / pb["votes_total"] if pb["votes_total"] else 0.0
        summary += (f"\nPrivate deployments one at a time (counts only): {pb['below']} of {pb['strata']} deployment x "
                    f"type Agree strata with n >= {min_n} are below {gate} ({pb['ci_below']} with the whole interval "
                    f"below), holding {pb['votes_below']:,} of all {pb['votes_total']:,} gated Agree votes "
                    f"({share:.1%}).\n")
    if not len(b):
        return summary + f"\nNo city x type stratum with n >= {min_n} gated Agree votes is below {gate}.\n"
    head = "| city | type | n | served | observed (95% CI) | below the gate by | whole CI below |"
    rule = "|---|---|---:|---:|---:|---:|---|"
    if df is not None:
        head += " threshold whose lower bound reaches the gate (votes kept) |"
        rule += "---:|"
    lines = [summary, head, rule]
    for r in b.itertuples():
        line = (f"| {r.city} | {r.label_type} | {r.n:,} | {r.served:.3f} | {r.observed:.3f} "
                f"({r.ci_lo:.3f}–{r.ci_hi:.3f}) | {r.shortfall:.3f} | {'yes' if r.ci_below else 'no'} |")
        if df is not None and r.city.startswith("other ("):
            line += " n/a (pooled) |"
        elif df is not None:
            t, kept = city_gate(df, r.city, r.label_type, served_column, gate, min_n)
            line += f" {t:.4f} ({kept:.0%}) |" if t is not None else f" none with n >= {min_n} |"
        lines.append(line)
    return "\n".join(lines) + "\n"


def plot_city_reliability(strata, label_type, path, min_n=50, gate=im.GATE, served_label="served (mean)"):
    """Served vs observed precision per city for one label type: one panel per verdict, one row per city.

    Each city is a series of two marks: its mean served accuracy (what the map promised) and its observed precision
    with the Wilson 95% interval. A city sits on the gate line's right only if the votes it cast were right at least
    92% of the time. Strata with n < ``min_n`` are not drawn.
    """
    cities = _city_order(strata, label_type, min_n) + [ALL_CITIES]
    s = strata[strata.label_type == label_type].set_index(["city", "verdict"])
    shown = s[s.n >= min_n]
    fig, axes = plt.subplots(1, len(VERDICTS), figsize=(11, 0.32 * len(cities) + 2.0), sharey=True, squeeze=False)
    axes = axes[0]
    y = np.arange(len(cities))[::-1]
    for ax, v in zip(axes, VERDICTS):
        ax.axvline(gate, color=COLORS["gate"], linestyle="--", linewidth=1)
        drawn = 0
        for yi, city in zip(y, cities):
            if (city, v) not in shown.index:
                continue
            r = shown.loc[(city, v)]
            ax.plot(r.served, yi, "D", color=COLORS["new"], markersize=7, markeredgecolor="white", zorder=2)
            ax.plot([r.ci_lo, r.ci_hi], [yi, yi], color=COLORS["observed"], linewidth=2, solid_capstyle="round",
                    zorder=3)
            ax.plot(r.observed, yi, "o", color=COLORS["observed"], markersize=8, markeredgecolor="white", zorder=4)
            ax.annotate(f"n={r.n:,}", (1.0, yi), xytext=(4, 0), textcoords="offset points", va="center",
                        fontsize=7, color=COLORS["gate"], annotation_clip=False)
            drawn += 1
        if not drawn:
            ax.text(0.5, 0.5, f"no stratum with n >= {min_n}\ngated {v} votes", transform=ax.transAxes,
                    ha="center", va="center", color=COLORS["gate"])
        ax.set_title(f"{label_type}: gated {v} votes")
        ax.set_xlabel("precision against the human majority")
        lows = shown.xs(v, level="verdict").ci_lo if v in shown.index.get_level_values("verdict") else []
        ax.set_xlim(min(0.5, np.floor(min(lows) * 10) / 10) if len(lows) else 0.5, 1.0)
        ax.grid(True, axis="x", alpha=0.3)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(cities, fontsize=8)
    for lab in axes[0].get_yticklabels():
        if lab.get_text() == ALL_CITIES:
            lab.set_fontweight("bold")
    handles = [plt.Line2D([], [], color=COLORS["gate"], linestyle="--", linewidth=1, label=f"gate {gate}"),
               plt.Line2D([], [], color=COLORS["observed"], marker="o", markersize=8, linewidth=2,
                          label="observed precision, Wilson 95% CI"),
               plt.Line2D([], [], color=COLORS["new"], marker="D", markersize=7, linestyle="none",
                          label=served_label)]
    fig.legend(handles=handles, loc="upper center", ncol=3, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.45 / fig.get_figheight()))
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------------------------------

def load_labels(path):
    """Read ``labels_oof.csv`` (``labelled``/``tie`` written as True/False) and drop the internal schema."""
    df = pd.read_csv(path, low_memory=False, true_values=["True"], false_values=["False"])
    return df[~df.schema.isin(EXCLUDED_SCHEMAS)]


def main(argv=None):
    p = argparse.ArgumentParser(description="Per-city precision of the AI's gated votes (issue #15).")
    p.add_argument("--labels", required=True, help="labels_oof.csv from calibration.calibrate_from_production")
    p.add_argument("--cities-json", default=None,
                   help="saved /v3/api/cities response; only its public deployments are named")
    p.add_argument("--served-column", choices=SERVED_COLUMNS, default="ai_accuracy",
                   help="served accuracy the gate applies to (default: what production served)")
    p.add_argument("--min-n", type=int, default=50, help="gated votes a stratum needs to be shown (default 50)")
    p.add_argument("--out-dir", required=True)
    args = p.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    cities = load_cities(args.cities_json) if args.cities_json else {}
    df = load_labels(args.labels)
    gated_mask = df.index.isin(gated_rows(df, args.served_column).index)
    df = assign_display_city(df, cities, count_mask=gated_mask)
    strata = per_city_strata(df, args.served_column)

    gated = gated_rows(df, args.served_column)
    n_named = df.loc[~df.city.str.startswith("other ("), "city"].nunique()
    header = [
        f"# Per-city precision of gated votes ({args.served_column} >= {im.GATE})", "",
        f"Input: `{os.path.basename(args.labels)}`, {int(df.labelled.astype(bool).sum()):,} labelled labels, "
        f"{len(gated):,} gated votes; {n_named} public deployments in the data are named, the rest pooled. "
        f"Strata with fewer than {args.min_n} gated votes are shown as —.", "",
        "`served` is the mean served accuracy of the gated votes, `observed` their precision against the human "
        "majority (ties count 0.5) with a Wilson 95% interval, `gap` served minus observed for that verdict.", ""]
    body = markdown_tables(strata, args.min_n)
    below = ["## Agree strata below the gate", "",
             markdown_below_gate(strata, args.min_n, df=df, served_column=args.served_column)]
    text = "\n".join(header) + "\n" + body + "\n" + "\n".join(below)
    with open(os.path.join(args.out_dir, "per_city.md"), "w") as f:
        f.write(text)
    strata.to_csv(os.path.join(args.out_dir, "per_city.csv"), index=False)
    served_label = {"ai_accuracy": "served (mean)", "oof_accuracy": "Phase 0 map, out-of-fold (mean)",
                    "new_accuracy": "Phase 0 map (mean)"}[args.served_column]
    for lt in LABEL_TYPES:
        plot_city_reliability(strata, lt, os.path.join(args.out_dir, f"per_city_{lt.lower()}.png"), args.min_n,
                              served_label=served_label)
    print(text)
    print(f"wrote {args.out_dir}/per_city.md, per_city.csv and per_city_<type>.png")


if __name__ == "__main__":
    main()
