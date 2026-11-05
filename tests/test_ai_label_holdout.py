"""ai_label_holdout.py (issue #79): the seeded, pool-stable hold-out of AI-placed labels carved before selection, its
record in generation_run.json, and the moved campaign tiers. Pure pandas; no GPU, no network."""
import doctest
import hashlib
from types import SimpleNamespace

import pandas as pd
import pytest

import ai_label_holdout as h
import data_generation as dg


def test_doctests():
    failed, _ = doctest.testmod(h)
    assert failed == 0


def pool(n_ai=40, n_human=10, city="toy-city", incorrect_every=4, time_created="2026-09-22 19:03"):
    """``n_human`` human rows then ``n_ai`` AI rows; every ``incorrect_every``-th AI row is incorrect."""
    rows = [{"city": city, "label_id": i, "author": "human", "label_class": "correct" if i % 2 else "incorrect",
             "class_source": "margin", "pano_source": "gsv", "time_created": time_created} for i in range(n_human)]
    rows += [{"city": city, "label_id": n_human + i, "author": "ai",
              "label_class": "incorrect" if i % incorrect_every == 0 else "correct", "class_source": "margin",
              "pano_source": "gsv", "time_created": time_created} for i in range(n_ai)]
    return pd.DataFrame(rows)


def args(**kw):
    base = {"seed": 42, "ai_holdout_frac": h.DEFAULT_FRAC, "include_ai_labels": False}
    base.update(kw)
    return SimpleNamespace(**base)


# 1. the draw ---------------------------------------------------------------------------------------------------------

def test_holdout_fraction_is_deterministic_seeded_and_independent_of_the_downsample_rank():
    u = "toy-city:7"
    assert h.holdout_fraction(42, u) == h.holdout_fraction(42, u)
    assert 0 <= h.holdout_fraction(42, u) < 1
    assert h.holdout_fraction(42, u) != h.holdout_fraction(43, u)
    # not the downsample stream: data_generation.label_rank hashes f"{seed}:{uid}" with no salt
    unsalted = int(dg.label_rank(42, u)[:12], 16) / 16 ** 12
    assert h.holdout_fraction(42, u) != unsalted
    assert h.holdout_fraction(42, u, salt="") == int(hashlib.sha256(f":42:{u}".encode()).hexdigest()[:12], 16) / 16 ** 12


# 2. carve -------------------------------------------------------------------------------------------------------------

def test_carve_moves_only_ai_rows_and_partitions_the_pool():
    e = pool()
    rest, held = h.carve(e, 42, 0.5)
    assert len(held) and (held.label_uid.str.split(":").str[1].astype(int) >= 10).all()  # AI ids only
    humans = e[e.author == "human"].reset_index(drop=True)
    pd.testing.assert_frame_equal(rest[rest.author == "human"].reset_index(drop=True), humans)
    assert list(rest.columns) == list(e.columns)
    uid = lambda d: set(d.city + ":" + d.label_id.astype(str))  # noqa: E731
    assert uid(rest) | set(held.label_uid) == uid(e) and not uid(rest) & set(held.label_uid)
    assert list(held.columns) == h.HOLDOUT_COLUMNS and held.label_uid.is_monotonic_increasing
    assert (held.holdout_rank < 0.5).all()


def test_carve_on_a_pool_without_ai_rows_holds_out_nothing():
    e = pool(n_ai=0)
    rest, held = h.carve(e, 42)
    assert len(held) == 0 and list(held.columns) == h.HOLDOUT_COLUMNS
    pd.testing.assert_frame_equal(rest, e)


# 3. pool stability ----------------------------------------------------------------------------------------------------

def test_one_label_entering_or_leaving_moves_at_most_that_label():
    e = pool(n_ai=200)
    _, held = h.carve(e, 42)
    extra = pd.DataFrame([{"city": "toy-city", "label_id": 9999, "author": "ai", "label_class": "incorrect",
                           "class_source": "margin", "pano_source": "gsv", "time_created": ""}])
    _, more = h.carve(pd.concat([e, extra], ignore_index=True), 42)
    assert set(more.label_uid) - set(held.label_uid) <= {"toy-city:9999"}
    assert set(held.label_uid) <= set(more.label_uid)
    gone = held.label_uid.iloc[0]
    _, fewer = h.carve(e[(e.city + ":" + e.label_id.astype(str)) != gone], 42)
    assert set(held.label_uid) - set(fewer.label_uid) == {gone} and set(fewer.label_uid) <= set(held.label_uid)


# 4. the fraction ------------------------------------------------------------------------------------------------------

def test_fraction_is_honoured_in_expectation():
    e = pool(n_ai=2000, n_human=0)
    _, held = h.carve(e, 42, 0.5)
    assert 0.45 < len(held) / 2000 < 0.55
    _, held3 = h.carve(e, 42, 0.3)
    assert 0.25 < len(held3) / 2000 < 0.35 and set(held3.label_uid) <= set(held.label_uid)  # nested by frac


@pytest.mark.parametrize("bad", [0, 1, -0.1, 1.5])
def test_fraction_bounds_raise(bad):
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        h.carve(pool(), 42, bad)


def test_frac_arg_is_an_argparse_usage_error():
    import argparse
    assert h.frac_arg("0.25") == 0.25
    with pytest.raises(argparse.ArgumentTypeError):
        h.frac_arg("1")


# 5. rejected labels are candidates ------------------------------------------------------------------------------------

def test_every_rejected_ai_label_is_a_candidate():
    e = pool(n_ai=40, incorrect_every=4)
    rejected = set("toy-city:" + e[(e.author == "ai") & (e.label_class == "incorrect")].label_id.astype(str))
    seen = set()
    for seed in range(30):
        _, held = h.carve(e, seed)
        seen |= set(held.label_uid[held.label_class == "incorrect"])
    assert seen == rejected


# 6. tier and stratum --------------------------------------------------------------------------------------------------

def test_stratum_uses_the_campaign_tier_and_blank_when_unknown():
    e = pd.concat([pool(n_ai=30, n_human=0, city="laurens-ia", time_created="2026-09-22T01:15:14Z"),
                   pool(n_ai=30, n_human=0, city="toy-city")], ignore_index=True)
    _, held = h.carve(e, 42)
    by_city = held.groupby("city").tier.unique().to_dict()
    assert list(by_city["laurens-ia"]) == ["band"] and list(by_city["toy-city"]) == [""]
    r = held.iloc[0]
    assert r.stratum == f"{r.city}/{r.label_class}/{r.tier}"
    # no time_created column at all (old caches): tier is blank, nothing raises
    _, held2 = h.carve(e.drop(columns="time_created"), 42)
    assert (held2.tier == "").all()


def test_moved_campaign_names_still_resolve_from_ai_labels_eval():
    from analysis import ai_labels_eval as ae
    assert ae.CAMPAIGNS is h.CAMPAIGNS and ae.TIERS is h.TIERS and ae.tier_of is h.tier_of


# 7. record ------------------------------------------------------------------------------------------------------------

def test_record_writes_a_sorted_unique_csv_and_returns_counts_and_hash(tmp_path):
    e = pool(n_ai=60)
    rest, held = h.carve(e, 42)
    ai_rest = rest[rest.author == "ai"]
    selection = {f"toy-city:{i}": c for i, c in zip(ai_rest.label_id[:5], ai_rest.label_class[:5])}
    rec = h.record(held, rest, selection, str(tmp_path), args(include_ai_labels=True))
    path = tmp_path / h.HOLDOUT_FILE
    written = pd.read_csv(path, dtype=str, keep_default_na=False)
    assert written.label_uid.tolist() == sorted(held.label_uid) and written.label_uid.is_unique
    assert rec["holdout_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert rec["policy"] == "included_tagged" and rec["holdout_frac"] == 0.5 and rec["holdout_seed"] == 42
    assert rec["holdout_salt"] == h.HOLDOUT_SALT and rec["holdout_file"] == h.HOLDOUT_FILE
    hc = rec["holdout_counts"]["toy-city"]
    assert hc["correct"] + hc["incorrect"] == len(held)
    assert sum(rec["included_counts"]["toy-city"].values()) == 5
    assert sum(rec["eligible_ai_counts"]["toy-city"].values()) == 60


def test_record_under_excluded_counts_no_included_rows(tmp_path):
    rest, held = h.carve(pool(), 42)
    sel = {f"toy-city:{i}": "x" for i in rest.label_id}  # even if an AI uid were in the selection
    rec = h.record(held, rest, sel, str(tmp_path), args())
    assert rec["policy"] == "excluded" and rec["included_counts"] == {}


def test_record_writes_an_empty_csv_when_no_ai_rows(tmp_path):
    rest, held = h.carve(pool(n_ai=0), 42)
    rec = h.record(held, rest, {}, str(tmp_path), args(include_ai_labels=True))
    written = pd.read_csv(tmp_path / h.HOLDOUT_FILE)
    assert len(written) == 0 and list(written.columns) == h.HOLDOUT_COLUMNS
    assert rec["holdout_counts"] == {} and rec["included_counts"] == {} and rec["eligible_ai_counts"] == {}


def test_record_is_byte_identical_across_runs(tmp_path):
    rest, held = h.carve(pool(), 42)
    a = h.record(held, rest, {}, str(tmp_path), args())["holdout_sha256"]
    rest2, held2 = h.carve(pool().sample(frac=1, random_state=3), 42)  # row order from the API does not matter
    assert h.record(held2, rest2, {}, str(tmp_path), args())["holdout_sha256"] == a


# 8. load_holdout ------------------------------------------------------------------------------------------------------

def test_load_holdout_round_trips_and_rejects_duplicates(tmp_path):
    rest, held = h.carve(pool(), 42)
    h.record(held, rest, {}, str(tmp_path), args())
    back = h.load_holdout(tmp_path / h.HOLDOUT_FILE)
    assert back.label_uid.tolist() == held.label_uid.tolist() and back.label_id.map(type).eq(str).all()
    dup = pd.concat([back, back.head(1)])
    dup.to_csv(tmp_path / "dup.csv", index=False)
    with pytest.raises(ValueError, match="duplicate label_uid"):
        h.load_holdout(tmp_path / "dup.csv")


def test_apply_prints_the_summary_and_carves_under_both_policies(capsys):
    e = pool()
    r1, h1 = h.apply(e, args(include_ai_labels=False))
    r2, h2 = h.apply(e, args(include_ai_labels=True))
    assert h1.label_uid.tolist() == h2.label_uid.tolist() and len(h1)
    out = capsys.readouterr().out
    assert "policy excluded" in out and "policy included_tagged" in out and "held out of training" in out
