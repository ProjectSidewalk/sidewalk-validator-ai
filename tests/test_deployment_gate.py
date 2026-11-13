"""analysis/deployment_gate.py (issue #52): the per-deployment status report, the replay, the power tables, the slices
and the privacy guards. Synthetic frames only: no network (the listing is stubbed), no private data."""
import doctest
import json
import os

import numpy as np
import pandas as pd
import pytest

from analysis import deployment_gate as g

LISTING = {"cities": [
    {"city_id": "alpha-wa", "visibility": "public", "country_id": "usa", "url": "https://alpha.example.org"},
    {"city_id": "beta-or", "visibility": "public", "country_id": "usa", "url": "https://beta.example.org"},
    {"city_id": "hidden-x", "visibility": "private", "country_id": "atlantis", "city_name_short": "Hiddenburg",
     "city_name_formatted": "Hiddenburg, Atlantis", "url": "https://sidewalk-hiddenburg.example.org"},
    {"city_id": "springfield-ma", "visibility": "private", "country_id": "usa"},
    {"city_id": "springfield-il", "visibility": "public", "country_id": "usa"},
]}
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOC = os.path.join(REPO, "docs", "deployment-gate-2026-11.md")
OUT = os.path.join(REPO, "docs", "deployment-gate-2026-11")


def fetch_ok(url):
    return LISTING


def fetch_fail(url):
    raise ConnectionError("offline")


def rows(schema, label_type, verdict, n, n_right, *, date="2026-11-01", acc=0.95, start=1, votes=(2, 0),
         wrong_votes=None):
    """``n`` assessed labels; the first ``n_right`` have human votes that make the AI right, the rest wrong."""
    agree_right = votes if verdict == "Agree" else votes[::-1]
    wrong = wrong_votes or agree_right[::-1]
    out = []
    for i in range(n):
        a, x = agree_right if i < n_right else wrong
        out.append({"schema": schema, "label_id": start + i, "label_type": label_type, "deleted": "f", "severity": 2,
                    "ai_result": verdict, "ai_accuracy": acc, "ai_confidence": 0.99, "ai_date": date,
                    "model_id": f"sidewalk-validator-ai-{label_type.lower()}", "h_agree": a, "h_dis": x,
                    "h_unsure": 0, "expert1": None, "expert2": None})
    return out


def frame(*parts):
    return pd.DataFrame([r for p in parts for r in p])


def loaded(df):
    d, counts = g.load_pull(df)
    return g.add_outcomes(d), counts


# --------------------------------------------------------------------------------------------------------------------
# the rule
# --------------------------------------------------------------------------------------------------------------------

def test_doctests():
    assert doctest.testmod(g).failed == 0


def test_status_boundaries():
    r = g.Rule()
    assert g.status_from_counts(45, 45, r)["status"] == "insufficient"
    assert g.status_from_counts(50, 50, r)["status"] == "pass"
    assert g.status_from_counts(49, 50, r)["status"] == "undecided"
    assert g.status_from_counts(40, 50, r)["status"] == "fail"
    # at 50 only a perfect run passes; the threshold comes from the rule, not a literal
    assert g.pass_threshold(50, r) == 50


def test_checkpoint_evaluates_first_rows_only():
    r = g.Rule()
    # 100 right, then 37 wrong: n = 137, evaluated on the first 100 (checkpoint), so it still passes
    s = g.status_from_outcomes([1.0] * 100 + [0.0] * 37, r)
    assert (s["n"], s["checkpoint"], s["status"]) == (137, 100, "pass")
    assert s["precision_all"] == pytest.approx(100 / 137)
    assert g.checkpoints(r, 3300) == [50, 100, 200, 400, 800, 1600, 3200]
    with pytest.raises(ValueError):
        g.Rule(schedule=(100, 50))


def test_more_needed_derived_from_rule():
    r = g.Rule()
    assert g.more_needed_if_perfect([], r) == r.schedule[0]
    assert g.more_needed_if_perfect([1.0] * 50, r) == 0
    # one error in the first 50: 50 is out of reach, the next checkpoint (100) passes with 99/100
    assert g.more_needed_if_perfect([0.0] + [1.0] * 49, r) == 50
    assert g.classify(99, 100, r) == "pass"
    r2 = g.Rule(z=2.5)
    assert g.more_needed_if_perfect([], r2) == min(c for c in g.checkpoints(r2, 10_000) if g.classify(c, c, r2) == "pass")


# --------------------------------------------------------------------------------------------------------------------
# reference
# --------------------------------------------------------------------------------------------------------------------

def test_reference_rules():
    df = frame(rows("sidewalk_alpha", "CurbRamp", "Agree", 1, 1, votes=(2, 0)),            # decided, right
               rows("sidewalk_alpha", "CurbRamp", "Agree", 1, 1, votes=(1, 0), start=2),   # one vote: any_vote only
               rows("sidewalk_alpha", "CurbRamp", "Agree", 1, 1, votes=(1, 1), start=3),   # tie
               rows("sidewalk_alpha", "CurbRamp", "Disagree", 1, 1, votes=(2, 0), start=4),  # Disagree right: h_dis > h_agree
               rows("sidewalk_alpha", "CurbRamp", "Disagree", 1, 0, votes=(2, 0), start=5))  # Disagree wrong
    df.loc[0, "h_unsure"] = 5  # Unsure changes nothing
    d, _ = loaded(df)
    m = d.set_index("label_id")
    assert m.loc[1, "out_majority2"] == 1.0
    assert np.isnan(m.loc[2, "out_majority2"]) and m.loc[2, "out_any_vote"] == 1.0
    assert np.isnan(m.loc[3, "out_majority2"]) and m.loc[3, "out_tie_half"] == 0.5 and np.isnan(m.loc[3, "out_any_vote"])
    assert (m.loc[4, "h_agree"], m.loc[4, "h_dis"]) == (0, 2) and m.loc[4, "out_majority2"] == 1.0
    assert m.loc[5, "out_majority2"] == 0.0


def test_excluded_validator_sensitivity_when_columns_present():
    df = frame(rows("sidewalk_alpha", "CurbRamp", "Agree", 60, 60, votes=(2, 0)))
    df["h_agree_excl"], df["h_dis_excl"] = 1, 0
    d, _ = loaded(df)
    assert d["out_majority2_no_excluded"].isna().all()  # one vote left: undecided
    rec = g.compute_strata(d, g.Rule(), "x")[0][("sidewalk_alpha", "CurbRamp", "Agree", "unrecorded")]
    assert rec["sensitivity"]["majority2_no_excluded"]["status"] == "insufficient"


# --------------------------------------------------------------------------------------------------------------------
# loading, keys and eras
# --------------------------------------------------------------------------------------------------------------------

def test_label_id_collisions_stay_separate_and_repeats_raise():
    df = frame(rows("sidewalk_alpha", "CurbRamp", "Agree", 60, 60), rows("sidewalk_beta", "CurbRamp", "Agree", 60, 0))
    d, _ = loaded(df)
    strata, _ = g.compute_strata(d, g.Rule(), "x")
    assert strata[("sidewalk_alpha", "CurbRamp", "Agree", "unrecorded")]["status"] == "pass"
    assert strata[("sidewalk_beta", "CurbRamp", "Agree", "unrecorded")]["status"] == "fail"
    with pytest.raises(ValueError, match="not unique"):
        g.load_pull(frame(rows("sidewalk_alpha", "CurbRamp", "Agree", 2, 2), rows("sidewalk_alpha", "CurbRamp", "Agree", 1, 1)))


def test_drops_are_counted():
    df = frame(rows("sidewalk_alpha", "CurbRamp", "Agree", 3, 3), rows("sidewalk_alpha_bak_x", "CurbRamp", "Agree", 1, 1),
               rows("sidewalk_validation", "CurbRamp", "Agree", 1, 1))
    df.loc[0, "deleted"] = "t"
    df.loc[1, "model_id"] = "sidewalk-validator-ai-obstacle"
    d, c = g.load_pull(df)
    assert (c["deleted_rows"], c["backup_schema_rows"], c["internal_schema_rows"], c["model_type_mismatch"]) == (1, 1, 1, 1)
    assert len(d) == 1


def test_era_by_date_and_changeover_day():
    df = frame(rows("sidewalk_alpha", "CurbRamp", "Agree", 1, 1, date="2026-10-28"),
               rows("sidewalk_alpha", "CurbRamp", "Agree", 1, 1, date="2026-10-27", start=2),
               rows("sidewalk_alpha", "CurbRamp", "Agree", 1, 1, date="2026-10-26", start=3))
    d, c = g.load_pull(df)
    assert d.set_index("label_id")["era"].to_dict() == {1: "current", 2: "changeover", 3: "previous"}
    assert c["era_key"].startswith("date")


def test_era_by_revision_when_present():
    df = frame(rows("sidewalk_alpha", "CurbRamp", "Agree", 2, 2, date="2026-10-20"),
               rows("sidewalk_alpha", "CurbRamp", "Agree", 2, 2, date="2026-11-05", start=3))
    df["validator_revision"] = ["old", "old", "new", "new"]
    df.loc[0, "validator_revision"] = None
    d, c = g.load_pull(df)
    assert c["era_key"] == "validator_revision" and c["current_revisions"] == {"sidewalk-validator-ai-curbramp": "new"}
    assert d.set_index("label_id")["era"].to_dict() == {1: "previous", 2: "previous", 3: "current", 4: "current"}


def test_shadow_and_vote_regimes_are_never_pooled():
    df = frame(rows("sidewalk_alpha", "CurbRamp", "Agree", 60, 60), rows("sidewalk_alpha", "CurbRamp", "Agree", 60, 0, start=61))
    df["ai_vote_cast"] = ["t"] * 60 + ["f"] * 60
    d, _ = loaded(df)
    strata, _ = g.compute_strata(d, g.Rule(), "x")
    assert strata[("sidewalk_alpha", "CurbRamp", "Agree", "vote")]["status"] == "pass"
    assert strata[("sidewalk_alpha", "CurbRamp", "Agree", "shadow")]["status"] == "fail"


def test_no_gated_and_type_policies():
    df = frame(rows("sidewalk_alpha", "CurbRamp", "Disagree", 30, 30, acc=0.9199),
               rows("sidewalk_alpha", "NoSidewalk", "Disagree", 5, 5, start=100),
               rows("sidewalk_alpha", "Other", "Agree", 5, 5, start=200))
    d, _ = loaded(df)
    strata, _ = g.compute_strata(d, g.Rule(), "x")
    assert strata[("sidewalk_alpha", "CurbRamp", "Disagree", "unrecorded")]["status"] == "no_gated"
    assert strata[("sidewalk_alpha", "NoSidewalk", "Disagree", "unrecorded")]["status"] == "never_votes"
    assert strata[("sidewalk_alpha", "Other", "Agree", "unrecorded")]["status"] == "not_served"
    tb = g.types_block(d["label_type"].unique())
    assert tb["NoSidewalk"] == {"served": False, "verdicts": {"Agree": "eligible", "Disagree": "never"}}
    assert tb["Other"]["served"] is False and tb["CurbRamp"]["served"] is True


def test_base_rate_and_lift():
    # 60 Agree all right; 40 Disagree labels the humans call incorrect: base rate of "correct" = 60 / 100
    df = frame(rows("sidewalk_alpha", "CurbRamp", "Agree", 60, 60), rows("sidewalk_alpha", "CurbRamp", "Disagree", 40, 40, acc=0.5, start=61))
    d, _ = loaded(df)
    rec = g.compute_strata(d, g.Rule(), "x")[0][("sidewalk_alpha", "CurbRamp", "Agree", "unrecorded")]
    assert rec["base_rate"] == pytest.approx(0.6) and rec["lift"] == pytest.approx(0.4)


# --------------------------------------------------------------------------------------------------------------------
# naming, privacy and the report contract
# --------------------------------------------------------------------------------------------------------------------

def _report(df):
    d, counts = loaded(df)
    return g.build_report(d, counts, LISTING, g.Rule(), "pull@x", {"file": "p.csv", "sha256": "0" * 64,
                                                                   "ai_date_min": "", "ai_date_max": ""}, {})


def test_private_unlisted_and_ambiguous_never_named():
    df = frame(rows("sidewalk_alpha", "CurbRamp", "Agree", 60, 60),
               rows("sidewalk_hidden", "CurbRamp", "Agree", 60, 60), rows("sidewalk_unlisted", "CurbRamp", "Agree", 60, 60),
               rows("sidewalk_springfield", "CurbRamp", "Agree", 60, 60))
    public, private, strata = _report(df)
    text = json.dumps(public)
    assert set(public["deployments"]) == {"alpha-wa"}
    terms = g.private_terms(LISTING, df["schema"].unique())
    assert g.find_private_terms(text, terms) == []
    assert public["private"]["n_deployments_current_era"] == 3      # three private schemas: shown as a count
    assert public["private"]["cells"]["CurbRamp"]["Agree"] == {"n_deployments": 3, "strata_by_status": {"pass": 3},
                                                               "n_decided": 180}
    assert {r["schema"] for r in private["strata"]} == {"sidewalk_alpha", "sidewalk_hidden", "sidewalk_unlisted",
                                                        "sidewalk_springfield"}
    assert len(strata) == 4


def test_private_cells_suppressed_under_three():
    df = frame(rows("sidewalk_alpha", "CurbRamp", "Agree", 60, 60), rows("sidewalk_hidden", "CurbRamp", "Agree", 60, 60))
    public, _, _ = _report(df)
    assert public["private"]["n_deployments_current_era"] == "fewer than 3"
    assert public["private"]["cells"]["CurbRamp"]["Agree"]["n_decided"] == "suppressed"


def test_public_report_has_no_mode_and_validates():
    df = frame(rows("sidewalk_alpha", "CurbRamp", "Agree", 60, 60), rows("sidewalk_alpha", "CurbRamp", "Disagree", 5, 5, acc=0.5, start=100))
    public, private, _ = _report(df)
    g.validate_report(public)
    bad = json.loads(json.dumps(public))
    bad["deployments"]["alpha-wa"]["CurbRamp"]["Agree"]["unrecorded"]["mode"] = "vote"
    with pytest.raises(ValueError, match="mode"):
        g.validate_report(bad)
    bad = json.loads(json.dumps(public))
    bad["deployments"]["alpha-wa"]["CurbRamp"]["Agree"]["unrecorded"]["ci_lo"] = 0.5
    with pytest.raises(ValueError, match="bar"):
        g.validate_report(bad)
    # the private file carries modes under each policy
    assert {r["mode_strict"] for r in private["strata"]} <= set(g.MODES)
    assert private["decisions"]["strict"]["sidewalk_alpha"]["CurbRamp"] == {"Agree": "vote", "Disagree": "shadow"}


def test_policies():
    rec = {"status": "insufficient", "existing": True}
    assert g.recommended_mode(rec, "CurbRamp", "Agree", "strict") == "shadow"
    assert g.recommended_mode(rec, "CurbRamp", "Agree", "grandfather") == "vote"
    assert g.recommended_mode({"status": "fail", "existing": True}, "CurbRamp", "Agree", "grandfather") == "shadow"
    assert g.recommended_mode({"status": "insufficient", "existing": False}, "CurbRamp", "Agree", "grandfather") == "shadow"
    assert g.recommended_mode({"status": "pass"}, "NoSidewalk", "Agree", "strict") == "off"   # not served
    assert g.recommended_mode({"status": "pass"}, "NoSidewalk", "Disagree", "strict") == "off"


def test_expert_round_stratum_represents_nosidewalk_step_c():
    """The shape step (c) needs: masked-round evidence, a model revision, Agree-only, a stricter min n allowed."""
    r = g.Rule()
    df = frame(rows("sidewalk_alpha", "CurbRamp", "Agree", 60, 60))
    public, _, _ = _report(df)
    ok = g.expert_round_stratum(51, 51, r, "docs/round_b.json@abc1234", "3-of-5 fold ensemble", "NoSidewalk", "Agree")
    thin = g.expert_round_stratum(29, 29, r, "docs/round_b.json@abc1234", "3-of-5 fold ensemble", "NoSidewalk", "Agree")
    strict = g.expert_round_stratum(51, 51, r, "x", "y", "NoSidewalk", "Agree", min_n=100)
    weighted = g.expert_round_stratum(48.5, 51, r, "x", "y", "NoSidewalk", "Agree", basis="prior_weighted")
    assert (ok["status"], thin["status"], strict["status"]) == ("pass", "insufficient", "insufficient")
    assert weighted["basis"] == "prior_weighted"
    g.add_stratum(public, "beta-or", "NoSidewalk", "Agree", "expert_round", ok)
    g.add_stratum(public, "beta-or", "NoSidewalk", "Disagree", "expert_round",
                  g.expert_round_stratum(0, 0, r, "x", "y", "NoSidewalk", "Disagree"))
    g.validate_report(public)
    assert public["deployments"]["beta-or"]["NoSidewalk"]["Disagree"]["expert_round"]["status"] == "never_votes"
    assert public["types"]["NoSidewalk"]["served"] is False
    with pytest.raises(ValueError):
        g.expert_round_stratum(1, 1, r, "x", "y", "NoSidewalk", "Agree", basis="weird")


def test_find_private_terms_word_bounded_and_silent():
    terms = g.private_terms(LISTING, ["sidewalk_alpha", "sidewalk_hidden"], extra=["Secretville"])
    assert "sidewalk-hiddenburg.example.org" in terms and "atlantis" in terms and "hidden" in terms
    assert g.find_private_terms("alpha-wa and beta-or", terms) == []
    assert g.find_private_terms("HIDDENBURG", terms) and g.find_private_terms("in secretville", terms)
    assert g.find_private_terms("sidewalk_hidden", terms)
    assert g.find_private_terms("unhiddenly", terms) == []          # no substring hits inside words
    with pytest.raises(g.PrivacyError) as e:
        g.assert_public("see Hiddenburg", terms, "f.json")
    assert "iddenburg" not in str(e.value).lower()                  # the error never echoes the term


def test_write_public_json_refuses_private_text(tmp_path):
    terms = g.private_terms(LISTING, [])
    with pytest.raises(g.PrivacyError):
        g.write_public_json({"x": "hidden-x"}, str(tmp_path / "a.json"), terms)
    assert not (tmp_path / "a.json").exists()


def test_listing_failure_writes_nothing(tmp_path):
    df = frame(rows("sidewalk_alpha", "CurbRamp", "Agree", 60, 60))
    p = tmp_path / "pull.csv"
    df.to_csv(p, index=False)
    out = tmp_path / "public"
    rc = g.main(["gate", "--assessments", str(p), "--public-out", str(out)], fetch=fetch_fail)
    assert rc == 2 and not out.exists()


def test_gate_cli_end_to_end_and_acceptance(tmp_path):
    df = frame(rows("sidewalk_alpha", "Crosswalk", "Agree", 60, 60), rows("sidewalk_hidden", "CurbRamp", "Agree", 5, 5))
    p = tmp_path / "pull.csv"
    df.to_csv(p, index=False)
    key = tmp_path / "key.csv"
    pd.DataFrame({"label_uid": ["alpha-wa:1", "sidewalk_alpha:2"]}).to_csv(key, index=False)
    out, priv = tmp_path / "public", tmp_path / "private"
    args = ["gate", "--assessments", str(p), "--public-out", str(out), "--private-out", str(priv),
            "--exclude-items", str(key)]
    assert g.main(args + ["--expect", "60,5,0,0,0"], fetch=fetch_ok) == 0
    rep = json.loads((out / "deployment_status.json").read_text())
    assert rep["summary"]["acceptance_check"]["ok"]
    assert rep["exclusions"]["rows_removed"] == 2                        # API-form and schema-form keys both map
    assert rep["deployments"]["alpha-wa"]["Crosswalk"]["Agree"]["unrecorded"]["n"] == 58
    assert "hidden" not in (out / "deployment_status.json").read_text()
    assert (priv / "deployment_status_all.csv").exists()
    prov = json.loads((out / "provenance.json").read_text())
    assert str(tmp_path) not in json.dumps(prov)                         # basenames only
    assert g.main(args + ["--expect", "61,5,0,0,0"], fetch=fetch_ok) == 1


# --------------------------------------------------------------------------------------------------------------------
# back-test and power
# --------------------------------------------------------------------------------------------------------------------

def test_replay_graduates_at_expected_checkpoint():
    r = g.Rule()
    o = [0.0] * 3 + [1.0] * 297      # early errors, clean tail
    rep = g.replay_stratum(o, r)
    want = next(c for c in g.checkpoints(r, 300) if g.classify(sum(o[:c]), c, r) == "pass")
    assert rep["graduated_at"] == want == 200
    assert rep["withheld"][1] + rep["kept"][1] == 300
    never = g.replay_stratum([1.0, 0.0] * 50, r)
    assert never["graduated_at"] is None and never["withheld"][1] == 100 and never["kept"][1] == 0


def test_backtest_order_is_deterministic_and_counts_add_up():
    df = frame(rows("sidewalk_alpha", "CurbRamp", "Agree", 120, 120, date="2026-09-01"),
               rows("sidewalk_hidden", "CurbRamp", "Agree", 70, 60, date="2026-09-01"))
    d, _ = loaded(df.sample(frac=1, random_state=0))
    pub, priv = g.backtest(d, LISTING, g.Rule())
    p = pub["pooled"]["CurbRamp"]["Agree"]
    total = p["withheld_in_graduated"]["n"] + p["kept"]["n"] + p["never_graduated"]["n"] + p["strata_under_min_n"]["n"]
    assert total == 190
    d2, _ = loaded(df)
    assert g.backtest(d2, LISTING, g.Rule())[0] == pub
    assert "hidden" not in json.dumps(pub)


def test_power_bounds():
    r = g.Rule()
    allowed = [n - g.pass_threshold(n, r) for n in (50, 100, 200, 400, 800, 1600)]
    assert allowed == sorted(allowed)
    assert max(g.single_look_pass(0.92, n, r) for n in range(50, 400)) <= 0.0262
    assert g.single_look_pass(1.0, 50, r) == 1.0
    # repeated looks raise the false-pass rate; the cross-check against the review's 20,000-path simulation (0.10)
    assert g.ever_pass(0.92, list(range(50, 1001, 50)), r) == pytest.approx(0.102, abs=0.002)
    assert g.ever_pass(0.92, g.checkpoints(r, 2000), r) > g.single_look_pass(0.92, 1600, r)
    z = g.z_for(r, g.checkpoints(r, 2000))
    assert g.ever_pass(0.92, g.checkpoints(r, 2000), g.Rule(z=z)) <= 0.05


# --------------------------------------------------------------------------------------------------------------------
# slices
# --------------------------------------------------------------------------------------------------------------------

def test_join_tags_one_to_one_and_coverage(tmp_path):
    df = frame(rows("sidewalk_alpha", "CurbRamp", "Agree", 3, 3), rows("sidewalk_hidden", "CurbRamp", "Agree", 2, 2))
    d, _ = loaded(df)
    pd.DataFrame({"label_id": [1, 2], "tags": ['["narrow"]', "[]"]}).to_csv(tmp_path / "alpha-wa.csv", index=False)
    pd.DataFrame({"label_id": [1], "tags": ['["x"]']}).to_csv(tmp_path / "hidden-x.csv", index=False)
    j, cov = g.join_tags(d, str(tmp_path), g.listing_cities(LISTING))
    assert cov == {"rows": 5, "public_rows": 3, "exported": 2}               # private never joined
    s = g.confirmed_slices(j, "tag", "out_majority2", min_n=1)
    assert {r["slice"]: r["n"] for r in s} == {"narrow": 1, "(none)": 1, g.NOT_EXPORTED: 3}
    pd.DataFrame({"label_id": [1, 1], "tags": ["[]", "[]"]}).to_csv(tmp_path / "alpha-wa.csv", index=False)
    with pytest.raises(ValueError, match="not unique"):
        g.join_tags(d, str(tmp_path), g.listing_cities(LISTING))


def test_schema_city_map_one_to_one():
    with pytest.raises(ValueError):
        g.schema_city_map(["sidewalk_alpha", "sidewalk_alpha_wa"], {"alpha-wa": "public"})


# --------------------------------------------------------------------------------------------------------------------
# the committed doc
# --------------------------------------------------------------------------------------------------------------------

@pytest.mark.skipif(not os.path.exists(os.path.join(OUT, "deployment_status.json")), reason="not yet run")
def test_committed_doc_matches_json_and_report_validates():
    assert g.check_doc(DOC, OUT) == []
    with open(os.path.join(OUT, "deployment_status.json")) as f:
        rep = json.load(f)
    g.validate_report(rep)
    assert rep["summary"]["acceptance_check"]["ok"]
    assert rep["summary"]["acceptance_check"]["got"] == {"Crosswalk": 885, "CurbRamp": 834, "NoCurbRamp": 129,
                                                         "Obstacle": 199, "SurfaceProblem": 226}
