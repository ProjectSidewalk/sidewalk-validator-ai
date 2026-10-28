"""Masked expert re-label of the AI-placed labels the round-3 validator and a human disagreed about.

Follow-up to ``analysis/ai_labels_eval.py`` (``docs/ai-labels-2026-10.md``): on the unseen Vancouver AI labels the
validator called 89 wrong, and a single human vote had approved 57 of them. One vote cannot say who is right. This
builds a review list for a project lead to vote on the live site with the votes masked, in the design of
``analysis/ai_disagree_relabel.py`` (Gallery ``?labelIds=`` lists, masking CSS or console snippet, the lead's own
prod votes are the record), and scores the result against the validator and the original human votes.

    # 1. the packet (key stays private; the rater must not see it)
    RATER=<the rater's user_id> EXPERTS="<role>=<user_id>,..." python analysis/ai_labels_relabel.py build \\
        --run RUN --city vancouver-wa --out RUN/relabel
    # 2. the rater votes through RUN/relabel/index.html (one Gallery list, masked)
    # 3. a fresh rawLabels pull of the city into RUN/relabel/rawlabels_after/, then
    RATER=... python analysis/ai_labels_relabel.py score --run RUN --out RUN/relabel

**Cells.** Every unseen (not in a round-3 train/val split), scored AI label of the city falls in one of four cells by
the validator's verdict and the human majority; the packet takes all of the first three and a seeded sample of the
fourth so the rater cannot tell which labels are contested; labels the rater already voted on are left out (the
rater would be re-rating their own vote, and the API carries no vote timestamps, so a repeated vote is invisible):

- ``contested``: validator Disagree, human Agree (the question);
- ``confirmed``: validator Disagree, human Disagree;
- ``missed``: validator Agree, human Disagree;
- ``decoy``: validator Agree, human Agree (sampled).

The list is one seeded shuffle of all cells. The key (``private-key-do-not-share/key.csv``) records each label's cell,
the validator's probabilities and whether a lead or another account cast the original vote (``prior_voters``).

**Scoring** reads the rater's vote per label from a rawLabels pull taken after rating (``validations[].user_id``; the
public API carries them within minutes). ``cant_tell`` is Unsure. The report gives, per cell, how often the rater sided
with the validator and with the original human vote, Wilson intervals, and the validator's precision on its Disagrees
against the rater.
"""
import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data_generation as dg  # noqa: E402
from analysis.ai_labels_eval import load_run, read_labels, wilson  # noqa: E402

SEED = 20261028
CELLS = ("contested", "confirmed", "missed", "decoy")
# the masking CSS of analysis/ai_disagree_relabel.py (PR #66), copied so this branch does not depend on that one
GALLERY_MASK_CSS = """\
.card-validation-info, .validation-info-content { display: none !important; }
.label-detail__vote-count, .label-detail__vote-icon { display: none !important; }
.label-detail__admin-prev-validations, .label-detail__validator-comments, .label-detail__comments-section,
.label-detail__comments-count, .label-detail__admin-username { display: none !important; }
#ps-tooltip { display: none !important; }
.ai-icon { display: none !important; }
"""
VOTE_TO_VERDICT = {"Agree": "correct", "Disagree": "incorrect", "Unsure": "cant_tell"}


def seeded_rank(seed, label_uid):
    """A label's own seeded rank, ``sha256(f"{seed}:{label_uid}")``, so a sample is stable when the pool changes."""
    return hashlib.sha256(f"{seed}:{label_uid}".encode()).hexdigest()


def cell_of(verdict, true):
    """
    >>> [cell_of(v, t) for v, t in [("incorrect", "correct"), ("incorrect", "incorrect"), ("correct", "incorrect"),
    ...                              ("correct", "correct")]]
    ['contested', 'confirmed', 'missed', 'decoy']
    """
    if verdict == "incorrect":
        return "contested" if true == "correct" else "confirmed"
    return "missed" if true == "incorrect" else "decoy"


def pick(pool, seed=SEED, n_decoys=50):
    """The packet rows: every ``contested``, ``confirmed`` and ``missed`` label plus the ``n_decoys`` decoys with the
    lowest seeded rank, in one seeded order (``packet_id`` 1..n). ``pool`` needs ``label_uid, verdict, true``.

    >>> pool = pd.DataFrame({"label_uid": [f"c:{i}" for i in range(8)],
    ...                      "verdict": ["incorrect"] * 3 + ["correct"] * 5,
    ...                      "true": ["correct", "incorrect", "correct", "incorrect", "correct", "correct", "correct", "correct"]})
    >>> p = pick(pool, "s", n_decoys=2)
    >>> sorted(p.cell.value_counts().to_dict().items())
    [('confirmed', 1), ('contested', 2), ('decoy', 2), ('missed', 1)]
    >>> p.packet_id.tolist() == list(range(1, 7)) and pick(pool, "s", n_decoys=2).label_uid.tolist() == p.label_uid.tolist()
    True
    """
    df = pool.copy()
    df["cell"] = [cell_of(v, t) for v, t in zip(df.verdict, df["true"])]
    df["rank"] = df.label_uid.map(lambda u: seeded_rank(seed, u))
    decoys = df[df.cell == "decoy"].sort_values("rank").head(n_decoys)
    out = pd.concat([df[df.cell != "decoy"], decoys]).sort_values("rank").reset_index(drop=True)
    out["packet_id"] = np.arange(1, len(out) + 1)
    return out.drop(columns="rank")


def prior_votes(validations_json, rater_id):
    """``(rater's vote or "", voter ids)`` over the human Agree/Disagree/Unsure votes on a label.

    >>> v = json.dumps([{"user_id": "r", "validation": "Agree", "validator_type": "Human"},
    ...                 {"user_id": "x", "validation": "Disagree", "validator_type": "Human"},
    ...                 {"user_id": "ai", "validation": "Disagree", "validator_type": "AI"}])
    >>> prior_votes(v, "r"), prior_votes(v, "q"), prior_votes(None, "r")
    (('Agree', ['r', 'x']), ('', ['r', 'x']), ('', []))
    """
    if not isinstance(validations_json, str) or not validations_json:
        return "", []
    votes = [v for v in json.loads(validations_json) if v.get("validator_type") == "Human"]
    mine = [v["validation"] for v in votes if str(v.get("user_id")) == rater_id]
    return (mine[-1] if mine else ""), [str(v.get("user_id")) for v in votes]


def gallery_url(base, label_ids):
    """The Gallery page that shows exactly ``label_ids``, in order (SidewalkWebpage#5444; at most 500)."""
    if len(label_ids) > 500:
        raise ValueError("Gallery review lists hold at most 500 labels")
    return f"{base}/gallery?labelIds={','.join(str(i) for i in label_ids)}"


def mask_console_js(css=GALLERY_MASK_CSS):
    """The masking CSS as a one-line snippet for the browser console (re-run after every page load)."""
    flat = " ".join(css.split())
    return ("(()=>{const s=document.createElement('style');s.id='ps-mask';s.textContent=" + json.dumps(flat)
            + ";document.getElementById('ps-mask')?.remove();document.head.appendChild(s);"
            "console.log('votes masked');})();")


def index_html(city, url, n):
    import html as h
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Re-label list</title><style>
:root {{ color-scheme: light dark; --bg:#fff; --fg:#1f2328; --panel:#f6f8fa; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#0d1117; --fg:#e6edf3; --panel:#161b22; }} }}
body {{ background:var(--bg); color:var(--fg); font:15px/1.5 system-ui, sans-serif; margin:0 16px 48px; max-width:900px; }}
pre {{ background:var(--panel); padding:12px; overflow-x:auto; white-space:pre-wrap; }}
</style></head><body>
<h1>AI-placed curb ramps, masked re-label: {n} labels in {h.escape(city)}</h1>
<ol>
<li>Sign in on the site. <strong>Before opening the list</strong>, mask the votes: paste the console snippet below into
the developer console on the Gallery page (or add the CSS to Stylus for <code>*.cs.washington.edu</code>). Re-run the
snippet after every page load. Check once on a label whose votes you know: no counts, no vote icons, no tooltips.</li>
<li>For each label vote <strong>Agree</strong> (it is a curb ramp, placed on it), <strong>Disagree</strong> (not a curb
ramp, or placed off it) or <strong>Unsure</strong>. Your vote on the site is the record; nothing else to fill in.</li>
</ol>
<h2>Review list</h2>
<p><a href="{h.escape(url)}" target="_blank" rel="noopener">{h.escape(city)}</a> &middot; {n} labels</p>
<h2>Console snippet (JavaScript)</h2><pre>{h.escape(mask_console_js())}</pre>
<h2>Masking CSS (for Stylus)</h2><pre>{h.escape(GALLERY_MASK_CSS)}</pre>
</body></html>
"""


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def city_url(cities_json, city):
    c = json.load(open(cities_json))
    rows = [x for x in c["cities"] if x["city_id"] == city]
    if not rows:
        raise ValueError(f"{city} not in {cities_json}")
    return rows[0]["url"].rstrip("/")


def cmd_build(args):
    rater = os.environ.get("RATER", "")
    if not rater:
        sys.exit("RATER (the rater's user_id) must be set")
    experts = dg.parse_experts(os.environ.get("EXPERTS"))
    df = load_run(args.run)
    pool = df[(df.author == "ai") & df.scored & (df.clean == 1) & (df.city == args.city)].copy()
    before = args.rawlabels_before or os.path.join(args.run, "rawlabels", f"{args.city}.csv")
    raw = pd.read_csv(before, dtype=str, usecols=["label_id", "validations"])
    pool = pool.merge(raw, on="label_id", how="left", validate="one_to_one")
    pv = pool.validations.map(lambda v: prior_votes(v, rater))
    pool["prior_rater_vote"] = [a for a, _ in pv]
    pool["prior_voters"] = [" ".join("lead" if u in experts else "other" for u in b) for _, b in pv]
    # labels the rater already voted on are left out: the rater would be re-rating their own vote, and a repeated vote
    # is invisible in the API (no timestamps). Pull rawLabels right before building so recent votes count.
    n_rater = int((pool.prior_rater_vote != "").sum())
    pool = pool[pool.prior_rater_vote == ""]
    picked = pick(pool, args.seed, args.n_decoys)
    key = picked[["packet_id", "label_uid", "city", "label_id", "pano_id", "cell", "verdict", "true", "expert_class",
                  "p_correct", "p_incorrect", "confidence", "round3_split", "prior_rater_vote", "prior_voters",
                  "n_votes", "tier"]].copy()
    key["rater_id"] = rater
    priv = os.path.join(args.out, "private-key-do-not-share")
    os.makedirs(priv, exist_ok=True)
    kpath = os.path.join(priv, "key.csv")
    if os.path.exists(kpath) and not args.force:
        sys.exit(f"{kpath} exists; --force to rebuild")
    key.to_csv(kpath, index=False)
    base = city_url(os.path.join(args.run, "rawlabels", "cities.json"), args.city)
    url = gallery_url(base, key.label_id.tolist())
    with open(os.path.join(args.out, "index.html"), "w") as f:
        f.write(index_html(args.city, url, len(key)))
    with open(os.path.join(args.out, "gallery_url.txt"), "w") as f:
        f.write(url + "\n")
    with open(os.path.join(args.out, "gallery-mask.css"), "w") as f:
        f.write(GALLERY_MASK_CSS)
    man = {"built": datetime.now(timezone.utc).isoformat(timespec="seconds"), "city": args.city, "seed": args.seed,
           "n_decoys": args.n_decoys, "n": int(len(key)), "cells": key.cell.value_counts().to_dict(),
           "excluded_rater_voted": n_rater, "rawlabels_before": os.path.abspath(before),
           "key_sha256": sha256_file(kpath), "labels_sha256": sha256_file(os.path.join(args.run, "labels.csv")),
           "predictions_sha256": sha256_file(os.path.join(args.run, "predictions.csv")),
           "rawlabels_before_sha256": sha256_file(before),
           "css_sha256": hashlib.sha256(GALLERY_MASK_CSS.encode()).hexdigest(), "gallery_url": url}
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(man, f, indent=2)
    print(json.dumps({k: v for k, v in man.items() if k != "gallery_url"}, indent=2))
    print(f"\n{len(key)} labels -> {args.out}/index.html")


def score_rows(key, after, rater):
    """``key`` joined with the rater's vote from the after-rating pull (``verdict_rater``: correct / incorrect /
    cant_tell / "" when unvoted). ``new_vote`` is False when the vote equals a prior vote by the rater (then the API
    cannot tell a re-vote from no vote)."""
    av = after.set_index("label_id").validations
    votes = key.label_id.astype(str).map(lambda i: prior_votes(av.get(i), rater)[0])
    out = key.copy()
    out["rater_vote"] = votes
    out["verdict_rater"] = votes.map(VOTE_TO_VERDICT).fillna("")
    out["new_vote"] = (out.rater_vote != "") & (out.rater_vote != out.prior_rater_vote.fillna(""))
    return out


def _rate(k, n):
    lo, hi = wilson(k, n)
    return f"{k}/{n} = {k / n:.1%} [{lo:.0%}, {hi:.0%}]" if n else "—"


def score_markdown(rows):
    L = [f"<!-- analysis/ai_labels_relabel.py score, {datetime.now(timezone.utc).date()} -->", "",
         "| cell | n | voted | Unsure | rater sides with validator | rater sides with original human vote |",
         "|---|---:|---:|---:|---|---|"]
    voted = rows[rows.verdict_rater != ""]
    for cell in CELLS:
        g = rows[rows.cell == cell]
        v = voted[voted.cell == cell]
        dec = v[v.verdict_rater != "cant_tell"]
        L.append(f"| {cell} | {len(g)} | {len(v)} | {int((v.verdict_rater == 'cant_tell').sum())} | "
                 f"{_rate(int((dec.verdict_rater == dec.verdict).sum()), len(dec))} | "
                 f"{_rate(int((dec.verdict_rater == dec['true']).sum()), len(dec))} |")
    dis = voted[(voted.verdict == "incorrect") & (voted.verdict_rater != "cant_tell")]
    L += ["", f"Validator Disagree precision against the rater (contested + confirmed, Unsure excluded): "
              f"{_rate(int((dis.verdict_rater == 'incorrect').sum()), len(dis))}."]
    pr = voted[voted.prior_rater_vote.fillna("") != ""]
    if len(pr):
        L.append(f"{len(pr)} rated labels carried a prior vote by the rater; {int(pr.new_vote.sum())} changed. "
                 f"Unchanged ones cannot be told from unrated ones (no vote timestamps in the API).")
    return "\n".join(L) + "\n"


def cmd_score(args):
    rater = os.environ.get("RATER", "")
    if not rater:
        sys.exit("RATER must be set")
    key = pd.read_csv(os.path.join(args.out, "private-key-do-not-share", "key.csv"), dtype={"label_id": str})
    after_path = os.path.join(args.out, "rawlabels_after", f"{key.city.iloc[0]}.csv")
    after = pd.read_csv(after_path, dtype=str, usecols=["label_id", "validations"])
    rows = score_rows(key, after, rater)
    rows.to_csv(os.path.join(args.out, "private-key-do-not-share", "scored.csv"), index=False)
    md = score_markdown(rows)
    with open(os.path.join(args.out, "results.md"), "w") as f:
        f.write(md)
    man = json.load(open(os.path.join(args.out, "manifest.json")))
    man["scored"] = {"on": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                     "rawlabels_after_sha256": sha256_file(after_path),
                     "voted": int((rows.verdict_rater != "").sum()), "results_sha256": hashlib.sha256(md.encode()).hexdigest()}
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(man, f, indent=2)
    print(md)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--run", required=True)
    b.add_argument("--city", default="vancouver-wa")
    b.add_argument("--out", required=True)
    b.add_argument("--seed", default=str(SEED))
    b.add_argument("--n-decoys", type=int, default=50)
    b.add_argument("--rawlabels-before", help="a rawLabels pull taken just before building (default: the run's pull)")
    b.add_argument("--force", action="store_true")
    b.set_defaults(func=cmd_build)
    s = sub.add_parser("score")
    s.add_argument("--run", required=True)
    s.add_argument("--out", required=True)
    s.set_defaults(func=cmd_score)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
