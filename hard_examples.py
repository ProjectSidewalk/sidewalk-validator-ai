"""Build the public hard-examples list (``label_uid,reason``) from a production AI-vs-humans export.

A label is *hard* when the deployed validator got it wrong on strong evidence:

* ``unanimous_crowd``: the AI's verdict disagreed with the human majority (``kind != "ok"``), at least three humans
  voted Agree/Disagree, and none of them voted the other way (Unsure votes are ignored);
* ``expert``: one of the project leads' accounts (configured, not named) voted Agree or Disagree and that vote
  differs from the AI's. Their vote columns in the export are named by the ``EXPERTS`` environment variable that
  ``analysis/analyze.py`` uses (``EXPERTS="expert1=<user_id>,expert2=<user_id>"``; the keys are the column names), or
  by ``--experts expert1,expert2``.

A label meeting both is listed once, as ``expert``. Only ``label_uid`` and ``reason`` are written: no vote counts and
no names, so the output is safe to commit. ``data_generation.py --hard-examples`` sets ``hard = 1`` from it, and
``evaluate.py`` reports the ``hard`` slice of the test splits.

The input (``ai_vs_humans.csv``, columns ``label_uid, kind, h_agree, h_dis, ai`` and one per expert, ...) is a private
export produced by ``analysis/`` and is never committed.

Usage::

    EXPERTS="expert1=<user_id>,expert2=<user_id>" python hard_examples.py ~/Data/sidewalk-validator-ai-prod-2026-10-17/ai_vs_humans.csv \\
        analysis/hard_examples_2026-10-17.csv
"""
import argparse
import os

import pandas as pd

VOTES = ("Agree", "Disagree")


def expert_columns(arg=None):
    """Expert vote columns: ``--experts a,b`` if given, else the keys of the ``EXPERTS`` environment variable.

    >>> expert_columns("expert1, expert2")
    ['expert1', 'expert2']
    """
    if arg:
        return [c.strip() for c in arg.split(",") if c.strip()]
    return [kv.split("=", 1)[0] for kv in os.environ.get("EXPERTS", "").split(",") if "=" in kv]


def hard_examples(df, experts):
    """``label_uid,reason`` rows from an AI-vs-humans frame (see the module docstring for the rule); ``experts`` are
    the expert vote columns.

    >>> df = pd.DataFrame({"label_uid": ["a:1", "a:2", "a:3", "a:4"], "kind": ["false_agree", "ok", "ok", "false_disagree"],
    ...                    "h_agree": [0, 5, 2, 2], "h_dis": [3, 0, 1, 1], "expert1": [None, None, "Disagree", None],
    ...                    "expert2": [None, None, None, None], "ai": ["Agree", "Agree", "Agree", "Disagree"]})
    >>> hard_examples(df, ["expert1", "expert2"]).values.tolist()
    [['a:1', 'unanimous_crowd'], ['a:3', 'expert']]
    """
    unanimous = ((df["kind"] != "ok") & (df["h_agree"] + df["h_dis"] >= 3)
                 & ((df["h_agree"] == 0) | (df["h_dis"] == 0)))
    expert = pd.Series(False, index=df.index)
    for who in experts:
        expert |= df[who].isin(VOTES) & (df[who] != df["ai"])
    out = df.loc[unanimous | expert, ["label_uid"]].copy()
    out["reason"] = expert[unanimous | expert].map({True: "expert", False: "unanimous_crowd"})
    return out.sort_values("label_uid").reset_index(drop=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("ai_vs_humans", help="private ai_vs_humans.csv export")
    p.add_argument("out", help="output CSV (label_uid,reason)")
    p.add_argument("--experts", default=None, help="comma list of expert vote columns; default: keys of env EXPERTS")
    args = p.parse_args(argv)
    experts = expert_columns(args.experts)
    df = pd.read_csv(args.ai_vs_humans)
    assert df["label_uid"].is_unique
    missing = sorted(set(experts) - set(df.columns))
    if missing:
        p.error(f"expert columns {missing} are not in {args.ai_vs_humans}")
    if not experts:
        print("Warning: no expert columns configured (--experts or EXPERTS); only unanimous_crowd examples.")
    out = hard_examples(df, experts)
    out.to_csv(args.out, index=False)
    print(f"{len(out)} hard examples of {len(df)}: {out.reason.value_counts().to_dict()}")


if __name__ == "__main__":
    main()
