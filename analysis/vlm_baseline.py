"""Open-weight vision-language models as zero-shot label validators, against production (issue #16).

Asks current open-weight VLMs, run locally with MLX on Apple silicon, whether a crowdsourced Project Sidewalk label
is correct, on exactly the crops the production and retrained validators see (label-centred, tilt-corrected,
depth-sized, 640 px WebP), and compares them with the five production DINOv2 validators on the same crops. Also
builds the blinded explanation-rating packet the issue asks for.

**Score, not sample.** Each VLM gets one prompt per crop (``build_prompt``: a ~3-sentence definition of the label
type, "The labelled point is at the centre of the image.", "Is this a correct <type> label? Answer yes or no.").
The score is read from the next-token distribution after the prompt, never from sampled text:
``p_incorrect = P(no) / (P(yes) + P(no))``, where P(yes) sums the probability of every single-token spelling of yes
in ``YES_VARIANTS`` that the tokenizer has (likewise no), computed in log space (``yes_no_score``). ``answer_mass``
(P(yes) + P(no)) records how much of the distribution the two answers hold. ``score --check-greedy N`` also runs
greedy generation on the first N crops and records whether its first word agrees with the argmax of the score.

Subcommands (all outputs with label-level rows go OUTSIDE the repo, under ``~/Data/...``):

``score``
    One MLX model over the sample; writes ``<out-dir>/scores/<model>[__<variant>].csv`` (one row per crop:
    ``label_uid, model, variant, p_yes, p_no, answer_mass, log_odds_incorrect, p_incorrect, p_correct, top_token,
    prompt_tokens, latency_s``). Resumable: rows already in the CSV are skipped.
``baseline``
    The five production validators (the revisions sidewalk-ai-api's ``model_map.py`` pins, ``PRODUCTION``) on the
    same crops, preprocessed as ``evaluate.py`` does; writes ``<out-dir>/scores/production.csv``.
``report``
    Metrics per model x type, per split, per public city (n >= 30, private deployments pooled), on the ``hard``
    slice; a paired bootstrap of AUC (best VLM vs production, per type); latency and size; the prompt ablation; one
    ROC figure. Writes aggregate CSVs, ``report.md``, ``roc.png`` and ``provenance.json`` to ``--doc-dir``.
``explain``
    Picks the explanation items (``select_explanation_items``: public deployments only, balanced across types, half
    where the best VLM disagrees with the crowd, half where it agrees but production disagrees) and asks the best VLM
    "In at most 30 words, why?" as a second turn after its own answer; writes ``<out-dir>/explanations.csv``.
``packet``
    The blinded rater packet from ``explanations.csv``: shuffled items with random 6-hex ids, a blank coding sheet,
    and the key (written outside every checkout; its sha256 is the only part to commit). ``--seed`` has no default.
``agreement``
    Unblinds a filled sheet against the key and prints VLM-vs-rater and rater-vs-crowd agreement.

Install (on top of ``requirements.txt``; the production baseline needs the pinned transformers 4.55, and mlx-vlm
upgrades transformers, so run ``baseline`` from a second venv with ``requirements.txt`` only)::

    uv venv --python 3.12 .venv && source .venv/bin/activate && uv pip install -r requirements.txt mlx-vlm==0.7.10
    uv venv --python 3.12 .venv-prod && .venv-prod/bin/pip install -r requirements.txt   # (uv pip also fine)

Usage (``$S`` is the sample directory, ``$O`` the private output directory)::

    S=~/Data/sidewalk-validator-ai-retrain-2026-10/round3; O=~/Data/sidewalk-validator-ai-vlm-2026-10
    python analysis/vlm_baseline.py score --model qwen3vl-8b --sample-dir $S --out-dir $O --check-greedy 10
    python analysis/vlm_baseline.py score --model qwen3vl-8b --sample-dir $S --out-dir $O \\
        --label-types CurbRamp --variant no-definition                 # the prompt ablation
    .venv-prod/bin/python analysis/vlm_baseline.py baseline --sample-dir $S --out-dir $O --device mps
    python analysis/vlm_baseline.py report --sample-dir $S --out-dir $O --doc-dir docs/vlm-baseline-2026-10 \\
        --cities-json ~/Data/sidewalk-validator-ai-prod-2026-10-17/cities.json
    python analysis/vlm_baseline.py explain --model qwen3vl-8b --sample-dir $S --out-dir $O \\
        --cities-json ~/Data/sidewalk-validator-ai-prod-2026-10-17/cities.json --seed 20261021
    python analysis/vlm_baseline.py packet --out-dir $O --packet-dir $O/packet \\
        --key $O/private-key-do-not-share/packet_key.csv --seed $SECRET_SEED
    python analysis/vlm_baseline.py agreement $O/packet/coding_sheet_filled.csv \\
        --key $O/private-key-do-not-share/packet_key.csv --key-sha256 docs/vlm-baseline-2026-10/packet_key.sha256

Example (the score)::

    >>> import numpy as np
    >>> lp = np.log(np.array([0.6, 0.2, 0.1, 0.1]))      # vocab: yes, Yes, no, other
    >>> s = yes_no_score(lp, yes_ids=[0, 1], no_ids=[2])
    >>> round(s["p_incorrect"], 4), round(s["answer_mass"], 4)
    (0.1111, 0.9)
"""
import argparse
import datetime
import hashlib
import json
import os
import platform
import subprocess
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import evaluate  # noqa: E402
from calibration.per_city import load_cities, private_label, public_name  # noqa: E402

LABEL_TYPES = ("CurbRamp", "NoCurbRamp", "Obstacle", "SurfaceProblem", "Crosswalk")

# --------------------------------------------------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------------------------------------------------

MODELS = {
    "qwen3vl-2b": "mlx-community/Qwen3-VL-2B-Instruct-4bit",
    "qwen3vl-8b": "mlx-community/Qwen3-VL-8B-Instruct-4bit",
    "gemma4-12b": "mlx-community/gemma-4-12B-it-qat-4bit",
}
"""Short name -> mlx-community repo. Revisions are resolved from the local HF cache and recorded in provenance."""

PRODUCTION = {
    # Copied from sidewalk-ai-api sidewalk_ai_api/model_map.py ("validator_models", training_date 08-16-2025).
    "CurbRamp": ("projectsidewalk/sidewalk-validator-ai-curbramp", "969ad5a33079db7885c9e9a4eb5cd0f39d8fe57a"),
    "NoCurbRamp": ("projectsidewalk/sidewalk-validator-ai-nocurbramp", "9f3f3bb2f40864a468fb65e43644ee5ba54e3e19"),
    "Obstacle": ("projectsidewalk/sidewalk-validator-ai-obstacle", "8f0fe8fc48748fac35098e74c1848d015126deef"),
    "SurfaceProblem": ("projectsidewalk/sidewalk-validator-ai-surfaceproblem",
                       "0e1439daa5667c8a81f63e75f835bd51221df4c2"),
    "Crosswalk": ("projectsidewalk/sidewalk-validator-ai-crosswalk", "b7936e5bd17a4b96db301f954e5d3a5fe9eaf361"),
}
PRODUCTION_NAME = "production"

# --------------------------------------------------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------------------------------------------------

TYPE_NAMES = {"CurbRamp": "curb ramp", "NoCurbRamp": "missing curb ramp", "Obstacle": "obstacle",
              "SurfaceProblem": "surface problem", "Crosswalk": "crosswalk"}

DEFINITIONS = {
    # Condensed from the Project Sidewalk labeling guide's definitions (projectsidewalk.org, "Labeling Guide").
    "CurbRamp": ("A curb ramp is a short ramp cut into or built up to a curb so that people using wheelchairs, "
                 "walkers or strollers can move between the sidewalk and the street, often with yellow tactile "
                 "paving at the bottom. Driveways, alleys and flush building entrances are not curb ramps."),
    "NoCurbRamp": ("A missing curb ramp label marks a place where a curb ramp is needed but absent: a sidewalk "
                   "meets a street crossing or corner at a raised curb, with no ramp for wheelchair users. The "
                   "label is wrong if a curb ramp is actually present, or if no pedestrian route crosses there."),
    "Obstacle": ("An obstacle label marks an object in the sidewalk's pedestrian path that blocks it or narrows it "
                 "so a wheelchair user could not easily pass, such as a pole, sign, fire hydrant, tree, bin, "
                 "parked vehicle or overgrown vegetation. An object beside the path that leaves it clear is not "
                 "an obstacle."),
    "SurfaceProblem": ("A surface problem label marks damage to the sidewalk's walking surface that makes it hard "
                       "to roll or walk on, such as large cracks, uplifted or sunken slabs (often from tree roots), "
                       "bumps, missing or broken pavement, loose gravel or dirt, or grass growing over the "
                       "sidewalk."),
    "Crosswalk": ("A crosswalk label marks a designated place for pedestrians to cross a road, usually painted "
                  "stripes or parallel lines on the roadway surface, sometimes raised or textured paving. The "
                  "label is wrong if there is no marked crossing at the labelled point."),
}
CENTRE_SENTENCE = "The labelled point is at the centre of the image."
WHY_PROMPT = "In at most 30 words, why?"

YES_VARIANTS = ("yes", "Yes", "YES", " yes", " Yes", " YES")
NO_VARIANTS = ("no", "No", "NO", " no", " No", " NO")
VARIANTS = ("definition", "no-definition")


def build_prompt(label_type, variant="definition"):
    """The question for one label type. ``variant`` is ``definition`` (default) or ``no-definition`` (ablation).

    >>> build_prompt("Obstacle", "no-definition")
    'The labelled point is at the centre of the image. Is this a correct obstacle label? Answer yes or no.'
    """
    if label_type not in TYPE_NAMES:
        raise ValueError(f"unknown label type {label_type!r}")
    if variant not in VARIANTS:
        raise ValueError(f"unknown prompt variant {variant!r} (choose from {VARIANTS})")
    parts = [DEFINITIONS[label_type]] if variant == "definition" else []
    parts += [CENTRE_SENTENCE, f"Is this a correct {TYPE_NAMES[label_type]} label? Answer yes or no."]
    return " ".join(parts)


# --------------------------------------------------------------------------------------------------------------------
# Scoring logic -- pure, unit-tested
# --------------------------------------------------------------------------------------------------------------------

def answer_token_ids(encode, variants):
    """Token ids of every spelling in ``variants`` that encodes to exactly one token, deduplicated, with the spellings.

    ``encode`` maps a string to a list of ids (``tokenizer.encode(s, add_special_tokens=False)``). Multi-token
    spellings are dropped: only a single token can be read off the next-token distribution.

    >>> vocab = {"yes": [5], "Yes": [6], " yes": [5], "YES": [7, 8]}
    >>> answer_token_ids(lambda s: vocab.get(s, [0, 0]), ("yes", "Yes", " yes", "YES"))
    ([5, 6], ['yes', 'Yes', ' yes'])
    """
    ids, used = [], []
    for v in variants:
        enc = list(encode(v))
        if len(enc) == 1:
            used.append(v)
            if enc[0] not in ids:
                ids.append(int(enc[0]))
    return ids, used


def _logsumexp(x):
    x = np.asarray(x, dtype=np.float64)
    m = x.max()
    return float(m + np.log(np.exp(x - m).sum()))


def yes_no_score(logprobs, yes_ids, no_ids):
    """Score one next-token distribution (log-probabilities over the vocabulary).

    Returns ``p_yes``, ``p_no``, ``answer_mass`` (their sum), ``log_odds_incorrect`` (log P(no) - log P(yes)) and
    ``p_incorrect`` = P(no) / (P(yes) + P(no)), computed from the log-odds so it stays well resolved even when one
    answer holds almost all the mass. ``p_correct`` is ``1 - p_incorrect``.
    """
    lp = np.asarray(logprobs, dtype=np.float64)
    if not yes_ids or not no_ids:
        raise ValueError("need at least one yes and one no token id")
    ly, ln = _logsumexp(lp[list(yes_ids)]), _logsumexp(lp[list(no_ids)])
    log_odds = ln - ly
    p_incorrect = float(1.0 / (1.0 + np.exp(-log_odds)))
    return {"p_yes": float(np.exp(ly)), "p_no": float(np.exp(ln)), "answer_mass": float(np.exp(ly) + np.exp(ln)),
            "log_odds_incorrect": float(log_odds), "p_incorrect": p_incorrect, "p_correct": 1.0 - p_incorrect}


def parse_answer(text):
    """``yes``, ``no`` or ``None`` from the first word of a generated answer.

    >>> parse_answer(" Yes, it is a curb ramp."), parse_answer("no."), parse_answer("The image shows")
    ('yes', 'no', None)
    """
    words = str(text).strip().split()
    if not words:
        return None
    w = words[0].strip(".,!:;\"'*()").lower()
    return w if w in ("yes", "no") else None


def greedy_agrees(text, p_incorrect):
    """Whether greedy text agrees with the score's argmax (``no`` iff ``p_incorrect > 0.5``); None if unparsable.

    >>> greedy_agrees("No", 0.9), greedy_agrees("yes", 0.9), greedy_agrees("maybe", 0.9) is None
    (True, False, True)
    """
    ans = parse_answer(text)
    if ans is None:
        return None
    return (ans == "no") == (p_incorrect > 0.5)


def verdict_from_score(p_incorrect):
    """The model's verdict at the argmax, with ``evaluate``'s tie rule (incorrect only when strictly above 0.5)."""
    return np.where(np.asarray(p_incorrect, dtype=float) > 0.5, "incorrect", "correct")


# --------------------------------------------------------------------------------------------------------------------
# Input
# --------------------------------------------------------------------------------------------------------------------

def load_sample(sample_dir):
    """``sample.csv`` with an absolute ``image`` column (``<dir>/crops/<label_type>/<crop_path>``); unique uids."""
    df = pd.read_csv(os.path.join(sample_dir, "sample.csv"), dtype={"label_id": str})
    if df.label_uid.duplicated().any():
        raise ValueError("sample.csv has duplicate label_uid")
    df["image"] = [os.path.join(sample_dir, "crops", t, p) for t, p in zip(df.label_type, df.crop_path)]
    return df


SUBSET_SHARES = {("test", "correct"): 0.3, ("test", "incorrect"): 0.3, ("test_heldout", "correct"): 0.2,
                 ("test_heldout", "incorrect"): 0.2}
"""Per-type composition of ``balanced_subset``: both classes balanced, both splits kept (60/40 test/held-out)."""


def balanced_subset(sample, per_type=200, seed=20261021):
    """A class-balanced subset with both splits, ``per_type`` rows per label type (``SUBSET_SHARES``), sampled
    with ``seed`` after sorting by ``label_uid`` so it depends only on the data. Used when scoring every crop with a
    large model would not fit the time budget.

    >>> s = pd.DataFrame({"label_uid": [f"c:{i}" for i in range(40)], "label_type": "CurbRamp",
    ...                   "split": ["test"] * 20 + ["test_heldout"] * 20, "label_class": ["correct", "incorrect"] * 20})
    >>> balanced_subset(s, per_type=10).groupby(["split", "label_class"]).size().tolist()
    [3, 3, 2, 2]
    """
    rng = np.random.default_rng(seed)
    out = []
    for (t, split, cls), g in sample.sort_values("label_uid").groupby(["label_type", "split", "label_class"]):
        k = int(round(per_type * SUBSET_SHARES[(split, cls)]))
        if len(g) < k:
            raise ValueError(f"{t} {split} {cls}: only {len(g)} rows for {k}")
        out.append(g.iloc[np.sort(rng.permutation(len(g))[:k])])
    return pd.concat(out).sort_values(["label_type", "label_uid"])


def score_path(out_dir, model, variant="definition"):
    suffix = "" if variant == "definition" else f"__{variant}"
    return os.path.join(out_dir, "scores", f"{model}{suffix}.csv")


# --------------------------------------------------------------------------------------------------------------------
# MLX inference -- needs mlx-vlm; not covered by the CPU tests
# --------------------------------------------------------------------------------------------------------------------

class MlxVlm:
    """One mlx-vlm model with the yes/no scorer and a second-turn explanation. Batch size 1."""

    def __init__(self, repo):
        from mlx_vlm import load

        self.repo = repo
        self.model, self.processor = load(repo)
        self.tokenizer = getattr(self.processor, "tokenizer", self.processor)
        enc = lambda s: self.tokenizer.encode(s, add_special_tokens=False)  # noqa: E731
        self.yes_ids, self.yes_used = answer_token_ids(enc, YES_VARIANTS)
        self.no_ids, self.no_used = answer_token_ids(enc, NO_VARIANTS)

    def chat(self, turns):
        """Chat-template text for ``turns`` (``[(role, text), ...]``); the image goes in the FIRST user turn."""
        messages = []
        for i, (role, text) in enumerate(turns):
            content = [{"type": "text", "text": text}]
            if i == 0:
                content = [{"type": "image"}] + content
            messages.append({"role": role, "content": content})
        kwargs = {}
        if "gemma" in self.repo.lower() or "qwen3" in self.repo.lower():
            kwargs["enable_thinking"] = False   # instruct models; a no-op where the template has no switch
        return self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, **kwargs)

    def score(self, image, question):
        import mlx.core as mx
        from mlx_vlm import stream_generate

        prompt = self.chat([("user", question)])
        t0 = time.perf_counter()
        r = next(iter(stream_generate(self.model, self.processor, prompt, image=[image], max_tokens=1,
                                      temperature=0.0)))
        lp = np.array(r.logprobs.astype(mx.float32)).reshape(-1)
        latency = time.perf_counter() - t0
        out = yes_no_score(lp, self.yes_ids, self.no_ids)
        out.update(top_token=self.tokenizer.decode([int(lp.argmax())]), prompt_tokens=int(r.prompt_tokens),
                   latency_s=latency)
        return out

    def generate(self, image, turns, max_tokens):
        from mlx_vlm import stream_generate

        prompt = self.chat(turns)
        t0 = time.perf_counter()
        text = "".join(r.text for r in stream_generate(self.model, self.processor, prompt, image=[image],
                                                         max_tokens=max_tokens, temperature=0.0))
        return text.strip(), time.perf_counter() - t0


def run_score(args):
    import mlx.core as mx

    sample = load_sample(args.sample_dir)
    if args.subset:   # before the type filter, so a one-type run scores the same crops as the full run
        sample = balanced_subset(sample, args.subset)
    if args.label_types:
        sample = sample[sample.label_type.isin(args.label_types.split(","))]
    if args.limit:
        # balanced pilot: the first --limit rows per type (sample.csv order), both splits and classes represented
        sample = (sample.sort_values(["label_type", "label_class", "label_uid"])
                  .groupby(["label_type", "label_class"]).head(max(1, args.limit // 2)))
    path = score_path(args.out_dir, args.model, args.variant)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    done = set(pd.read_csv(path).label_uid) if os.path.exists(path) else set()
    todo = sample[~sample.label_uid.isin(done)]
    print(f"{args.model} ({MODELS[args.model]}), variant {args.variant}: {len(todo)} to score, {len(done)} done")
    vlm = MlxVlm(MODELS[args.model])
    print(f"yes tokens {vlm.yes_used} -> {vlm.yes_ids}; no tokens {vlm.no_used} -> {vlm.no_ids}")
    meta = {"model": args.model, "repo": MODELS[args.model], "variant": args.variant,
            "yes_tokens": vlm.yes_used, "yes_ids": vlm.yes_ids, "no_tokens": vlm.no_used, "no_ids": vlm.no_ids}
    with open(path.replace(".csv", ".meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    rows, t_start = [], time.time()
    checks = []
    for i, r in enumerate(todo.itertuples()):
        s = vlm.score(r.image, build_prompt(r.label_type, args.variant))
        rows.append({"label_uid": r.label_uid, "model": args.model, "variant": args.variant, **s})
        if len(checks) < args.check_greedy and not done:
            text, _ = vlm.generate(r.image, [("user", build_prompt(r.label_type, args.variant))], max_tokens=4)
            checks.append({"label_uid": r.label_uid, "greedy_text": text, "p_incorrect": s["p_incorrect"],
                           "top_token": s["top_token"], "agrees": greedy_agrees(text, s["p_incorrect"])})
        if len(rows) >= 25 or i == len(todo) - 1:
            pd.DataFrame(rows).to_csv(path, mode="a", header=not os.path.exists(path), index=False)
            rows = []
            el = time.time() - t_start
            print(f"  {i + 1}/{len(todo)}  {el:.0f}s  peak {mx.get_peak_memory() / 1e9:.1f} GB", flush=True)
    if checks:
        c = pd.DataFrame(checks)
        c.to_csv(path.replace(".csv", ".greedy_check.csv"), index=False)
        print(f"greedy check: {int((c.agrees == True).sum())}/{len(c)} agree "  # noqa: E712
              f"({int(c.agrees.isna().sum())} unparsable)")
    meta["peak_memory_gb"] = mx.get_peak_memory() / 1e9
    with open(path.replace(".csv", ".meta.json"), "w") as f:
        json.dump(meta, f, indent=1)


# --------------------------------------------------------------------------------------------------------------------
# Production baseline -- needs torch + transformers 4.55 (requirements.txt)
# --------------------------------------------------------------------------------------------------------------------

def run_baseline(args):
    import torch
    import transformers
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    sample = load_sample(args.sample_dir)
    path = score_path(args.out_dir, PRODUCTION_NAME)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rows = []
    for label_type, g in sample.groupby("label_type"):
        repo, rev = PRODUCTION[label_type]
        proc = AutoImageProcessor.from_pretrained(repo, revision=rev, use_fast=False)
        model = AutoModelForImageClassification.from_pretrained(repo, revision=rev).to(args.device).eval()
        label2id = {v: int(k) for k, v in model.config.id2label.items()}
        if set(label2id) != set(evaluate.CLASSES):
            raise ValueError(f"unexpected id2label {model.config.id2label}")
        for i, r in enumerate(g.itertuples()):
            t0 = time.perf_counter()
            inputs = proc(Image.open(r.image).convert("RGB"), return_tensors="pt").to(args.device)
            with torch.no_grad():
                p = model(**inputs).logits.softmax(-1).float().cpu().numpy()[0]
            latency = time.perf_counter() - t0
            if i == 0:
                latency = float("nan")   # first call includes kernel compilation; excluded from latency stats
            rows.append({"label_uid": r.label_uid, "model": PRODUCTION_NAME, "variant": "definition",
                         "p_correct": float(p[label2id["correct"]]), "p_incorrect": float(p[label2id["incorrect"]]),
                         "latency_s": latency})
        print(f"{label_type}: {len(g)} crops")
        del model
    pd.DataFrame(rows).to_csv(path, index=False)
    meta = {"model": PRODUCTION_NAME, "repos": PRODUCTION, "device": args.device, "torch": torch.__version__,
            "transformers": transformers.__version__}
    with open(path.replace(".csv", ".meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"wrote {path}")


# --------------------------------------------------------------------------------------------------------------------
# Report helpers -- pure, unit-tested
# --------------------------------------------------------------------------------------------------------------------

REPORT_LEVELS = (0.92, 0.95)


def predictions(sample, scores):
    """Join per-crop scores (``label_uid, model, variant, p_correct, p_incorrect, ...``) to the sample; one row per
    (model, variant, label). ``true`` is ``label_class``, as ``evaluate.slice_metrics`` expects."""
    keep = ["label_uid", "city", "label_type", "label_class", "split", "hard"]
    out = scores.merge(sample[keep], on="label_uid", how="inner", validate="many_to_one")
    if out.duplicated(["model", "variant", "label_uid"]).any():
        raise ValueError("a score file has duplicate rows for one label")
    return out.assign(true=out.label_class)


def metrics_row(df, levels=REPORT_LEVELS):
    """The flat row the tables print: n, AUC, balanced accuracy at the argmax, recall at precision for both classes.

    ``r_incorrect_p95`` is the recall of ``incorrect`` at precision >= 0.95 (``evaluate.recall_at_precision``).
    """
    m = evaluate.slice_metrics(df, levels)
    row = {"n": m["n"], "n_incorrect": m["n_by_class"]["incorrect"], "auc": m.get("auc"),
           "balanced_accuracy": m.get("balanced_accuracy"), "accuracy": m.get("accuracy")}
    for c in evaluate.CLASSES:
        for p in levels:
            rp = m.get("recall_at_precision", {}).get(c, {}).get(f"{p:.2f}")
            row[f"r_{c}_p{int(round(p * 100))}"] = rp["recall"] if rp else None
    return row


def metrics_table(pred, by, levels=REPORT_LEVELS, min_n=1):
    """``metrics_row`` per group of ``by`` (groups with fewer than ``min_n`` rows are skipped)."""
    rows = []
    for key, g in pred.groupby(list(by), sort=True):
        if len(g) < min_n:
            continue
        key = key if isinstance(key, tuple) else (key,)
        rows.append({**dict(zip(by, key)), **metrics_row(g, levels)})
    return pd.DataFrame(rows)


def display_city(cities_series, cities):
    """Public ``city_id`` or the pooled private label, per row. The count is of distinct private deployments.

    >>> display_city(pd.Series(["seattle-wa", "secret-a", "secret-b"]), {"seattle-wa": "public"}).tolist()
    ['seattle-wa', 'other (2 private deployments)', 'other (2 private deployments)']
    """
    names = {c: public_name(c, cities) for c in cities_series.unique()}
    n_private = sum(v is None for v in names.values())
    return cities_series.map(lambda c: names[c] if names[c] is not None else private_label(n_private))


def is_public(cities_series, cities):
    """Boolean mask: the row's deployment is listed as public."""
    names = {c: public_name(c, cities) for c in cities_series.unique()}
    return cities_series.map(lambda c: names[c] is not None).astype(bool)


def per_city_table(pred, cities, min_n=30, levels=REPORT_LEVELS):
    """Per model x displayed city (all types pooled): public cities with >= ``min_n`` rows, one pooled private row.

    Public cities below ``min_n`` are left out (their count is in ``n_cities_below_min``); the private row is kept
    at any size, because it is the only place private rows appear.
    """
    p = pred.assign(display_city=display_city(pred.city, cities).values,
                    public=is_public(pred.city, cities).values)
    out = []
    for model, g in p.groupby("model"):
        sizes = g.groupby("display_city").size()
        for city, n in sizes.items():
            pub = bool(g[g.display_city == city].public.iloc[0])
            if pub and n < min_n:
                continue
            out.append({"model": model, "city": city, "public": pub,
                        **metrics_row(g[g.display_city == city], levels)})
    t = pd.DataFrame(out)
    return t.sort_values(["model", "public", "n"], ascending=[True, False, False]).reset_index(drop=True)


def paired_bootstrap_auc(y, score_a, score_b, n_boot=1000, seed=0):
    """AUC of two scores on the same labels, and a paired, class-stratified bootstrap of each and their difference.

    ``y`` is 1 for the positive class (``incorrect``). Resampling is by label, within class, so every resample has
    both classes. Returns ``{"auc_a", "auc_b", "diff", "a_lo", "a_hi", "b_lo", "b_hi", "diff_lo", "diff_hi"}``
    (95% percentile intervals).
    """
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y).astype(int)
    a, b = np.asarray(score_a, dtype=float), np.asarray(score_b, dtype=float)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    rng = np.random.default_rng(seed)
    boots = np.empty((n_boot, 2))
    for i in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        boots[i] = roc_auc_score(y[idx], a[idx]), roc_auc_score(y[idx], b[idx])
    d = boots[:, 0] - boots[:, 1]
    q = lambda x: np.percentile(x, [2.5, 97.5])  # noqa: E731
    auc_a, auc_b = roc_auc_score(y, a), roc_auc_score(y, b)
    (alo, ahi), (blo, bhi), (dlo, dhi) = q(boots[:, 0]), q(boots[:, 1]), q(d)
    return {"auc_a": auc_a, "auc_b": auc_b, "diff": auc_a - auc_b, "a_lo": alo, "a_hi": ahi, "b_lo": blo,
            "b_hi": bhi, "diff_lo": dlo, "diff_hi": dhi}


def latency_table(scores):
    """Median and p90 seconds per crop by model (NaN latencies, such as a warm-up call, are ignored)."""
    rows = []
    for (model, variant), g in scores.groupby(["model", "variant"]):
        lat = g.latency_s.dropna()
        rows.append({"model": model, "variant": variant, "n": len(g), "latency_median_s": float(lat.median()),
                     "latency_p90_s": float(lat.quantile(0.9)),
                     "answer_mass_median": float(g.answer_mass.median()) if "answer_mass" in g else None})
    return pd.DataFrame(rows)


def best_model(headline, exclude=(PRODUCTION_NAME,)):
    """The VLM with the highest mean per-type AUC in a headline table (``model, label_type, auc``)."""
    h = headline[~headline.model.isin(exclude)]
    return h.groupby("model").auc.mean().idxmax()


def markdown(df, floatfmt="{:.3f}"):
    """A pipe table (no tabulate dependency)."""
    cols = list(df.columns)
    fmt = lambda v: ("" if v is None or (isinstance(v, float) and np.isnan(v))  # noqa: E731
                     else floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v))
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(fmt(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(lines)


# --------------------------------------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------------------------------------

def sha256_of(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _git(cwd, *args):
    try:
        out = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def hf_snapshot(repo):
    """``(revision, size_bytes)`` of a repo's local HF cache snapshot, or ``(None, None)``."""
    try:
        from huggingface_hub import scan_cache_dir
    except ImportError:
        return None, None
    for r in scan_cache_dir().repos:
        if r.repo_id == repo and r.repo_type == "model":
            revs = sorted(r.revisions, key=lambda x: x.last_modified)
            return (revs[-1].commit_hash, int(revs[-1].size_on_disk)) if revs else (None, None)
    return None, None


def roc_figure(pred, models, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    # Reference categorical order (blue, orange, aqua, yellow); production in primary ink, solid.
    colors = {PRODUCTION_NAME: "#0b0b0b"}
    palette = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
    for i, m in enumerate([m for m in models if m != PRODUCTION_NAME]):
        colors[m] = palette[i % len(palette)]
    fig, axes = plt.subplots(1, len(LABEL_TYPES), figsize=(15, 3.4), sharey=True, facecolor="#fcfcfb")
    for ax, t in zip(axes, LABEL_TYPES):
        for m in models:
            g = pred[(pred.label_type == t) & (pred.model == m)]
            if g.empty:
                continue
            fpr, tpr, _ = roc_curve(g.true == "incorrect", g.p_incorrect)
            auc = evaluate.slice_metrics(g)["auc"]
            ax.plot(fpr, tpr, color=colors[m], lw=2, ls="-" if m == PRODUCTION_NAME else "--",
                    label=f"{m} (AUC {auc:.2f})")
        ax.plot([0, 1], [0, 1], color="#c3c2b7", lw=0.8)
        ax.set_facecolor("#fcfcfb")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_title(t, fontsize=10)
        ax.set_xlabel("false positive rate", fontsize=8)
        ax.legend(fontsize=7, loc="lower right", frameon=False)
        ax.tick_params(labelsize=7)
        ax.set_aspect("equal")
    axes[0].set_ylabel("true positive rate (incorrect)", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def run_report(args):
    sample = load_sample(args.sample_dir)
    cities = load_cities(args.cities_json)
    score_dir = os.path.join(args.out_dir, "scores")
    files = sorted(f for f in os.listdir(score_dir) if f.endswith(".csv") and ".greedy_check" not in f)
    scores = pd.concat([pd.read_csv(os.path.join(score_dir, f)) for f in files], ignore_index=True)
    pred = predictions(sample, scores)
    main_all = pred[pred.variant == "definition"]
    subset = set(balanced_subset(sample, args.subset).label_uid) if args.subset else set(sample.label_uid)
    complete = [m for m, g in main_all.groupby("model") if subset <= set(g.label_uid)]
    main = main_all[main_all.model.isin(complete) & main_all.label_uid.isin(subset)]
    full = [m for m, g in main_all.groupby("model") if g.label_uid.nunique() == len(sample)]
    os.makedirs(args.doc_dir, exist_ok=True)
    w = lambda df, name: df.to_csv(os.path.join(args.doc_dir, name), index=False, float_format="%.4f")  # noqa: E731

    headline = metrics_table(main, ["model", "label_type"])
    pooled = metrics_table(main, ["model"]).assign(label_type="all (pooled)")
    w(pd.concat([headline, pooled], ignore_index=True), "headline.csv")
    by_split = metrics_table(main, ["model", "label_type", "split"])
    by_split_pooled = metrics_table(main, ["model", "split"]).assign(label_type="all (pooled)")
    w(pd.concat([by_split, by_split_pooled], ignore_index=True), "by_split.csv")
    hard = metrics_table(main[main.hard == 1], ["model"]).assign(slice="hard == 1, all types, subset")
    full_all = main_all[main_all.model.isin(full)]
    if len(full_all):
        full_tab = pd.concat([metrics_table(full_all, ["model", "label_type"]),
                              metrics_table(full_all, ["model"]).assign(label_type="all (pooled)")])
        hard = pd.concat([hard, metrics_table(full_all[full_all.hard == 1], ["model"])
                          .assign(slice="hard == 1, all types, full sample")], ignore_index=True)
    else:
        full_tab = pd.DataFrame()
    w(full_tab, "full_sample.csv")
    w(hard, "hard.csv")
    city = per_city_table(main, cities, min_n=args.min_city_n)
    w(city, "per_city.csv")
    on_subset = (scores.variant == "definition") & scores.model.isin(complete) & scores.label_uid.isin(subset)
    lat = latency_table(scores[on_subset | (scores.variant != "definition")])
    sizes = {}
    for m in lat.model.unique():
        if m == PRODUCTION_NAME:
            sz = [hf_snapshot(PRODUCTION[t][0])[1] for t in LABEL_TYPES]
            sizes[m] = sum(s for s in sz if s) / len(LABEL_TYPES) if all(sz) else None
        else:
            sizes[m] = hf_snapshot(MODELS[m])[1]
    lat["size_gb"] = lat.model.map(lambda m: sizes.get(m) / 1e9 if sizes.get(m) else None)
    w(lat, "latency.csv")

    best = best_model(headline)
    boot = []
    for t in LABEL_TYPES:
        a = main[(main.model == best) & (main.label_type == t)].set_index("label_uid")
        b = main[(main.model == PRODUCTION_NAME) & (main.label_type == t)].set_index("label_uid")
        if a.empty or b.empty:
            continue
        b = b.loc[a.index]
        r = paired_bootstrap_auc(a.true == "incorrect", a.p_incorrect, b.p_incorrect, args.n_boot, args.seed)
        boot.append({"label_type": t, "model_a": best, "model_b": PRODUCTION_NAME, "n": len(a), **r})
    boot = pd.DataFrame(boot)
    w(boot, "bootstrap_auc.csv")

    abl = pred[pred.label_uid.isin(pred[pred.variant == "no-definition"].label_uid)]
    abl_models = abl[abl.variant == "no-definition"].model.unique()
    abl = abl[abl.model.isin(abl_models) & (abl.model != PRODUCTION_NAME)]
    ablation = metrics_table(abl, ["model", "label_type", "variant"]) if len(abl) else pd.DataFrame()
    w(ablation, "ablation.csv")

    greedy = []
    for f in sorted(os.listdir(score_dir)):
        if f.endswith(".greedy_check.csv"):
            c = pd.read_csv(f"{score_dir}/{f}")
            greedy.append({"file": f.replace(".greedy_check.csv", ""), "n": len(c),
                           "agree": int((c.agrees == True).sum()),  # noqa: E712
                           "unparsable": int(c.agrees.isna().sum())})
    greedy = pd.DataFrame(greedy)
    w(greedy, "greedy_check.csv")

    models = [PRODUCTION_NAME] + [m for m in MODELS if m in complete]
    roc_figure(main, models, os.path.join(args.doc_dir, "roc.png"))

    cols = ["model", "label_type", "n", "auc", "balanced_accuracy", "r_incorrect_p95", "r_incorrect_p92",
            "r_correct_p95", "r_correct_p92"]
    with open(os.path.join(args.doc_dir, "report.md"), "w") as f:
        f.write(f"# VLM baseline report (generated {datetime.date.today()})\n\nbest VLM by mean per-type AUC: "
                f"{best}\n\nheadline set: {len(subset)} crops (balanced_subset per_type={args.subset}); "
                f"models on it: {', '.join(complete)}\n\n## headline\n\n{markdown(pd.concat([headline, pooled])[cols])}\n\n"
                f"## by split\n\n{markdown(pd.concat([by_split, by_split_pooled])[cols[:2] + ['split'] + cols[2:]])}"
                f"\n\n## hard\n\n{markdown(hard[['slice', 'model', 'n', 'n_incorrect', 'auc', 'balanced_accuracy']])}\n\n"
                f"## full sample (models that scored all {len(sample)} crops)\n\n"
                f"{markdown(full_tab[cols]) if len(full_tab) else '(none)'}\n\n"
                f"## per city\n\n{markdown(city[['model', 'city', 'n', 'auc', 'balanced_accuracy']])}\n\n"
                f"## bootstrap\n\n{markdown(boot)}\n\n## latency\n\n{markdown(lat)}\n\n"
                f"## ablation\n\n{markdown(ablation) if len(ablation) else '(none)'}\n\n"
                f"## greedy check\n\n{markdown(greedy) if len(greedy) else '(none)'}\n")
    write_provenance(args, sample, files, score_dir, complete)
    print(open(os.path.join(args.doc_dir, "report.md")).read())


def write_provenance(args, sample, files, score_dir, complete):
    import matplotlib
    import sklearn

    here = os.path.dirname(os.path.abspath(__file__))
    metas = {}
    for f in files:
        mp = os.path.join(score_dir, f.replace(".csv", ".meta.json"))
        if os.path.exists(mp):
            metas[f] = json.load(open(mp))
    models = {}
    for m in complete:
        if m == PRODUCTION_NAME:
            models[m] = {t: {"repo": r, "revision": rev, "size_bytes": hf_snapshot(r)[1]}
                         for t, (r, rev) in PRODUCTION.items()}
        else:
            rev, size = hf_snapshot(MODELS[m])
            models[m] = {"repo": MODELS[m], "revision": rev, "size_bytes": size, "quantisation": "4-bit (MLX)"}
    prov = {
        "date": datetime.date.today().isoformat(),
        "git_commit": _git(here, "rev-parse", "HEAD"),
        "git_dirty": bool(_git(here, "status", "--porcelain", "--untracked-files=no")),
        "hardware": f"{platform.machine()} {_sysctl('machdep.cpu.brand_string')}, "
                    f"{int(_sysctl('hw.memsize') or 0) / 2**30:.0f} GB, macOS {platform.mac_ver()[0]}",
        "python": platform.python_version(),
        "packages": {"numpy": np.__version__, "pandas": pd.__version__, "scikit-learn": sklearn.__version__,
                     "matplotlib": matplotlib.__version__, **_mlx_versions()},
        "inputs": {"sample.csv": {"sha256": sha256_of(os.path.join(args.sample_dir, "sample.csv")),
                                  "rows": int(len(sample))},
                   "cities.json": {"sha256": sha256_of(args.cities_json)}},
        "scores": {f: {"sha256": sha256_of(os.path.join(score_dir, f)),
                       "rows": int(len(pd.read_csv(os.path.join(score_dir, f)))), "meta": metas.get(f)}
                   for f in files},
        "models": models,
        "prompts": {t: build_prompt(t) for t in LABEL_TYPES},
        "outputs": {f: sha256_of(os.path.join(args.doc_dir, f)) for f in sorted(os.listdir(args.doc_dir))
                    if f.endswith(".csv")},
        "bootstrap": {"n_boot": args.n_boot, "seed": args.seed, "stratified_by_class": True},
        "min_city_n": args.min_city_n,
        "headline_subset_per_type": args.subset,
    }
    old = {}
    pp = os.path.join(args.doc_dir, "provenance.json")
    if os.path.exists(pp):
        old = json.load(open(pp))
    prov = {**{k: v for k, v in old.items() if k in ("packet", "explain")}, **prov}
    with open(pp, "w") as f:
        json.dump(prov, f, indent=1, sort_keys=True)


def _sysctl(key):
    try:
        return subprocess.run(["sysctl", "-n", key], capture_output=True, text=True, timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _mlx_versions():
    out = {}
    for name in ("mlx", "mlx-vlm", "transformers", "torch"):
        try:
            from importlib.metadata import version
            out[name] = version(name)
        except Exception:  # noqa: BLE001 -- a missing optional package is simply not recorded
            pass
    return out


# --------------------------------------------------------------------------------------------------------------------
# Explanations and the blinded packet -- selection, blinding and unblinding are pure and unit-tested
# --------------------------------------------------------------------------------------------------------------------

EXPLAIN_CATEGORIES = ("vlm_disagrees_with_crowd", "vlm_agrees_production_disagrees")
FALLBACK_CATEGORY = "fallback_any_disagreement"


def select_explanation_items(best_pred, prod_pred, cities, n_total=100, seed=0):
    """Rows to explain: public deployments only, ``n_total / 5`` per type, half per category.

    ``best_pred`` and ``prod_pred`` are prediction frames (``label_uid, label_type, city, true, p_incorrect``) for
    the best VLM and production. Categories: ``vlm_disagrees_with_crowd`` (the VLM's argmax verdict differs from
    ``label_class``) and ``vlm_agrees_production_disagrees``. A short pool is filled from any other row where either
    model disagrees with the crowd (``fallback_any_disagreement``). Deterministic in ``seed``.
    """
    m = best_pred.merge(prod_pred[["label_uid", "p_incorrect"]], on="label_uid", suffixes=("", "_prod"),
                        validate="one_to_one")
    m = m[is_public(m.city, cities).values].sort_values("label_uid").reset_index(drop=True)
    m["vlm_verdict"] = verdict_from_score(m.p_incorrect)
    m["prod_verdict"] = verdict_from_score(m.p_incorrect_prod)
    vlm_wrong = m.vlm_verdict != m.true
    prod_wrong = m.prod_verdict != m.true
    per_type = n_total // len(LABEL_TYPES)
    rng = np.random.default_rng(seed)
    picks = []
    for t in LABEL_TYPES:
        pools = {EXPLAIN_CATEGORIES[0]: m[(m.label_type == t) & vlm_wrong],
                 EXPLAIN_CATEGORIES[1]: m[(m.label_type == t) & ~vlm_wrong & prod_wrong]}
        chosen = []
        for i, (cat, pool) in enumerate(pools.items()):
            want = per_type // 2 + (per_type % 2 if i == 0 else 0)
            take = pool.iloc[rng.permutation(len(pool))[:want]].assign(category=cat)
            chosen.append(take)
        got = pd.concat(chosen)
        short = per_type - len(got)
        if short > 0:
            rest = m[(m.label_type == t) & (vlm_wrong | prod_wrong) & ~m.label_uid.isin(got.label_uid)]
            chosen.append(rest.iloc[rng.permutation(len(rest))[:short]].assign(category=FALLBACK_CATEGORY))
        picks.append(pd.concat(chosen))
    return pd.concat(picks, ignore_index=True)


KEY_COLUMNS = ["packet_id", "label_uid", "label_type", "category", "label_class", "vlm_model", "vlm_answer",
               "p_incorrect", "p_incorrect_prod"]
SHEET_COLUMNS = ["packet_id", "label_type", "vlm_answer", "vlm_reason", "reason_correct", "would_change_my_vote",
                 "my_verdict", "notes"]
RUBRIC = {"reason_correct": ("yes", "partly", "no"), "would_change_my_vote": ("yes", "no"),
          "my_verdict": ("correct", "incorrect", "cant_tell")}


def packet_key(items, seed):
    """Blinding key: a random ``p<6 hex>`` id per item and a shuffled order. Deterministic in ``seed`` and the item
    set, so the seed must stay secret until the rating is done."""
    rng = np.random.default_rng(seed)
    s = items.sort_values("label_uid").reset_index(drop=True)
    ids = set()
    while len(ids) < len(s):
        ids.add(f"p{int(rng.integers(0, 16 ** 6)):06x}")
    s = s.assign(packet_id=rng.permutation(sorted(ids)))
    return s.iloc[rng.permutation(len(s))].reset_index(drop=True)


def blank_sheet(key):
    """The rater's sheet in packet order: what the rater sees (type, the VLM's answer and reason) + blank codes."""
    sheet = pd.DataFrame({"packet_id": key.packet_id, "label_type": key.label_type, "vlm_answer": key.vlm_answer,
                          "vlm_reason": key.vlm_reason})
    for c in SHEET_COLUMNS[4:]:
        sheet[c] = ""
    return sheet[SHEET_COLUMNS]


def validate_codes(sheet):
    """Raise if a filled sheet uses a value outside the rubric (blank cells are allowed)."""
    for col, values in RUBRIC.items():
        bad = set(sheet[col].fillna("").astype(str).str.strip().str.lower()) - set(values) - {""}
        if bad:
            raise ValueError(f"{col}: values {sorted(bad)} are not in the rubric ({', '.join(values)})")


def unblind(sheet, key):
    """Join a filled sheet to its key; codes lower-cased and stripped. Raises on ids missing from the key."""
    s = sheet[["packet_id"] + list(RUBRIC)].copy()
    for c in RUBRIC:
        s[c] = s[c].fillna("").astype(str).str.strip().str.lower()
    m = s.merge(key, on="packet_id", how="inner", validate="one_to_one")
    if len(m) != len(sheet):
        raise ValueError("sheet has packet ids that are not in the key")
    return m


def agreement_tables(m):
    """From an unblinded sheet: agreement of the rater's verdict with the VLM and with the crowd, per category, and
    the distribution of the reason codes. Rows without a rater verdict (blank or ``cant_tell``) are excluded from
    the agreement rates and counted."""
    vlm_verdict = np.where(m.vlm_answer.str.lower() == "no", "incorrect", "correct")
    r = m.assign(vlm_verdict=vlm_verdict, rated=m.my_verdict.isin(["correct", "incorrect"]))
    rows = []
    for cat, g in [("all", r)] + list(r.groupby("category")):
        rated = g[g.rated]
        rows.append({"category": cat, "n": len(g), "n_rated": len(rated),
                     "rater_agrees_vlm": float((rated.my_verdict == rated.vlm_verdict).mean()) if len(rated) else None,
                     "rater_agrees_crowd": float((rated.my_verdict == rated.label_class).mean()) if len(rated)
                     else None,
                     "reason_correct_yes": float((g.reason_correct == "yes").mean()),
                     "reason_correct_partly": float((g.reason_correct == "partly").mean()),
                     "reason_correct_no": float((g.reason_correct == "no").mean()),
                     "would_change_vote_yes": float((g.would_change_my_vote == "yes").mean())})
    return pd.DataFrame(rows)


def inside_this_repo(path):
    """True if ``path`` would land in any checkout (main or worktree) of the repo that holds this script."""
    here = os.path.dirname(os.path.abspath(__file__))
    d = os.path.dirname(os.path.abspath(path))
    while not os.path.isdir(d):
        d = os.path.dirname(d)
    mine, theirs = _git(here, "rev-parse", "--git-common-dir"), _git(d, "rev-parse", "--git-common-dir")
    if mine is None:
        return os.path.abspath(path).startswith(os.path.dirname(here) + os.sep)
    if theirs is None:
        return False
    return os.path.realpath(os.path.join(here, mine)) == os.path.realpath(os.path.join(d, theirs))


def run_explain(args):
    sample = load_sample(args.sample_dir)
    cities = load_cities(args.cities_json)
    scores = pd.concat([pd.read_csv(score_path(args.out_dir, m)) for m in (args.model, PRODUCTION_NAME)])
    pred = predictions(sample, scores)
    pred = pred[pred.variant == "definition"]
    items = select_explanation_items(pred[pred.model == args.model], pred[pred.model == PRODUCTION_NAME], cities,
                                     args.n_items, args.seed)
    print(items.groupby(["label_type", "category"]).size().to_string())
    vlm = MlxVlm(MODELS[args.model])
    img = dict(zip(sample.label_uid, sample.image))
    out = []
    for r in items.itertuples():
        q = build_prompt(r.label_type)
        answer = "No." if r.p_incorrect > 0.5 else "Yes."
        reason, latency = vlm.generate(img[r.label_uid], [("user", q), ("assistant", answer), ("user", WHY_PROMPT)],
                                       max_tokens=args.max_tokens)
        out.append({"label_uid": r.label_uid, "label_type": r.label_type, "category": r.category,
                    "label_class": r.true, "vlm_model": args.model, "vlm_answer": answer.rstrip("."),
                    "vlm_reason": reason, "p_incorrect": r.p_incorrect, "p_incorrect_prod": r.p_incorrect_prod,
                    "latency_s": latency, "n_words": len(reason.split())})
    path = os.path.join(args.out_dir, "explanations.csv")
    pd.DataFrame(out).to_csv(path, index=False)
    print(f"wrote {len(out)} explanations to {path}; median latency "
          f"{np.median([o['latency_s'] for o in out]):.2f}s")


def render_item(image_path, title, answer, reason, out_path, width=640):
    """The rater's image: the crop with a small cyan ring at the centre (the label point) and a caption panel."""
    import textwrap

    from PIL import Image, ImageDraw, ImageFont

    im = Image.open(image_path).convert("RGB")
    im = im.resize((width, int(im.height * width / im.width)))
    d = ImageDraw.Draw(im)
    cx, cy, rr = im.width // 2, im.height // 2, 10
    d.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], outline=(0, 255, 255), width=2)
    try:
        font = ImageFont.truetype("Arial.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
    lines = [title, f"VLM answer: {answer}"] + textwrap.wrap(f"VLM reason: {reason}", 62)
    panel = Image.new("RGB", (width, 28 * len(lines) + 16), (255, 255, 255))
    pd_ = ImageDraw.Draw(panel)
    for i, line in enumerate(lines):
        pd_.text((10, 8 + 28 * i), line, fill=(0, 0, 0), font=font)
    out = Image.new("RGB", (width, im.height + panel.height), (255, 255, 255))
    out.paste(im, (0, 0))
    out.paste(panel, (0, im.height))
    out.save(out_path, quality=90)


def run_packet(args, parser):
    if inside_this_repo(args.key):
        parser.error("--key must be outside every checkout of this repo until the rating is done "
                     "(commit only <key>.sha256)")
    ex = pd.read_csv(os.path.join(args.out_dir, "explanations.csv"))
    sample = load_sample(args.sample_dir)
    img = dict(zip(sample.label_uid, sample.image))
    key = packet_key(ex, args.seed)
    os.makedirs(os.path.join(args.packet_dir, "items"), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.key)), exist_ok=True)
    for i, k in enumerate(key.itertuples()):
        name = TYPE_NAMES[k.label_type]
        render_item(img[k.label_uid], f"{i + 1:03d}  {k.packet_id}  label type: {name}", k.vlm_answer,
                    k.vlm_reason, os.path.join(args.packet_dir, "items", f"{i + 1:03d}_{k.packet_id}.jpg"))
    blank_sheet(key).to_csv(os.path.join(args.packet_dir, "coding_sheet.csv"), index=False)
    with open(os.path.join(args.packet_dir, "RUBRIC.md"), "w") as f:
        f.write(RUBRIC_TEXT)
    key[KEY_COLUMNS + ["vlm_reason"]].to_csv(args.key, index=False)
    digest = sha256_of(args.key)
    with open(args.key + ".sha256", "w") as f:
        f.write(f"{digest}  {os.path.basename(args.key)}\n")
    if args.doc_dir:
        os.makedirs(args.doc_dir, exist_ok=True)
        with open(os.path.join(args.doc_dir, "packet_key.sha256"), "w") as f:
            f.write(f"{digest}  {os.path.basename(args.key)}\n")
        pp = os.path.join(args.doc_dir, "provenance.json")
        prov = json.load(open(pp)) if os.path.exists(pp) else {}
        ex_path = os.path.join(args.out_dir, "explanations.csv")
        prov["packet"] = {"n_items": int(len(key)), "key_sha256": digest,
                          "explanations_csv": {"sha256": sha256_of(ex_path), "rows": int(len(ex))},
                          "items_by_type_category": {f"{t}/{c}": int(n) for (t, c), n in
                                                     ex.groupby(["label_type", "category"]).size().items()},
                          "vlm_model": sorted(ex.vlm_model.unique()), "why_prompt": WHY_PROMPT,
                          "median_reason_words": float(ex.n_words.median()),
                          "median_reason_latency_s": float(ex.latency_s.median())}
        with open(pp, "w") as f:
            json.dump(prov, f, indent=1, sort_keys=True)
    print(f"wrote {len(key)} items to {args.packet_dir}/items, coding_sheet.csv and RUBRIC.md; key {args.key} "
          f"(sha256 {digest})")


RUBRIC_TEXT = """# Rating the VLM explanations (issue #16)

Each item shows a label crop (the labelled point is the centre, marked with a cyan ring), the label type, the
model's yes/no answer to "Is this a correct <type> label?", and its reason ("In at most 30 words, why?").
Fill one row of `coding_sheet.csv` per item, in any order. Do not look anything up; judge from the crop.

| column | values | meaning |
|---|---|---|
| `reason_correct` | `yes` / `partly` / `no` | Is the reason a true description of what the crop shows, and does it bear on the question? `partly`: some of it is right, or it is right but misses the deciding feature. |
| `would_change_my_vote` | `yes` / `no` | Had you been validating this label, would reading this answer and reason have changed your vote? |
| `my_verdict` | `correct` / `incorrect` / `cant_tell` | Your own verdict on the label, from the crop alone. |
| `notes` | free text | Optional. |

Rate the reason, not the answer: a wrong answer can have a correct description, and a right answer a wrong one.
"""


def run_agreement(args):
    if args.key_sha256:
        expected = open(args.key_sha256).read().split()[0]
        if sha256_of(args.key) != expected:
            raise SystemExit(f"{args.key} does not match {args.key_sha256}")
    key = pd.read_csv(args.key, dtype={"packet_id": str})
    sheet = pd.read_csv(args.sheet, dtype=str)
    validate_codes(sheet)
    table = agreement_tables(unblind(sheet, key))
    print(markdown(table))
    if args.out:
        table.to_csv(args.out, index=False, float_format="%.4f")


# --------------------------------------------------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p, out=True):
        p.add_argument("--sample-dir", default=os.path.expanduser("~/Data/sidewalk-validator-ai-retrain-2026-10/round3"),
                       help="directory with sample.csv and crops/ (default: the round-3 sample)")
        if out:
            p.add_argument("--out-dir", default=os.path.expanduser("~/Data/sidewalk-validator-ai-vlm-2026-10"),
                           help="private output directory, outside the repo")

    s = sub.add_parser("score", help="score the sample with one MLX VLM")
    common(s)
    s.add_argument("--model", required=True, choices=sorted(MODELS))
    s.add_argument("--variant", default="definition", choices=VARIANTS)
    s.add_argument("--label-types", default=None, help="comma-separated subset (default all five)")
    s.add_argument("--subset", type=int, default=0,
                   help="score only balanced_subset(per_type=N): class-balanced, both splits (0 = all crops)")
    s.add_argument("--limit", type=int, default=0, help="pilot: about this many crops per type, both classes")
    s.add_argument("--check-greedy", type=int, default=0,
                   help="also generate greedily on the first N crops and compare with the score's argmax")

    b = sub.add_parser("baseline", help="score the sample with the five production validators")
    common(b)
    b.add_argument("--device", default="cpu", help="cpu or mps")

    r = sub.add_parser("report", help="tables, figure and provenance into --doc-dir")
    common(r)
    r.add_argument("--doc-dir", default="docs/vlm-baseline-2026-10")
    r.add_argument("--cities-json", required=True, help="saved /v3/api/cities response (decides public naming)")
    r.add_argument("--min-city-n", type=int, default=30)
    r.add_argument("--subset", type=int, default=200,
                   help="headline tables on balanced_subset(per_type=N), the crops every VLM scored (0 = all)")
    r.add_argument("--n-boot", type=int, default=1000)
    r.add_argument("--seed", type=int, default=20261021, help="bootstrap seed")

    e = sub.add_parser("explain", help="pick the explanation items and ask the best VLM why")
    common(e)
    e.add_argument("--model", required=True, choices=sorted(MODELS))
    e.add_argument("--cities-json", required=True)
    e.add_argument("--n-items", type=int, default=100)
    e.add_argument("--seed", type=int, default=20261021, help="selection seed (not the blinding seed)")
    e.add_argument("--max-tokens", type=int, default=60)

    k = sub.add_parser("packet", help="blinded rater packet (outside the repo) and its key")
    common(k)
    k.add_argument("--packet-dir", required=True, help="packet directory, outside the repo")
    k.add_argument("--key", required=True, help="key CSV, outside every checkout; <key>.sha256 is written beside it")
    k.add_argument("--doc-dir", default="docs/vlm-baseline-2026-10",
                   help="where packet_key.sha256 and the packet's provenance entry go ('' to skip)")
    k.add_argument("--seed", type=int, required=True,
                   help="blinding seed; no default, keep it secret until rating is done")

    a = sub.add_parser("agreement", help="unblind a filled sheet; VLM-vs-rater and rater-vs-crowd agreement")
    a.add_argument("sheet")
    a.add_argument("--key", required=True)
    a.add_argument("--key-sha256", default=None, help="committed hash file to check --key against")
    a.add_argument("--out", default=None, help="optional CSV copy of the table")

    args = parser.parse_args(argv)
    if args.command == "score":
        run_score(args)
    elif args.command == "baseline":
        run_baseline(args)
    elif args.command == "report":
        run_report(args)
    elif args.command == "explain":
        run_explain(args)
    elif args.command == "packet":
        run_packet(args, parser)
    else:
        run_agreement(args)


if __name__ == "__main__":
    main()
