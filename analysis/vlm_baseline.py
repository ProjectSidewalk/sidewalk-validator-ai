"""Open-weight vision-language models as zero-shot label validators, against production (issue #16).

Asks current open-weight VLMs, run locally with MLX on Apple silicon, whether a crowdsourced Project Sidewalk label
is correct, on the round-3 crops (label-centred, tilt-corrected at beta 0.95, depth-sized, 640 px WebP: the inputs
the retrained validators are trained on), and compares them with the five production DINOv2-L validators on the same
crops. The production models were trained without tilt correction, so these crops are not exactly what production
sees. Also builds the blinded, two-pass explanation-rating packet the issue asks for.

**Score, not sample.** Each VLM gets one prompt per crop (``build_prompt``: a short definition of the label type,
"The labelled point is at the centre of the image.", "Is this a correct <type> label? Answer yes or no."). The
score is read from the next-token distribution after the prompt, never from sampled text:
``p_incorrect = P(no) / (P(yes) + P(no))``, where P(yes) sums the probability of every single-token spelling of yes
in ``YES_VARIANTS`` that the tokenizer has (likewise no), computed in log space (``yes_no_score``). ``answer_mass``
(P(yes) + P(no)) records how much of the distribution the two answers hold. ``greedy`` checks, on crops stratified
by type, whether greedy generation's first word agrees with the argmax of the score.

Subcommands (all outputs with label-level rows go OUTSIDE the repo, under ``~/Data/...``):

``score``
    One MLX model over the sample (or ``--subset N``: ``balanced_subset(per_type=N)``); writes
    ``<out-dir>/scores/<model>[__<variant>].csv`` (one row per crop: ``label_uid, model, variant, p_yes, p_no,
    answer_mass, log_odds_incorrect, p_incorrect, p_correct, top_token, prompt_tokens, latency_s``). Resumable: rows
    already in the CSV are skipped. ``latency_s`` here is not used for the latency table (see ``timing``).
``baseline``
    The five production validators (the revisions sidewalk-ai-api's ``model_map.py`` pins, ``PRODUCTION``) on the
    same crops, preprocessed as ``evaluate.py`` does; writes ``<out-dir>/scores/production.csv``.
``timing``
    Times 10 crops per type (``stratified_crops``) at batch 1 after one warm-up call, for one model (``--with-reason``
    also times the explanation turn); writes ``<out-dir>/timing/<model>.csv`` with the load average around the run.
    Run each model alone on an otherwise idle machine.
``greedy``
    Greedy text vs the score's argmax on 2 crops per type; writes ``<out-dir>/greedy/<model>.csv``.
``report``
    On the headline subset (every model) and on the full sample (models that scored all crops): metrics per model x
    type, per split with a bootstrap CI for the test - held-out difference, within-city AUC per split, per public city
    (n >= 30) and per city x type (n >= 20), private deployments pooled, the ``hard`` slice; best VLM vs production
    with label-resampled and deployment-resampled bootstrap CIs; latency from ``timing``; the prompt ablation; the
    greedy check; one ROC figure. Writes aggregate CSVs, ``report.md``, ``roc.png`` and ``provenance.json`` to
    ``--doc-dir``. Fails if a model covers only part of the headline subset.
``explain``
    Picks the explanation items (``select_explanation_items``: public deployments only, ``--per-type`` per type,
    half where the VLM's verdict matches the crowd) and asks the VLM "In at most 30 words, why?" as a second turn
    after its own answer; ``--reuse`` copies reasons for items an earlier run already explained. Writes
    ``<out-dir>/explanations.csv``.
``packet``
    The blinded two-pass rater packet from ``explanations.csv``, as two zips with the same items and ids:
    ``...-pass1.zip`` (``items_bare/``, crop only, and ``pass1_sheet.csv`` for the rater's own verdict) and
    ``...-pass2.zip`` (``items_annotated/``, crop + answer + reason, and ``pass2_sheet.csv``), the second handed out
    only after the filled pass-1 sheet is back; random 6-hex ids, shuffled with a secret ``--seed`` (no default);
    the key is written outside the checkout and only its sha256 is committed. Each zip also carries ``rate.html``
    (``analysis/rating_page.py``), a browser page that saves answers as you go and exports the filled sheet.
``gallery``
    Adds ``rate.html`` to a packet built before the page existed and re-zips it; key, ids, sheets and crops are
    unchanged, and it refuses a key that does not match the committed sha256.
``agreement``
    Unblinds the two filled sheets against the key; prints rater-vs-VLM and rater-vs-crowd agreement and the reason
    codes, overall, by whether the VLM was right, and per category.

Issue #63 adds the klone path (transformers, bf16, batched with left padding; same prompts and score, so rows are
comparable with the MLX ones): ``score-torch`` (one ``TORCH_MODELS`` leg over every held-out crop of the round-3 run
root, ``--fewshot 8`` for the in-context arm; resumable; ``analysis/vlm_score.slurm`` launches it), ``timing-torch``
(batch-1 latency), ``export-heldout`` (split slices and round-3 scores), ``timing-summary`` and ``baseline
--split-root`` (production on the same crops). Their outputs carry private-city label ids: keep them outside the repo.

Install (on top of ``requirements.txt``; the production baseline needs the pinned transformers 4.55, and mlx-vlm
upgrades transformers, so run ``baseline`` and production ``timing`` from a second venv with ``requirements.txt``
only)::

    uv venv --python 3.12 .venv && source .venv/bin/activate && uv pip install -r requirements.txt mlx-vlm==0.7.10
    uv venv --python 3.12 .venv-prod && VIRTUAL_ENV=.venv-prod uv pip install -r requirements.txt

Usage (``$S`` is the sample directory, ``$O`` the private output directory)::

    S=~/Data/sidewalk-validator-ai-retrain-2026-10/round3; O=~/Data/sidewalk-validator-ai-vlm-2026-10
    C=~/Data/sidewalk-validator-ai-prod-2026-10-17/cities.json
    python analysis/vlm_baseline.py score --model qwen3vl-8b --subset 100 --sample-dir $S --out-dir $O
    python analysis/vlm_baseline.py score --model qwen3vl-8b --subset 100 --label-types CurbRamp \
        --variant no-definition --sample-dir $S --out-dir $O          # the prompt ablation
    python analysis/vlm_baseline.py score --model gemma4-12b --sample-dir $S --out-dir $O   # all crops
    .venv-prod/bin/python analysis/vlm_baseline.py baseline --sample-dir $S --out-dir $O --device mps
    .venv-prod/bin/python analysis/vlm_baseline.py timing --model production --out-dir $O
    python analysis/vlm_baseline.py timing --model gemma4-12b --with-reason --out-dir $O
    python analysis/vlm_baseline.py greedy --model gemma4-12b --out-dir $O
    python analysis/vlm_baseline.py report --cities-json $C --sample-dir $S --out-dir $O \
        --doc-dir docs/vlm-baseline-2026-10
    python analysis/vlm_baseline.py explain --model gemma4-12b --cities-json $C --seed 20261021 \
        --reuse $O/superseded/explanations_v1.csv --sample-dir $S --out-dir $O
    python analysis/vlm_baseline.py packet --out-dir $O --packet-dir $O/packet \
        --key $O/private-key-do-not-share/packet_key.csv --seed $SECRET_SEED
    python analysis/vlm_baseline.py agreement $O/packet/pass1_filled.csv $O/packet/pass2_filled.csv \
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
import html
import json
import os
import platform
import subprocess
import sys
import time

import numpy as np
import pandas as pd

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_DIR)
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
    # Written for this baseline from the label types' meaning in Project Sidewalk; not copied from a source in this
    # repo. They are not parallel: NoCurbRamp and Crosswalk end with a "the label is wrong if..." clause, CurbRamp
    # lists exclusions, SurfaceProblem and Obstacle neither, which may favour some types (see the doc's caveats).
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
    for i, r in enumerate(todo.itertuples()):
        s = vlm.score(r.image, build_prompt(r.label_type, args.variant))
        rows.append({"label_uid": r.label_uid, "model": args.model, "variant": args.variant, **s})
        if len(rows) >= 25 or i == len(todo) - 1:
            pd.DataFrame(rows).to_csv(path, mode="a", header=not os.path.exists(path), index=False)
            rows = []
            el = time.time() - t_start
            print(f"  {i + 1}/{len(todo)}  {el:.0f}s  peak {mx.get_peak_memory() / 1e9:.1f} GB", flush=True)
    meta["peak_memory_gb"] = mx.get_peak_memory() / 1e9
    with open(path.replace(".csv", ".meta.json"), "w") as f:
        json.dump(meta, f, indent=1)


def stratified_crops(sample, per_type, seed=20261021, subset=100):
    """``per_type`` crops per label type from ``balanced_subset(per_type=subset)``, half per class, seeded.

    Used for the timing runs (10 per type) and the greedy check (2 per type), so both cover every type and class
    rather than the first rows of one type.

    >>> s = pd.DataFrame({"label_uid": [f"c:{i}" for i in range(40)], "label_type": "CurbRamp",
    ...                   "split": ["test"] * 20 + ["test_heldout"] * 20, "label_class": ["correct", "incorrect"] * 20})
    >>> stratified_crops(s, 4, subset=10).label_class.value_counts().tolist()
    [2, 2]
    """
    rng = np.random.default_rng(seed)
    pool = balanced_subset(sample, subset).sort_values("label_uid")
    out = []
    for (_t, _c), g in pool.groupby(["label_type", "label_class"]):
        k = per_type // 2 + (per_type % 2 if _c == "correct" else 0)
        out.append(g.iloc[np.sort(rng.permutation(len(g))[:k])])
    return pd.concat(out).sort_values(["label_type", "label_uid"]).reset_index(drop=True)


def _load_note():
    """1/5/15-minute load averages, recorded beside every timing run."""
    try:
        return [round(x, 2) for x in os.getloadavg()]
    except OSError:
        return None


def run_timing(args):
    """Time ``--n-per-type`` crops per type at batch 1, one warm-up call first (excluded). Writes
    ``<out>/timing/<model>.csv`` (``label_uid, label_type, latency_s[, reason_latency_s]``) and a meta JSON with the
    load average before and after. Run it on an otherwise idle machine."""
    sample = load_sample(args.sample_dir)
    crops = stratified_crops(sample, args.n_per_type)
    out_dir = os.path.join(args.out_dir, "timing")
    os.makedirs(out_dir, exist_ok=True)
    meta = {"model": args.model, "n": int(len(crops)), "date": datetime.datetime.now().isoformat(timespec="seconds"),
            "load_before": _load_note(), "batch_size": 1}
    rows = []
    if args.model == PRODUCTION_NAME:
        import torch
        import transformers
        from PIL import Image
        from transformers import AutoImageProcessor, AutoModelForImageClassification

        meta.update(device=args.device, torch=torch.__version__, transformers=transformers.__version__)
        for label_type, g in crops.groupby("label_type"):
            repo, rev = PRODUCTION[label_type]
            proc = AutoImageProcessor.from_pretrained(repo, revision=rev, use_fast=False)
            model = AutoModelForImageClassification.from_pretrained(repo, revision=rev).to(args.device).eval()

            def once(path):
                t0 = time.perf_counter()
                inputs = proc(Image.open(path).convert("RGB"), return_tensors="pt").to(args.device)
                with torch.no_grad():
                    model(**inputs).logits.softmax(-1).float().cpu().numpy()
                return time.perf_counter() - t0
            once(g.image.iloc[0])   # warm-up
            rows += [{"label_uid": r.label_uid, "label_type": r.label_type, "latency_s": once(r.image)}
                     for r in g.itertuples()]
            del model
    else:
        vlm = MlxVlm(MODELS[args.model])
        first = crops.iloc[0]
        vlm.score(first.image, build_prompt(first.label_type))   # warm-up
        for r in crops.itertuples():
            q = build_prompt(r.label_type)
            sc = vlm.score(r.image, q)
            row = {"label_uid": r.label_uid, "label_type": r.label_type, "latency_s": sc["latency_s"],
                   "prompt_tokens": sc["prompt_tokens"]}
            if args.with_reason:
                answer = "No." if sc["p_incorrect"] > 0.5 else "Yes."
                text, lat = vlm.generate(r.image, [("user", q), ("assistant", answer), ("user", WHY_PROMPT)],
                                         max_tokens=60)
                row.update(reason_latency_s=lat, reason_words=len(text.split()))
            rows.append(row)
    meta["load_after"] = _load_note()
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(out_dir, f"{args.model}.csv"), index=False)
    with open(os.path.join(out_dir, f"{args.model}.meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"{args.model}: n {len(t)}, median {t.latency_s.median():.3f}s, p10 {t.latency_s.quantile(.1):.3f}, "
          f"p90 {t.latency_s.quantile(.9):.3f}; load {meta['load_before']} -> {meta['load_after']}")
    if "reason_latency_s" in t:
        print(f"  reason: median {t.reason_latency_s.median():.3f}s")


def run_greedy(args):
    """Greedy-vs-score check on ``--n-per-type`` crops per type (``stratified_crops``, seed differs from timing).
    Writes ``<out>/greedy/<model>.csv``: the greedy text, the score, the top token and whether they agree."""
    sample = load_sample(args.sample_dir)
    crops = stratified_crops(sample, args.n_per_type, seed=args.seed)
    vlm = MlxVlm(MODELS[args.model])
    rows = []
    for r in crops.itertuples():
        q = build_prompt(r.label_type)
        sc = vlm.score(r.image, q)
        text, _ = vlm.generate(r.image, [("user", q)], max_tokens=4)
        rows.append({"label_uid": r.label_uid, "label_type": r.label_type, "greedy_text": text,
                     "p_incorrect": sc["p_incorrect"], "top_token": sc["top_token"],
                     "agrees": greedy_agrees(text, sc["p_incorrect"])})
    out_dir = os.path.join(args.out_dir, "greedy")
    os.makedirs(out_dir, exist_ok=True)
    c = pd.DataFrame(rows)
    c.to_csv(os.path.join(out_dir, f"{args.model}.csv"), index=False)
    print(f"{args.model}: greedy agrees {int((c.agrees == True).sum())}/{len(c)} "  # noqa: E712
          f"({int(c.agrees.isna().sum())} unparsable)")


# --------------------------------------------------------------------------------------------------------------------
# Production baseline -- needs torch + transformers 4.55 (requirements.txt)
# --------------------------------------------------------------------------------------------------------------------

def run_baseline(args):
    import torch
    import transformers
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    if args.split_root:   # issue #63: every held-out crop, one CSV per type under <out-dir>/scores/production/
        sample = pd.concat([load_split_rows(args.split_root, t) for t in LABEL_TYPES], ignore_index=True)
        path = os.path.join(args.out_dir, "scores", PRODUCTION_NAME, "all.csv")
    else:
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
    meta = {"model": PRODUCTION_NAME, "repos": PRODUCTION, "device": args.device, "torch": torch.__version__,
            "transformers": transformers.__version__}
    if args.split_root:
        meta.update(_gpu_info())
        d = pd.DataFrame(rows).merge(sample[["label_uid", "label_type"]], on="label_uid", validate="one_to_one")
        for t, g in d.groupby("label_type"):
            g.drop(columns="label_type").to_csv(os.path.join(os.path.dirname(path), f"{t}.csv"), index=False)
        path = os.path.join(os.path.dirname(path), "production.csv")   # meta name only
    else:
        pd.DataFrame(rows).to_csv(path, index=False)
    with open(path.replace(".csv", ".meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"wrote {path}")


# --------------------------------------------------------------------------------------------------------------------
# Torch inference on klone (issue #63) -- needs torch + transformers; the pure helpers are unit-tested on CPU
# --------------------------------------------------------------------------------------------------------------------

TORCH_MODELS = {
    # short name -> repo, the chat-template switch that turns thinking off (None: the template has none), whether
    # the Hub code is needed, and the env it runs in on klone. Read the thinking switch from each repo's chat template.
    "qwen3.8-27b": {"repo": "Qwen/Qwen3.8-27B", "thinking": "enable_thinking", "remote_code": False,
                    "env": "vlm-scoreboard"},
    "gemma4-31b": {"repo": "google/gemma-4-31B-it", "thinking": "enable_thinking", "remote_code": False,
                   "env": "vlm-scoreboard"},
    "gemma4-12b-bf16": {"repo": "google/gemma-4-12B-it", "thinking": "enable_thinking", "remote_code": False,
                        "env": "vlm-scoreboard"},
    "molmo2-8b": {"repo": "allenai/Molmo2-8B", "thinking": None, "remote_code": True, "env": "molmo2"},
    "minicpm-v4.7-35b-a3b": {"repo": "openbmb/MiniCPM-V-4.7-35B-A3B", "thinking": "enable_thinking",
                             "remote_code": False, "env": "vlm-scoreboard"},
    # Muse has no on/off switch: its generation prompt ends at "<|start|>assistant" and the model picks a channel
    # (" to=self" is reasoning). The prefix opens the final-answer channel directly; reasoning_strength is the
    # template's own knob (default "high"), set to "low" so the system line does not ask for reasoning.
    "muse-glimmer-30b": {"repo": "meta-models/Muse-Glimmer-30B", "thinking": None, "remote_code": False,
                         "env": "vlm-scoreboard", "template_kwargs": {"reasoning_strength": "low"},
                         "answer_prefix": " to=user<|message|>"},
    "qwen3vl-8b-bf16": {"repo": "Qwen/Qwen3-VL-8B-Instruct", "thinking": None, "remote_code": False,
                        "env": "vlm-scoreboard"},
    "qwen3vl-32b-bf16": {"repo": "Qwen/Qwen3-VL-32B-Instruct", "thinking": None, "remote_code": False,
                         "env": "vlm-scoreboard"},
    "nemotron3.5-super-vl-120b": {"repo": "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16",
                                  "thinking": "enable_thinking", "remote_code": True, "env": "vlm-scoreboard"},
    # Phi's system prompt has the model open its answer with "<nothink>" or "<think>"; the prefix picks NOTHINK.
    # Its Hub code needs transformers < 5 (runs in the molmo2 env, 4.57.1) and takes "<image>" placeholders in
    # plain-text messages templated by the tokenizer, not image content parts (``image_token``).
    "phi4-rv-15b": {"repo": "microsoft/Phi-4-reasoning-vision-15B", "thinking": None, "remote_code": True,
                    "env": "molmo2", "answer_prefix": "<nothink>", "image_token": "<image>",
                    "local_path": True},   # its from_pretrained finds the vision weights only in a local directory
}
"""Roster of the transformers legs (issue #63). ``thinking`` is the template kwarg set to False, or None;
``template_kwargs`` are further template kwargs; ``answer_prefix`` is appended after the generation prompt, for
templates whose thinking is chosen by the model's first tokens rather than by a switch (read from each chat
template, 2026-10-26)."""

HELDOUT_SPLITS = ("test", "test_heldout")
FEWSHOT_SEED = 20261021
SPLIT_COLUMNS = ["label_uid", "city", "label_type", "label_class", "split", "pano_id"]


def chat_kwargs(short):
    """Extra ``apply_chat_template`` kwargs for a roster model: its thinking switch turned off, or nothing.

    >>> chat_kwargs("qwen3.8-27b"), chat_kwargs("qwen3vl-8b-bf16")
    ({'enable_thinking': False}, {})
    """
    spec = TORCH_MODELS[short]
    kw = dict(spec.get("template_kwargs") or {})
    if spec["thinking"]:
        kw[spec["thinking"]] = False
    return kw


def with_answer_prefix(text, short):
    """The templated prompt with the model's ``answer_prefix`` appended (unchanged when it has none).

    >>> with_answer_prefix("<|im_start|>assistant<|im_sep|>", "phi4-rv-15b")
    '<|im_start|>assistant<|im_sep|><nothink>'
    """
    return text + (TORCH_MODELS[short].get("answer_prefix") or "")


def leg_name(model, variant="definition", fewshot=0):
    """Directory name of one scored leg: ``<model>[-fewshot<N>][__<variant>]``.

    >>> leg_name("gemma4-31b", fewshot=8), leg_name("qwen3vl-8b-bf16", "no-definition")
    ('gemma4-31b-fewshot8', 'qwen3vl-8b-bf16__no-definition')
    """
    name = model + (f"-fewshot{fewshot}" if fewshot else "")
    return name + ("" if variant == "definition" else f"__{variant}")


def load_split_rows(root, label_type, splits=HELDOUT_SPLITS):
    """Rows of ``<root>/<Type>/split.csv`` in ``splits`` with an absolute ``image`` column; unique uids."""
    df = pd.read_csv(os.path.join(root, label_type, "split.csv"), dtype={"label_id": str}, low_memory=False)
    if "status" in df:
        df = df[df.status == "ok"]
    df = df[df.split.isin(splits)].copy()
    if df.label_uid.duplicated().any():
        raise ValueError(f"{label_type}: duplicate label_uid in split.csv")
    df["image"] = [os.path.join(root, label_type, p) for p in df.crop_path]
    return df.sort_values("label_uid").reset_index(drop=True)


def pilot_rows(df, limit):
    """The first ``limit`` rows of one type, half per class (by ``label_uid``), for smoke runs.

    >>> d = pd.DataFrame({"label_uid": list("abcdef"), "label_class": ["correct"] * 3 + ["incorrect"] * 3})
    >>> pilot_rows(d, 4).label_uid.tolist()
    ['a', 'b', 'd', 'e']
    """
    return (df.sort_values(["label_class", "label_uid"]).groupby("label_class").head(max(1, limit // 2))
            .sort_values(["label_class", "label_uid"]).reset_index(drop=True))


def last_token_positions(attention_mask):
    """Index of each row's last real token, for left OR right padding (``attention_mask`` B x T of 0/1).

    With left padding every row ends at ``T - 1``; with right padding row i ends at ``sum(mask_i) - 1``.

    >>> last_token_positions(np.array([[0, 0, 1, 1], [1, 1, 1, 1]])).tolist()   # left padded
    [3, 3]
    >>> last_token_positions(np.array([[1, 1, 0, 0], [1, 1, 1, 1]])).tolist()   # right padded
    [1, 3]
    """
    m = np.asarray(attention_mask).astype(int)
    t = m.shape[1]
    # the last index whose mask is 1: reverse, argmax finds the first 1 from the end
    return (t - 1 - np.argmax(m[:, ::-1], axis=1)).astype(int)


def score_rows(last_logits, yes_ids, no_ids, decode=None):
    """``yes_no_score`` per row of a B x V array of next-token logits (float32 log-softmax first)."""
    x = np.asarray(last_logits, dtype=np.float64)
    lp = x - (x.max(axis=1, keepdims=True) + np.log(np.exp(x - x.max(axis=1, keepdims=True)).sum(1, keepdims=True)))
    out = []
    for row in lp:
        s = yes_no_score(row, yes_ids, no_ids)
        s["top_token"] = decode([int(row.argmax())]) if decode else int(row.argmax())
        out.append(s)
    return out


def fewshot_pools(train, seed=FEWSHOT_SEED):
    """Per class, the train rows of one type in a seeded order: the candidate list the examples are drawn from.

    Sorted by ``label_uid`` first so the order depends only on the data, then permuted with ``seed``.
    """
    pools = {}
    rng = np.random.default_rng(seed)
    for cls in evaluate.CLASSES:
        g = train[train.label_class == cls].sort_values("label_uid").reset_index(drop=True)
        pools[cls] = g.iloc[rng.permutation(len(g))].reset_index(drop=True)
    return pools


def fewshot_examples(pools, query_pano=None, n=8):
    """The ``n`` in-context examples for one query: ``n // 2`` per class, the first of each pool whose panorama is
    not the query's (a clash is replaced by the pool's next example), interleaved correct/incorrect. For every query
    whose panorama is not among them the set is the same, so the prompt prefix is shared.

    >>> tr = pd.DataFrame({"label_uid": [f"c:{i}" for i in range(8)], "pano_id": list("ppqrstuv"),
    ...                    "label_class": ["correct", "incorrect"] * 4})
    >>> pools = {c: tr[tr.label_class == c].reset_index(drop=True) for c in ("correct", "incorrect")}
    >>> fewshot_examples(pools, "x", 4).label_uid.tolist(), fewshot_examples(pools, "p", 4).label_uid.tolist()
    (['c:0', 'c:1', 'c:2', 'c:3'], ['c:2', 'c:3', 'c:4', 'c:5'])
    """
    if n % 2:
        raise ValueError("n must be even (half per class)")
    picks = {}
    for cls in evaluate.CLASSES:
        p = pools[cls]
        p = p[p.pano_id != query_pano] if query_pano is not None else p
        if len(p) < n // 2:
            raise ValueError(f"only {len(p)} {cls} examples for {n // 2}")
        picks[cls] = p.head(n // 2).reset_index(drop=True)
    rows = []
    for i in range(n // 2):
        rows += [picks["correct"].iloc[i], picks["incorrect"].iloc[i]]
    return pd.DataFrame(rows).reset_index(drop=True)


def build_messages(question, image, examples=()):
    """Chat messages for one crop: optional ``(image, answer)`` example turns, each the same question with its
    answer as the assistant reply, then the query. The image is first in every user turn.

    >>> m = build_messages("Q?", "img", [("e1", "Yes"), ("e2", "No")])
    >>> [x["role"] for x in m], m[1]["content"][0]["text"], m[-1]["content"][0]["image"]
    (['user', 'assistant', 'user', 'assistant', 'user'], 'Yes', 'img')
    """
    msgs = []
    for ex_image, answer in examples:
        msgs.append({"role": "user", "content": [{"type": "image", "image": ex_image},
                                                 {"type": "text", "text": question}]})
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
    msgs.append({"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": question}]})
    return msgs


def text_only_messages(messages, image_token):
    """``build_messages`` output flattened to plain-text contents with ``image_token`` + newline where each image
    was, and the images in order, for processors that take placeholders in text (Phi-4-reasoning-vision).

    >>> m, ims = text_only_messages(build_messages("Q?", "q", [("e", "Yes")]), "<image>")
    >>> [x["content"] for x in m], ims
    (['<image>\\nQ?', 'Yes', '<image>\\nQ?'], ['e', 'q'])
    """
    out, images = [], []
    for m in messages:
        parts = []
        for p in m["content"]:
            if p.get("type") == "image":
                parts.append(image_token + "\n")
                images.append(p["image"])
            else:
                parts.append(p["text"])
        out.append({"role": m["role"], "content": "".join(parts)})
    return out, images


def todo_rows(df, path):
    """Rows of ``df`` whose ``label_uid`` is not yet in the score CSV at ``path`` (resumability)."""
    if not os.path.exists(path):
        return df
    done = set(pd.read_csv(path, usecols=["label_uid"]).label_uid)
    return df[~df.label_uid.isin(done)]


def answer_word(label_class):
    """The in-context answer for an example of ``label_class``."""
    return "No" if label_class == "incorrect" else "Yes"


def patch_cuda_prod(torch):
    """Work around ``Tensor.prod`` failing on CUDA ("CUDA driver error: invalid argument", torch 2.14.1+cu126 with
    klone's 580 driver, measured 2026-10-26 on A40 and H200; ``sum`` and ``cumprod`` work). Qwen3-VL and Molmo2 take
    ``prod`` of their image grids on the GPU. If a probe fails, ``torch.prod`` / ``Tensor.prod`` on CUDA tensors are
    routed through the last element of ``cumprod``, which is the same product. Returns whether it patched."""
    try:
        torch.ones(2, 2, device="cuda", dtype=torch.long).prod(-1).tolist()
        return False
    except RuntimeError:
        pass
    orig_t, orig_f = torch.Tensor.prod, torch.prod

    def _prod(x, dim=None, keepdim=False, *, dtype=None):
        if not x.is_cuda:
            return (orig_t(x, dtype=dtype) if dim is None else orig_t(x, dim, keepdim=keepdim, dtype=dtype))
        y = x if dtype is None else x.to(dtype)
        if dim is None:
            y, dim = y.reshape(-1), 0
        if y.shape[dim] == 0:
            shape = list(y.shape)
            shape[dim] = 1
            out = torch.ones(shape, dtype=y.dtype, device=y.device)
        else:
            out = y.cumprod(dim).narrow(dim, y.shape[dim] - 1, 1)
        return out if keepdim else out.squeeze(dim)

    torch.Tensor.prod = _prod
    torch.prod = lambda x, *a, **k: _prod(x, *a, **k) if x.is_cuda else orig_f(x, *a, **k)
    return True


class TorchVlm:
    """One transformers VLM in bf16 with the yes/no scorer, batched with left padding."""

    def __init__(self, short):
        import inspect

        import torch
        from transformers import AutoProcessor

        self.short, self.spec = short, TORCH_MODELS[short]
        self.repo, rc = self.spec["repo"], self.spec["remote_code"]
        self.source = self.repo
        if self.spec.get("local_path"):
            from huggingface_hub import snapshot_download
            self.source = snapshot_download(self.repo, local_files_only=True)
        self.torch = torch
        self.cuda_prod_patched = patch_cuda_prod(torch) if torch.cuda.is_available() else False
        self.processor = AutoProcessor.from_pretrained(self.source, trust_remote_code=rc, padding_side="left")
        self.tokenizer = getattr(self.processor, "tokenizer", self.processor)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        n_gpu = torch.cuda.device_count()
        self.device_map = "auto" if n_gpu > 1 else ("cuda" if n_gpu else "cpu")
        self.model = self._load(rc)
        self.model.eval()
        enc = lambda s: self.tokenizer.encode(s, add_special_tokens=False)  # noqa: E731
        self.yes_ids, self.yes_used = answer_token_ids(enc, YES_VARIANTS)
        self.no_ids, self.no_used = answer_token_ids(enc, NO_VARIANTS)
        self.keep_one = "logits_to_keep" in inspect.signature(self.model.forward).parameters
        self.input_device = "cuda:0" if n_gpu else "cpu"

    def _load(self, rc):
        import transformers

        kw = dict(dtype=self.torch.bfloat16, device_map=self.device_map, trust_remote_code=rc)
        errors = []
        for cls_name in ("AutoModelForImageTextToText", "AutoModelForCausalLM"):
            cls = getattr(transformers, cls_name)
            for attn in ("sdpa", None):
                try:
                    m = cls.from_pretrained(self.source, **kw, **({"attn_implementation": attn} if attn else {}))
                    self.loader = f"{cls_name}, attn={attn or 'default'}"
                    return m
                except (ValueError, KeyError, TypeError, ImportError) as e:
                    errors.append(f"{cls_name}/{attn}: {type(e).__name__}: {str(e)[:300]}")
        raise RuntimeError("could not load " + self.repo + "\n" + "\n".join(errors))

    def inputs(self, conversations):
        kw = chat_kwargs(self.short)
        if self.spec.get("image_token"):   # text placeholders, tokenizer template, flat image list
            texts, images = [], []
            for c in conversations:
                msgs, ims = text_only_messages(c, self.spec["image_token"])
                texts.append(with_answer_prefix(self.tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True, **kw), self.short))
                images += ims
            b = self.processor(text=texts, images=images, padding=True, return_tensors="pt")
        elif self.spec.get("answer_prefix"):   # template text + prefix, then the processor (no second BOS)
            texts = [with_answer_prefix(self.processor.apply_chat_template(c, tokenize=False, add_generation_prompt=True,
                                                                           **kw), self.short) for c in conversations]
            images = [[p["image"] for m in c for p in m["content"] if p.get("type") == "image"] for c in conversations]
            b = self.processor(text=texts, images=images, padding=True, return_tensors="pt", add_special_tokens=False)
        else:
            b = self.processor.apply_chat_template(conversations, tokenize=True, return_dict=True,
                                                   return_tensors="pt", padding=True, add_generation_prompt=True, **kw)
        out = {}
        for k, v in b.items():
            if hasattr(v, "to"):
                v = v.to(self.input_device)
                if v.is_floating_point():
                    v = v.to(self.model.dtype)   # bf16 unless a probe casts the model
            out[k] = v
        return out

    def score_batch(self, conversations):
        """Score a list of conversations; ``(rows, seconds)`` with per-row prompt token counts."""
        t0 = time.perf_counter()
        inp = self.inputs(conversations)
        mask = inp["attention_mask"].cpu().numpy()
        left = bool(mask[:, -1].all())
        kw = {"logits_to_keep": 1} if (self.keep_one and left) else {}
        with self.torch.inference_mode():
            logits = self.model(**inp, **kw).logits
        if kw:
            last = logits[:, -1, :]
        else:
            pos = self.torch.as_tensor(last_token_positions(mask), device=logits.device)
            last = logits[self.torch.arange(logits.shape[0], device=logits.device), pos]
        last = self.torch.log_softmax(last.float(), dim=-1).cpu().numpy()
        dt = time.perf_counter() - t0
        rows = score_rows(last, self.yes_ids, self.no_ids, decode=self.tokenizer.decode)
        for r, n in zip(rows, mask.sum(1)):
            r["prompt_tokens"] = int(n)
        return rows, dt


def _gpu_info():
    try:
        import torch
        if not torch.cuda.is_available():
            return {"gpu": "cpu", "n_gpus": 0}
        names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        # the count is in the name ("2xNVIDIA H200") so GPU-hours computed from it count every card
        gpu = names[0] if len(set(names)) == 1 else "+".join(names)
        return {"gpu": gpu if len(names) == 1 else f"{len(names)}x{gpu}", "n_gpus": len(names),
                "gpu_memory_gb": [round(torch.cuda.get_device_properties(i).total_memory / 1e9, 1)
                                  for i in range(len(names))]}
    except Exception as e:   # noqa: BLE001 -- provenance only
        return {"gpu": f"unknown ({type(e).__name__})"}


def _peak_memory_gb():
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    return round(sum(torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())) / 1e9, 2)


def _env_versions():
    out = {"python": platform.python_version(), "host": platform.node()}
    for mod in ("torch", "transformers", "accelerate", "PIL"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:   # noqa: BLE001
            out[mod] = None
    out["slurm_job_id"] = os.environ.get("SLURM_JOB_ID")
    out["slurm_restart_count"] = os.environ.get("SLURM_RESTART_COUNT", "0")
    out["git_commit"] = _git(REPO_DIR, "rev-parse", "HEAD")
    return out


def _read_meta(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def _write_meta(path, meta):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(meta, f, indent=1, default=str)
    os.replace(tmp, path)


def batch_check(batched, alone, uids):
    """Batch-of-N vs batch-of-1 agreement on the same rows (lists of score dicts): max and median absolute
    differences in ``p_incorrect`` and in ``log_odds_incorrect``, and whether every p is within 1e-3.

    >>> b = [{"p_incorrect": .5, "log_odds_incorrect": 0.0}, {"p_incorrect": .9, "log_odds_incorrect": 2.2}]
    >>> a = [{"p_incorrect": .5, "log_odds_incorrect": 0.0}, {"p_incorrect": .9005, "log_odds_incorrect": 2.2062}]
    >>> c = batch_check(b, a, ["x:1", "x:2"])
    >>> c["n"], round(c["max_abs_dp"], 4), c["worst_label_uid"], c["within_1e-3"]
    (2, 0.0005, 'x:2', True)
    """
    dp = np.abs([x["p_incorrect"] - y["p_incorrect"] for x, y in zip(batched, alone)])
    dl = np.abs([x["log_odds_incorrect"] - y["log_odds_incorrect"] for x, y in zip(batched, alone)])
    return {"n": int(len(dp)), "batch_size": int(len(batched)), "max_abs_dp": float(dp.max()),
            "median_abs_dp": float(np.median(dp)), "max_abs_dlogodds": float(dl.max()),
            "median_abs_dlogodds": float(np.median(dl)), "worst_label_uid": uids[int(dp.argmax())],
            "within_1e-3": bool(dp.max() <= 1e-3)}


def run_score_torch(args):
    """Score the held-out crops of ``--label-types`` with one roster model; one CSV + meta per type under
    ``<out-dir>/scores/<leg>/``. Resumable: rows already in the CSV are skipped (Slurm ``--requeue`` safe)."""
    from PIL import Image

    types = args.label_types.split(",") if args.label_types else list(LABEL_TYPES)
    leg = leg_name(args.model, args.variant, args.fewshot)
    out_dir = os.path.join(args.out_dir, "scores", leg)
    os.makedirs(out_dir, exist_ok=True)
    vlm = None
    gpu = _gpu_info()
    for t in types:
        df = load_split_rows(args.split_root, t)
        if args.limit:
            df = pilot_rows(df, args.limit)
        path = os.path.join(out_dir, f"{t}.csv")
        todo = todo_rows(df, path)
        print(f"{leg} {t}: {len(todo)} to score, {len(df) - len(todo)} done", flush=True)
        if todo.empty:
            continue
        if vlm is None:
            t_load = time.perf_counter()
            vlm = TorchVlm(args.model)
            load_s = time.perf_counter() - t_load
            print(f"loaded {vlm.repo} ({vlm.loader}, device_map={vlm.device_map}) in {load_s:.0f}s; "
                  f"yes {vlm.yes_used} no {vlm.no_used}", flush=True)
        question = build_prompt(t, args.variant)
        pools = fewshot_pools(load_split_rows(args.split_root, t, ("train",))) if args.fewshot else None
        base = fewshot_examples(pools, None, args.fewshot) if args.fewshot else None
        img_cache = {}

        def image(p):
            if p not in img_cache:
                img_cache[p] = Image.open(p).convert("RGB")
            return img_cache[p]

        def conv(r):
            exs = []
            if args.fewshot:
                ex = base if r.pano_id not in set(base.pano_id) else fewshot_examples(pools, r.pano_id, args.fewshot)
                exs = [(image(e.image), answer_word(e.label_class)) for e in ex.itertuples()]
            return build_messages(question, Image.open(r.image).convert("RGB"), exs)

        meta_path = path.replace(".csv", ".meta.json")
        meta = _read_meta(meta_path)
        meta.update({"model": args.model, "leg": leg, "repo": vlm.repo, "revision": hf_snapshot(vlm.repo)[0],
                     "variant": args.variant, "label_type": t, "prompt": question, "fewshot": args.fewshot,
                     "chat_kwargs": chat_kwargs(args.model), "loader": vlm.loader, "device_map": vlm.device_map,
                     "cuda_prod_patched": getattr(vlm, "cuda_prod_patched", False),
                     "yes_tokens": vlm.yes_used, "yes_ids": vlm.yes_ids, "no_tokens": vlm.no_used,
                     "no_ids": vlm.no_ids, "batch_size": args.batch, "n_crops": int(len(df)), **gpu})
        if args.fewshot:
            meta["fewshot_examples"] = base.label_uid.tolist()
            meta["fewshot_seed"] = FEWSHOT_SEED
        runs = meta.setdefault("runs", [])
        run = {"start": datetime.datetime.now().isoformat(timespec="seconds"), **_env_versions(),
               "n_todo": int(len(todo))}
        runs.append(run)
        swaps = 0
        rows, lat = [], []
        recs = list(todo.itertuples())
        for i in range(0, len(recs), args.batch):
            chunk = recs[i:i + args.batch]
            convs = [conv(r) for r in chunk]
            swaps += sum(args.fewshot > 0 and r.pano_id in set(base.pano_id) for r in chunk)
            scored, dt = vlm.score_batch(convs)
            if "batch_check" not in meta and len(chunk) > 1:
                # every row of the first batch scored again alone: padding and batch-composition effects. A single
                # row is not enough: near p = 0 or 1 a large log-odds change moves p by almost nothing.
                alone = [vlm.score_batch([c])[0][0] for c in convs]
                meta["batch_check"] = batch_check(scored, alone, [r.label_uid for r in chunk])
                bc = meta["batch_check"]
                print(f"batch check: max |batch - alone| p {bc['max_abs_dp']:.2e}, log-odds "
                      f"{bc['max_abs_dlogodds']:.2e} over {bc['n']} rows", flush=True)
                if bc["max_abs_dp"] > args.max_batch_diff:
                    _write_meta(meta_path, meta)
                    raise RuntimeError(f"batch-of-{len(chunk)} vs batch-of-1 differ by up to {bc['max_abs_dp']:.3g} "
                                       f"in p_incorrect; rerun with --batch 1")
            if i > 0:   # the first batch includes warm-up
                lat.append(dt / len(chunk))
            for r, s in zip(chunk, scored):
                rows.append({"label_uid": r.label_uid, "model": leg, "variant": args.variant, **s,
                             "latency_s": dt / len(chunk), "batch_size": len(chunk), "gpu": gpu.get("gpu")})
            if len(rows) >= args.chunk or i + args.batch >= len(recs):
                cols = ["label_uid", "model", "variant", "p_yes", "p_no", "answer_mass", "log_odds_incorrect",
                        "p_incorrect", "p_correct", "top_token", "prompt_tokens", "latency_s", "batch_size", "gpu"]
                pd.DataFrame(rows)[cols].to_csv(path, mode="a", header=not os.path.exists(path), index=False)
                rows = []
                run.update(n_scored=i + len(chunk), peak_memory_gb=_peak_memory_gb(),
                           s_per_crop_median=float(np.median(lat)) if lat else None, fewshot_swaps=int(swaps))
                meta["peak_memory_gb"] = max(meta.get("peak_memory_gb") or 0, run["peak_memory_gb"] or 0)
                _write_meta(meta_path, meta)
                print(f"  {t} {i + len(chunk)}/{len(recs)}  {run['s_per_crop_median'] or 0:.3f} s/crop  "
                      f"peak {run['peak_memory_gb']} GB", flush=True)
        run["end"] = datetime.datetime.now().isoformat(timespec="seconds")
        all_lat = [x.get("s_per_crop_median") for x in runs if x.get("s_per_crop_median")]
        meta["s_per_crop"] = float(np.median(all_lat)) if all_lat else None
        _write_meta(meta_path, meta)


def run_timing_torch(args):
    """Batch-1 latency for one roster model: ``--n-per-type`` crops per type (``stratified_crops``) after one
    warm-up, on whatever GPU the job got. Writes ``<out-dir>/timing/<model>.csv`` and ``.meta.json``."""
    from PIL import Image

    data = pd.concat([load_split_rows(args.split_root, t) for t in LABEL_TYPES], ignore_index=True)
    crops = stratified_crops(data, args.n_per_type)
    vlm = TorchVlm(args.model)
    gpu = _gpu_info().get("gpu")
    first = crops.iloc[0]
    vlm.score_batch([build_messages(build_prompt(first.label_type), Image.open(first.image).convert("RGB"))])
    rows = []
    for r in crops.itertuples():
        s, dt = vlm.score_batch([build_messages(build_prompt(r.label_type), Image.open(r.image).convert("RGB"))])
        rows.append({"label_uid": r.label_uid, "label_type": r.label_type, "latency_s": dt,
                     "prompt_tokens": s[0]["prompt_tokens"], "p_incorrect": s[0]["p_incorrect"],
                     "log_odds_incorrect": s[0]["log_odds_incorrect"], "gpu": gpu})
    out = os.path.join(args.out_dir, "timing")
    os.makedirs(out, exist_ok=True)
    t = pd.DataFrame(rows)
    stem = timing_stem(args.model, gpu)
    t.to_csv(os.path.join(out, f"{stem}.csv"), index=False)
    meta = {"model": args.model, "repo": vlm.repo, "batch_size": 1, "n": int(len(t)), **_gpu_info(),
            "peak_memory_gb": _peak_memory_gb(), "median_s_per_crop": float(t.latency_s.median()),
            "p90_s_per_crop": float(t.latency_s.quantile(0.9)), **_env_versions()}
    _write_meta(os.path.join(out, f"{stem}.meta.json"), meta)
    print(f"{args.model}: batch-1 median {meta['median_s_per_crop']:.3f}s p90 {meta['p90_s_per_crop']:.3f}s "
          f"on {meta.get('gpu')}")


def run_export_heldout(args):
    """The scoreboard's inputs from the round-3 run root: ``<out-dir>/splits/<Type>.csv`` (``SPLIT_COLUMNS``) and
    ``<out-dir>/scores/round3/<Type>.csv`` (``label_uid, p_incorrect`` on the held-out rows). Private-city rows:
    write outside the repo only."""
    for t in LABEL_TYPES:
        s = load_split_rows(args.split_root, t)
        pred = pd.read_csv(os.path.join(args.split_root, t, "predictions.csv"))
        pred = pred[pred.label_uid.isin(s.label_uid)]
        if len(pred) != len(s):
            raise ValueError(f"{t}: {len(pred)} round-3 predictions for {len(s)} held-out rows")
        for sub, df in (("splits", s[SPLIT_COLUMNS]), ("scores/round3", pred[["label_uid", "p_incorrect"]])):
            os.makedirs(os.path.join(args.out_dir, sub), exist_ok=True)
            df.sort_values("label_uid").to_csv(os.path.join(args.out_dir, sub, f"{t}.csv"), index=False)
        print(f"{t}: {len(s)} held-out rows")


def timing_stem(model, gpu):
    """File stem of one batch-1 timing run: the model and a slug of the GPU it ran on, so runs on several GPU
    types sit side by side. ``timing_model`` inverts it.

    >>> timing_stem("molmo2-8b", "NVIDIA L40S"), timing_model("molmo2-8b__nvidia-l40s")
    ('molmo2-8b__nvidia-l40s', 'molmo2-8b')
    """
    slug = "".join(c if c.isalnum() else "-" for c in str(gpu).lower()).strip("-")
    return f"{model}__{slug}"


def timing_model(stem):
    return stem.split("__")[0]


def batch_invariance(score_dir, timing_dir):
    """Per leg: the batch-1 timing crops' scores against the same crops in the full (batched) run, as a check on
    batching that does not depend on one batch. ``model, n, run_batch_size, max_abs_dp, median_abs_dp,
    max_abs_dlogodds``; legs without p in their timing CSV are skipped."""
    rows = []
    for f in sorted(os.listdir(timing_dir)) if os.path.isdir(timing_dir) else []:
        if not f.endswith(".csv"):
            continue
        t = pd.read_csv(os.path.join(timing_dir, f))
        leg = os.path.join(score_dir, timing_model(f[:-4]))
        if "p_incorrect" not in t or not os.path.isdir(leg):
            continue
        full = pd.concat([pd.read_csv(os.path.join(leg, x)) for x in os.listdir(leg) if x.endswith(".csv")])
        m = t.merge(full, on="label_uid", suffixes=("_b1", "_run"), validate="one_to_one")
        if m.empty:
            continue
        dp = (m.p_incorrect_b1 - m.p_incorrect_run).abs()
        dl = (m.log_odds_incorrect_b1 - m.log_odds_incorrect_run).abs()
        rows.append({"model": timing_model(f[:-4]), "timing_gpu": t.gpu.iloc[0] if "gpu" in t else None,
                     "run_gpu": ",".join(sorted(m.gpu.astype(str).unique())) if "gpu" in m else None,
                     "n": len(m), "run_batch_size": int(m.batch_size.max()),
                     "max_abs_dp": dp.max(), "median_abs_dp": dp.median(), "max_abs_dlogodds": dl.max()})
    return pd.DataFrame(rows)


def timing_summary(score_dir, timing_dir):
    """``timing.csv`` rows: batch-1 runs from ``timing/<model>.csv``, and each full run's per-crop latency (its batch
    latency / batch size, every batch including the first) per leg and GPU, at the leg's largest batch size."""
    rows = []
    if os.path.isdir(timing_dir):
        for f in sorted(os.listdir(timing_dir)):
            if f.endswith(".csv"):
                t = pd.read_csv(os.path.join(timing_dir, f))
                m = _read_meta(os.path.join(timing_dir, f.replace(".csv", ".meta.json")))
                rows.append({"model": timing_model(f[:-4]), "gpu": m.get("gpu"), "batch_size": 1, "n_crops": len(t),
                             "median_s_per_crop": t.latency_s.median(), "p90_s_per_crop": t.latency_s.quantile(.9),
                             "source": "batch-1 timing run"})
    legs = sorted(os.listdir(score_dir)) if os.path.isdir(score_dir) else []
    for leg in [x for x in legs if os.path.isdir(os.path.join(score_dir, x))]:
        csvs = [os.path.join(score_dir, leg, f) for f in os.listdir(os.path.join(score_dir, leg)) if f.endswith(".csv")]
        frames = [pd.read_csv(c) for c in csvs]
        frames = [f for f in frames if {"latency_s", "batch_size", "gpu"} <= set(f.columns)]
        if not frames:
            continue
        d = pd.concat(frames)
        for gpu, g in d.groupby("gpu"):
            g = g[g.batch_size == g.batch_size.max()]
            rows.append({"model": leg, "gpu": gpu, "batch_size": int(g.batch_size.iloc[0]), "n_crops": len(g),
                         "median_s_per_crop": g.latency_s.median(), "p90_s_per_crop": g.latency_s.quantile(.9),
                         "source": "full run"})
    return pd.DataFrame(rows, columns=["model", "gpu", "batch_size", "n_crops", "median_s_per_crop",
                                       "p90_s_per_crop", "source"])


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
        pc = m.get("per_class", {}).get(c)
        row[f"recall_{c}"] = pc["recall"] if pc and pc["support"] > 0 else None
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
            gc = g[g.display_city == city]
            mwt, ntypes = mean_within_type_auc(gc)
            out.append({"model": model, "city": city, "public": pub, **metrics_row(gc, levels),
                        "mean_within_type_auc": mwt, "n_types": ntypes})
    t = pd.DataFrame(out)
    return t.sort_values(["model", "public", "n"], ascending=[True, False, False]).reset_index(drop=True)


def mean_within_type_auc(df, min_n=10):
    """Unweighted mean of the per-type AUCs in ``df`` over types with both classes and >= ``min_n`` rows, and the
    number of such types. Unlike the pooled AUC, it is not moved by a model's per-type verdict offsets combined with
    a deployment's type mix.

    >>> d = pd.DataFrame({"label_type": ["A"] * 10 + ["B"] * 10, "true": ["correct", "incorrect"] * 10,
    ...                   "p_incorrect": [0.1, 0.2] * 5 + [0.9, 0.8] * 5})
    >>> mean_within_type_auc(d)
    (0.5, 2)
    """
    from sklearn.metrics import roc_auc_score
    aucs = [roc_auc_score(g.true == "incorrect", g.p_incorrect) for _, g in df.groupby("label_type")
            if len(g) >= min_n and g.true.nunique() == 2]
    return (float(np.mean(aucs)), len(aucs)) if aucs else (None, 0)


def per_city_type_table(pred, cities, min_n=20, levels=REPORT_LEVELS):
    """Per model x displayed city x label type: public cities with >= ``min_n`` rows of that type, plus one pooled
    private row per type at any size."""
    p = pred.assign(display_city=display_city(pred.city, cities).values,
                    public=is_public(pred.city, cities).values)
    out = []
    for (model, t, city), g in p.groupby(["model", "label_type", "display_city"]):
        pub = bool(g.public.iloc[0])
        if pub and len(g) < min_n:
            continue
        out.append({"model": model, "label_type": t, "city": city, "public": pub, **metrics_row(g, levels)})
    t = pd.DataFrame(out)
    if t.empty:
        return t
    return t.sort_values(["label_type", "model", "public", "n"], ascending=[True, True, False, False]
                         ).reset_index(drop=True)


def within_city_auc(y, score, city, min_n=10):
    """City-stratified AUC: per-city AUCs averaged with weights n_pos x n_neg (the number of incorrect/correct pairs
    each city contributes), over cities with both classes and >= ``min_n`` rows. ``(auc, n_cities)``; NaN if none.
    It cannot be raised by a score that only separates cities with many incorrect labels from cities with few.

    >>> y = [0, 1] * 10 + [1] * 10; s = list(range(20)) + [100] * 10; c = ["a"] * 20 + ["b"] * 10
    >>> within_city_auc(y, s, c)   # city b has one class, so only city a counts
    (0.55, 1)
    """
    from sklearn.metrics import roc_auc_score
    d = pd.DataFrame({"y": np.asarray(y).astype(int), "s": np.asarray(score, float), "c": np.asarray(city)})
    aucs, ws = [], []
    for _, g in d.groupby("c"):
        if len(g) < min_n or g.y.nunique() < 2:
            continue
        aucs.append(roc_auc_score(g.y, g.s))
        ws.append(int(g.y.sum()) * int((1 - g.y).sum()))
    if not aucs:
        return float("nan"), 0
    return float(np.average(aucs, weights=ws)), len(aucs)


def cluster_bootstrap_auc(y, score_a, score_b, groups, n_boot=1000, seed=0, max_tries=20):
    """Paired bootstrap that resamples whole deployments (``groups``) with replacement, for intervals that hold for
    other deployments rather than other labels from these ones. Draws without both classes are redrawn.

    Returns ``{"a_lo", "a_hi", "b_lo", "b_hi", "diff_lo", "diff_hi", "n_groups"}`` (95% percentile intervals).
    """
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y).astype(int)
    a, b = np.asarray(score_a, dtype=float), np.asarray(score_b, dtype=float)
    groups = np.asarray(groups)
    names = np.unique(groups)
    index = {g: np.flatnonzero(groups == g) for g in names}
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        for _try in range(max_tries):
            idx = np.concatenate([index[g] for g in rng.choice(names, len(names))])
            if 0 < y[idx].sum() < len(idx):
                boots.append((roc_auc_score(y[idx], a[idx]), roc_auc_score(y[idx], b[idx])))
                break
    boots = np.array(boots)
    q = lambda x: np.percentile(x, [2.5, 97.5])  # noqa: E731
    (alo, ahi), (blo, bhi), (dlo, dhi) = q(boots[:, 0]), q(boots[:, 1]), q(boots[:, 0] - boots[:, 1])
    return {"a_lo": alo, "a_hi": ahi, "b_lo": blo, "b_hi": bhi, "diff_lo": dlo, "diff_hi": dhi,
            "n_groups": int(len(names))}


def split_diff_bootstrap(df, n_boot=1000, seed=0):
    """AUC on ``test`` minus AUC on ``test_heldout`` for one model, with a 95% interval from resampling labels
    within split and class. ``{"auc_test", "auc_heldout", "diff", "diff_lo", "diff_hi", "n_test", "n_heldout"}``."""
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(seed)
    parts = {}
    for split in ("test", "test_heldout"):
        g = df[df.split == split]
        parts[split] = ((g.true == "incorrect").to_numpy().astype(int), g.p_incorrect.to_numpy(float))
    point = {k: roc_auc_score(*v) for k, v in parts.items()}

    def draw(y, s):
        pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        return roc_auc_score(y[idx], s[idx])
    d = np.array([draw(*parts["test"]) - draw(*parts["test_heldout"]) for _ in range(n_boot)])
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"auc_test": point["test"], "auc_heldout": point["test_heldout"],
            "diff": point["test"] - point["test_heldout"], "diff_lo": lo, "diff_hi": hi,
            "n_test": int(len(parts["test"][0])), "n_heldout": int(len(parts["test_heldout"][0]))}


def timing_table(timing_dir):
    """Per model: n, p10/median/p90 seconds per crop (and per reason), and the load average around the run."""
    rows = []
    if not os.path.isdir(timing_dir):
        return pd.DataFrame()
    for f in sorted(os.listdir(timing_dir)):
        if not f.endswith(".csv"):
            continue
        t = pd.read_csv(os.path.join(timing_dir, f))
        meta = json.load(open(os.path.join(timing_dir, f.replace(".csv", ".meta.json"))))
        row = {"model": f[:-4], "n": len(t), "p10_s": t.latency_s.quantile(0.1), "median_s": t.latency_s.median(),
               "p90_s": t.latency_s.quantile(0.9), "load_1min_before": (meta.get("load_before") or [None])[0],
               "load_1min_after": (meta.get("load_after") or [None])[0], "date": meta.get("date")}
        if "reason_latency_s" in t:
            row.update(reason_p10_s=t.reason_latency_s.quantile(0.1), reason_median_s=t.reason_latency_s.median(),
                       reason_p90_s=t.reason_latency_s.quantile(0.9))
        if "prompt_tokens" in t:
            row["prompt_tokens_median"] = t.prompt_tokens.median()
        rows.append(row)
    return pd.DataFrame(rows)


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


def check_coverage(main_all, subset):
    """Raise if a VLM has scored some but not all of the headline subset: a partial run would otherwise drop out of
    the tables silently. Production is checked the same way."""
    partial = [m for m, g in main_all.groupby("model")
               if len(subset & set(g.label_uid)) and not subset <= set(g.label_uid)]
    if partial:
        raise ValueError(f"partial coverage of the headline subset for {partial}: finish their `score` runs, or pass "
                         "the --subset they were scored with")


def run_report(args):
    sample = load_sample(args.sample_dir)
    cities = load_cities(args.cities_json)
    score_dir = os.path.join(args.out_dir, "scores")
    files = sorted(f for f in os.listdir(score_dir) if f.endswith(".csv"))
    scores = pd.concat([pd.read_csv(os.path.join(score_dir, f)) for f in files], ignore_index=True)
    pred = predictions(sample, scores)
    main_all = pred[pred.variant == "definition"]
    subset = set(balanced_subset(sample, args.subset).label_uid) if args.subset else set(sample.label_uid)
    check_coverage(main_all, subset)
    complete = [m for m, g in main_all.groupby("model") if subset <= set(g.label_uid)]
    main = main_all[main_all.model.isin(complete) & main_all.label_uid.isin(subset)]
    full = [m for m, g in main_all.groupby("model") if g.label_uid.nunique() == len(sample)]
    full_all = main_all[main_all.model.isin(full)]
    sets = {"subset": main, "full": full_all}
    os.makedirs(args.doc_dir, exist_ok=True)
    w = lambda df, name: df.to_csv(os.path.join(args.doc_dir, name), index=False, float_format="%.4f")  # noqa: E731

    def per_set(fn):
        return pd.concat([fn(d).assign(set=k) for k, d in sets.items() if len(d)], ignore_index=True)

    headline = metrics_table(main, ["model", "label_type"])
    pooled = metrics_table(main, ["model"]).assign(label_type="all (pooled)")
    w(pd.concat([headline, pooled], ignore_index=True), "headline.csv")
    full_tab = pd.concat([metrics_table(full_all, ["model", "label_type"]),
                          metrics_table(full_all, ["model"]).assign(label_type="all (pooled)")], ignore_index=True)
    w(full_tab, "full_sample.csv")
    by_split = per_set(lambda d: pd.concat([metrics_table(d, ["model", "label_type", "split"]),
                                            metrics_table(d, ["model", "split"]).assign(label_type="all (pooled)")],
                                           ignore_index=True))
    w(by_split, "by_split.csv")

    def split_rows(d):
        out = []
        for m, g in d.groupby("model"):
            for t in list(LABEL_TYPES) + ["all (pooled)"]:
                gt = g if t == "all (pooled)" else g[g.label_type == t]
                out.append({"model": m, "label_type": t, **split_diff_bootstrap(gt, args.n_boot, args.seed)})
        return pd.DataFrame(out)
    w(per_set(split_rows), "split_diff.csv")

    def within_rows(d):
        out = []
        for (m, split), g in d.groupby(["model", "split"]):
            for t in ["all (pooled)"] + list(LABEL_TYPES):
                gt = g if t == "all (pooled)" else g[g.label_type == t]
                wc, nc = within_city_auc(gt.true == "incorrect", gt.p_incorrect, gt.city, args.within_min_n)
                out.append({"model": m, "split": split, "label_type": t, "auc_within_city": wc, "n_cities": nc,
                            "n": len(gt)})
        return pd.DataFrame(out)
    w(per_set(within_rows), "within_city.csv")

    hard = per_set(lambda d: metrics_table(d[d.hard == 1], ["model"]))
    w(hard, "hard.csv")
    w(per_set(lambda d: per_city_table(d, cities, min_n=args.min_city_n)), "per_city.csv")
    w(per_set(lambda d: per_city_type_table(d, cities, min_n=args.min_city_type_n)), "per_city_type.csv")

    lat = timing_table(os.path.join(args.out_dir, "timing"))
    if len(lat):
        sizes = {}
        for m in lat.model:
            if m == PRODUCTION_NAME:
                sz = [hf_snapshot(PRODUCTION[t][0])[1] for t in LABEL_TYPES]
                sizes[m] = sum(sz) / len(sz) if all(sz) else None
            else:
                sizes[m] = hf_snapshot(MODELS[m])[1]
        lat["size_gb"] = lat.model.map(lambda m: sizes.get(m) / 1e9 if sizes.get(m) else None)
    w(lat, "latency.csv")
    mass = scores[scores.label_uid.isin(subset) & (scores.variant == "definition")
                  & (scores.model != PRODUCTION_NAME)].groupby("model").answer_mass.median()
    w(mass.rename("answer_mass_median").reset_index(), "answer_mass.csv")

    best = best_model(headline)
    boot = []
    for set_name, d in sets.items():
        if best not in set(d.model) or PRODUCTION_NAME not in set(d.model):
            continue
        for t in list(LABEL_TYPES) + ["all (pooled)"]:
            dt = d if t == "all (pooled)" else d[d.label_type == t]
            a = dt[dt.model == best].set_index("label_uid").sort_index()
            b = dt[dt.model == PRODUCTION_NAME].set_index("label_uid").loc[a.index]
            y = a.true == "incorrect"
            r = paired_bootstrap_auc(y, a.p_incorrect, b.p_incorrect, args.n_boot, args.seed)
            c = cluster_bootstrap_auc(y, a.p_incorrect, b.p_incorrect, a.city, args.n_boot, args.seed)
            boot.append({"set": set_name, "label_type": t, "model_a": best, "model_b": PRODUCTION_NAME, "n": len(a),
                         **r, **{f"city_{k}": v for k, v in c.items()}})
    boot = pd.DataFrame(boot)
    w(boot, "bootstrap_auc.csv")

    abl = pred[pred.label_uid.isin(pred[pred.variant == "no-definition"].label_uid)]
    abl_models = abl[abl.variant == "no-definition"].model.unique()
    abl = abl[abl.model.isin(abl_models) & (abl.model != PRODUCTION_NAME)]
    ablation = metrics_table(abl, ["model", "label_type", "variant"]) if len(abl) else pd.DataFrame()
    w(ablation, "ablation.csv")

    greedy_dir = os.path.join(args.out_dir, "greedy")
    greedy = []
    if os.path.isdir(greedy_dir):
        for f in sorted(os.listdir(greedy_dir)):
            c = pd.read_csv(os.path.join(greedy_dir, f))
            for t, g in [("all", c)] + list(c.groupby("label_type")):
                greedy.append({"model": f[:-4], "label_type": t, "n": len(g),
                               "agree": int((g.agrees == True).sum()),  # noqa: E712
                               "unparsable": int(g.agrees.isna().sum())})
    greedy = pd.DataFrame(greedy)
    w(greedy, "greedy_check.csv")

    models = [PRODUCTION_NAME] + [m for m in MODELS if m in complete]
    roc_figure(main, models, os.path.join(args.doc_dir, "roc.png"))

    cols = ["model", "label_type", "n", "auc", "balanced_accuracy", "r_incorrect_p95", "r_incorrect_p92",
            "r_correct_p95", "r_correct_p92"]
    tables = {"by split": by_split, "split difference (test - heldout)": pd.read_csv(f"{args.doc_dir}/split_diff.csv"),
              "within-city AUC": pd.read_csv(f"{args.doc_dir}/within_city.csv"),
              "hard (every subset row is incorrect: balanced accuracy there is recall of incorrect)": hard,
              "per city": pd.read_csv(f"{args.doc_dir}/per_city.csv"),
              "per city x type": pd.read_csv(f"{args.doc_dir}/per_city_type.csv"), "bootstrap": boot,
              "latency (timing runs)": lat, "ablation": ablation, "greedy check": greedy}
    with open(os.path.join(args.doc_dir, "report.md"), "w") as f:
        f.write(f"# VLM baseline report (generated {datetime.date.today()})\n\nbest VLM by mean per-type AUC on the "
                f"subset: {best}\n\nheadline set: {len(subset)} crops (balanced_subset per_type={args.subset}); "
                f"models on it: {', '.join(complete)}; on the full sample: {', '.join(full)}\n\n"
                f"## headline (subset)\n\n{markdown(pd.concat([headline, pooled])[cols])}\n\n"
                f"## full sample\n\n{markdown(full_tab[cols])}\n\n")
        for name, t in tables.items():
            f.write(f"## {name}\n\n{markdown(t) if len(t) else '(none)'}\n\n")
    write_provenance(args, sample, files, score_dir, sorted(set(complete) | set(full)))
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
        "venvs": _venvs(args.out_dir),
        "timing": _hash_dir(os.path.join(args.out_dir, "timing")),
        "greedy": _hash_dir(os.path.join(args.out_dir, "greedy")),
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
    prov = {**{k: v for k, v in old.items() if k in ("packet", "explain", "reuse_check")}, **prov}
    with open(pp, "w") as f:
        json.dump(prov, f, indent=1, sort_keys=True)


def _hash_dir(d):
    """``{file: {"sha256", "rows"}}`` for the CSVs in ``d`` (empty if absent)."""
    if not os.path.isdir(d):
        return {}
    return {f: {"sha256": sha256_of(os.path.join(d, f)), "rows": int(len(pd.read_csv(os.path.join(d, f))))}
            for f in sorted(os.listdir(d)) if f.endswith(".csv")}


def _venvs(out_dir):
    """The two environments: this one (VLM: mlx-vlm, transformers 5.x) and the production one (transformers 4.55,
    read from the production run's meta files, which that venv wrote)."""
    prod = {}
    for path in (os.path.join(out_dir, "scores", f"{PRODUCTION_NAME}.meta.json"),
                 os.path.join(out_dir, "timing", f"{PRODUCTION_NAME}.meta.json")):
        if os.path.exists(path):
            m = json.load(open(path))
            prod.update({k: m[k] for k in ("torch", "transformers", "device") if k in m})
    return {"vlm (.venv)": {"python": platform.python_version(), **_mlx_versions()},
            "production (.venv-prod, requirements.txt)": prod}


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

CATEGORIES = ("vlm_wrong_production_right", "vlm_wrong_production_wrong", "vlm_right_production_wrong",
              "vlm_right_production_right")
"""Item categories: whether the VLM's argmax verdict matches the crowd, crossed with production's."""


def item_category(vlm_wrong, prod_wrong):
    """``vlm_<right|wrong>_production_<right|wrong>`` per row.

    >>> item_category(np.array([True, False]), np.array([False, True])).tolist()
    ['vlm_wrong_production_right', 'vlm_right_production_wrong']
    """
    v = np.where(np.asarray(vlm_wrong), "vlm_wrong", "vlm_right")
    q = np.where(np.asarray(prod_wrong), "production_wrong", "production_right")
    return np.char.add(np.char.add(v.astype(str), "_"), q.astype(str))


def select_explanation_items(best_pred, prod_pred, cities, per_type=20, seed=0, keep=()):
    """Rows to explain: public deployments only, ``per_type`` per label type, chosen so that neither the VLM's answer
    nor its correctness says anything about the crowd verdict: for each type and each answer (yes / no), equal
    numbers of rows where the VLM is right and where it is wrong. So P(VLM right) = 0.5 per type, and P(crowd says
    correct | answer) = 0.5 per type and answer.

    Each type's quota is split evenly between the two answers; an answer with too few right/wrong pairs gives its
    remainder to the other one, and a type that still falls short is reported, not padded. Within each cell, rows in
    ``keep`` (``label_uid`` values whose reasons can be reused) come first, then rows where production is wrong (the
    comparison the issue asks for), then a seeded random draw. Adds ``category`` (``CATEGORIES``), ``vlm_correct``
    and ``vlm_answer``.
    """
    m = best_pred.merge(prod_pred[["label_uid", "p_incorrect"]], on="label_uid", suffixes=("", "_prod"),
                        validate="one_to_one")
    m = m[is_public(m.city, cities).values].sort_values("label_uid").reset_index(drop=True)
    verdict = verdict_from_score(m.p_incorrect)
    vlm_wrong = verdict != m.true.to_numpy()
    prod_wrong = verdict_from_score(m.p_incorrect_prod) != m.true.to_numpy()
    m["category"] = item_category(vlm_wrong, prod_wrong)
    m["vlm_correct"] = ~vlm_wrong
    m["vlm_answer"] = np.where(verdict == "incorrect", "No", "Yes")
    m["prio"] = (m.label_uid.isin(set(keep)).astype(int) * 2 + prod_wrong.astype(int))
    m["draw"] = np.random.default_rng(seed).permutation(len(m))
    picks = []
    for t in LABEL_TYPES:
        cells = {(a, c): g.sort_values(["prio", "draw"], ascending=[False, True])
                 for (a, c), g in m[m.label_type == t].groupby(["vlm_answer", "vlm_correct"])}
        avail = {a: min(len(cells.get((a, True), [])), len(cells.get((a, False), []))) for a in ("No", "Yes")}
        pairs = per_type // 2
        base = {"No": pairs - pairs // 2, "Yes": pairs // 2}
        take = {a: min(avail[a], base[a]) for a in base}
        left = pairs - sum(take.values())
        for a in ("No", "Yes"):                       # an answer short of pairs hands its quota to the other
            extra = min(avail[a] - take[a], left)
            take[a] += extra
            left -= extra
        for a in ("No", "Yes"):
            for c in (True, False):
                if take[a]:
                    picks.append(cells[(a, c)].head(take[a]))
    return pd.concat(picks, ignore_index=True).drop(columns=["prio", "draw"])


KEY_COLUMNS = ["packet_id", "label_uid", "label_type", "category", "vlm_correct", "label_class", "vlm_model",
               "vlm_answer", "vlm_reason", "p_incorrect", "p_incorrect_prod"]
PASS1_COLUMNS = ["packet_id", "label_type", "my_verdict", "notes_pass1"]
PASS2_COLUMNS = ["packet_id", "label_type", "vlm_answer", "vlm_reason", "reason_correct", "would_change_my_vote",
                 "notes_pass2"]
RUBRIC = {"my_verdict": ("correct", "incorrect", "cant_tell"), "reason_correct": ("yes", "partly", "no"),
          "would_change_my_vote": ("yes", "no")}


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


def blank_sheets(key):
    """``(pass1, pass2)`` in packet order. Pass 1 (bare crops) asks only for the rater's own verdict; pass 2
    (annotated crops) shows the VLM's answer and reason and asks about them."""
    pass1 = pd.DataFrame({"packet_id": key.packet_id, "label_type": key.label_type})
    pass2 = pass1.assign(vlm_answer=key.vlm_answer.values, vlm_reason=key.vlm_reason.values)
    for c in PASS1_COLUMNS[2:]:
        pass1[c] = ""
    for c in PASS2_COLUMNS[4:]:
        pass2[c] = ""
    return pass1[PASS1_COLUMNS], pass2[PASS2_COLUMNS]


def validate_codes(sheet):
    """Raise if a filled sheet uses a value outside the rubric (blank cells are allowed); checks the columns present."""
    for col, values in RUBRIC.items():
        if col not in sheet:
            continue
        bad = set(sheet[col].fillna("").astype(str).str.strip().str.lower()) - set(values) - {""}
        if bad:
            raise ValueError(f"{col}: values {sorted(bad)} are not in the rubric ({', '.join(values)})")


def unblind(pass1, pass2, key):
    """Join the two filled sheets to the key; codes lower-cased and stripped. Raises on ids missing from the key or
    from either sheet."""
    a = pass1[["packet_id", "my_verdict"]].copy()
    b = pass2[["packet_id", "reason_correct", "would_change_my_vote"]].copy()
    m = a.merge(b, on="packet_id", how="outer", validate="one_to_one", indicator=True)
    if (m._merge != "both").any():
        raise ValueError("the two sheets do not hold the same packet ids")
    m = m.drop(columns="_merge")
    for c in RUBRIC:
        m[c] = m[c].fillna("").astype(str).str.strip().str.lower()
    out = m.merge(key, on="packet_id", how="inner", validate="one_to_one")
    if len(out) != len(m):
        raise ValueError("sheet has packet ids that are not in the key")
    return out


def agreement_tables(m):
    """From unblinded sheets: the rater's pass-1 verdict against the VLM and the crowd, and the pass-2 reason codes,
    overall, by whether the VLM was right, and per category. Verdicts left blank or ``cant_tell`` are excluded from
    the agreement rates and counted (``n_rated``)."""
    vlm_verdict = np.where(m.vlm_answer.astype(str).str.lower() == "no", "incorrect", "correct")
    r = m.assign(vlm_verdict=vlm_verdict, rated=m.my_verdict.isin(["correct", "incorrect"]),
                 vlm_right=np.where(vlm_verdict == m.label_class, "vlm right", "vlm wrong"))
    groups = ([("all", "all", r)] + [("vlm", k, g) for k, g in r.groupby("vlm_right")]
              + [("category", k, g) for k, g in r.groupby("category")])
    rows = []
    for by, name, g in groups:
        rated = g[g.rated]
        rate = lambda x: float(x.mean()) if len(x) else None  # noqa: E731
        rows.append({"by": by, "group": name, "n": len(g), "n_rated": len(rated),
                     "rater_agrees_vlm": rate(rated.my_verdict == rated.vlm_verdict),
                     "rater_agrees_crowd": rate(rated.my_verdict == rated.label_class),
                     "reason_correct_yes": float((g.reason_correct == "yes").mean()),
                     "reason_correct_partly": float((g.reason_correct == "partly").mean()),
                     "reason_correct_no": float((g.reason_correct == "no").mean()),
                     "would_change_vote_yes": float((g.would_change_my_vote == "yes").mean())})
    return pd.DataFrame(rows)


def inside_this_repo(path):
    """True if ``path`` would land in this checkout or any of its git worktrees (they share one git directory).
    A separate clone of the repo elsewhere is not detected."""
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
    reuse = pd.read_csv(args.reuse) if args.reuse else pd.DataFrame(columns=["label_uid", "vlm_model", "vlm_reason"])
    reuse = reuse[reuse.vlm_model == args.model].set_index("label_uid")
    items = select_explanation_items(pred[pred.model == args.model], pred[pred.model == PRODUCTION_NAME], cities,
                                     args.per_type, args.seed, keep=reuse.index)
    print(items.groupby(["label_type", "category"]).size().to_string())
    print(f"{len(items)} items, {int(items.label_uid.isin(reuse.index).sum())} reasons reused from {args.reuse}")
    vlm = None
    img = dict(zip(sample.label_uid, sample.image))
    out = []
    for r in items.itertuples():
        answer = "No." if r.p_incorrect > 0.5 else "Yes."
        if r.label_uid in reuse.index:
            prev = reuse.loc[r.label_uid]
            reason, latency, source = prev.vlm_reason, prev.latency_s, "reused"
        else:
            if vlm is None:
                vlm = MlxVlm(MODELS[args.model])
            q = build_prompt(r.label_type)
            reason, latency = vlm.generate(img[r.label_uid], [("user", q), ("assistant", answer),
                                                                ("user", WHY_PROMPT)], max_tokens=args.max_tokens)
            source = "generated"
        out.append({"label_uid": r.label_uid, "label_type": r.label_type, "category": r.category,
                    "vlm_correct": bool(r.vlm_correct), "label_class": r.true, "vlm_model": args.model,
                    "vlm_answer": answer.rstrip("."), "vlm_reason": reason, "p_incorrect": r.p_incorrect,
                    "p_incorrect_prod": r.p_incorrect_prod, "latency_s": latency, "n_words": len(str(reason).split()),
                    "reason_source": source})
    path = os.path.join(args.out_dir, "explanations.csv")
    pd.DataFrame(out).to_csv(path, index=False)
    print(f"wrote {len(out)} explanations to {path}")


def render_item(image_path, title, answer, reason, out_path, width=640):
    """One rater image: the crop with a small cyan ring at the centre (the label point) and a caption panel. With
    ``answer`` and ``reason`` None the caption holds only ``title`` (the pass-1 bare item)."""
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
    lines = [title]
    if answer is not None:
        lines += [f"VLM answer: {answer}"] + textwrap.wrap(f"VLM reason: {reason}", 62)
    panel = Image.new("RGB", (width, 28 * len(lines) + 16), (255, 255, 255))
    pd_ = ImageDraw.Draw(panel)
    for i, line in enumerate(lines):
        pd_.text((10, 8 + 28 * i), line, fill=(0, 0, 0), font=font)
    out = Image.new("RGB", (width, im.height + panel.height), (255, 255, 255))
    out.paste(im, (0, 0))
    out.paste(panel, (0, im.height))
    out.save(out_path, quality=90)


PACKET_ZIPS = {1: "vlm-explanations-packet-pass1.zip", 2: "vlm-explanations-packet-pass2.zip"}


def run_packet(args, parser):
    """Two packets with the same items, ids and key: ``<packet-dir>/pass1/`` (bare crops, pass-1 sheet and rubric)
    and ``<packet-dir>/pass2/`` (annotated crops, pass-2 sheet and rubric), each also zipped into ``--out-dir``
    (``PACKET_ZIPS``). Hand out the pass-2 zip only after the filled pass-1 sheet has come back."""
    if inside_this_repo(args.key):
        parser.error("--key must be outside every checkout of this repo until the rating is done "
                     "(commit only <key>.sha256)")
    ex = pd.read_csv(os.path.join(args.out_dir, "explanations.csv"))
    sample = load_sample(args.sample_dir)
    img = dict(zip(sample.label_uid, sample.image))
    key = packet_key(ex, args.seed)
    dirs = {1: os.path.join(args.packet_dir, "pass1"), 2: os.path.join(args.packet_dir, "pass2")}
    os.makedirs(os.path.join(dirs[1], "items_bare"), exist_ok=True)
    os.makedirs(os.path.join(dirs[2], "items_annotated"), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.key)), exist_ok=True)
    for i, k in enumerate(key.itertuples()):
        title = f"{i + 1:03d}  {k.packet_id}  label type: {TYPE_NAMES[k.label_type]}"
        name = f"{i + 1:03d}_{k.packet_id}.jpg"
        render_item(img[k.label_uid], title, None, None, os.path.join(dirs[1], "items_bare", name))
        render_item(img[k.label_uid], title, k.vlm_answer, k.vlm_reason,
                    os.path.join(dirs[2], "items_annotated", name))
    pass1, pass2 = blank_sheets(key)
    pass1.to_csv(os.path.join(dirs[1], "pass1_sheet.csv"), index=False)
    pass2.to_csv(os.path.join(dirs[2], "pass2_sheet.csv"), index=False)
    key[KEY_COLUMNS].to_csv(args.key, index=False)
    digest = sha256_of(args.key)
    write_rating_pages(dirs, pass1, pass2, digest)
    zips = zip_packet(dirs, args.out_dir)
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
                          "vlm_correct_share_by_type": {t: float(g.vlm_correct.mean())
                                                        for t, g in ex.groupby("label_type")},
                          "reason_source": {k: int(v) for k, v in ex.reason_source.value_counts().items()},
                          "vlm_model": sorted(ex.vlm_model.unique()), "why_prompt": WHY_PROMPT,
                          "median_reason_words": float(ex.n_words.median()), "passes": 2,
                          "zips": {PACKET_ZIPS[n]: zip_contents(dirs[n]) for n in dirs}}
        with open(pp, "w") as f:
            json.dump(prov, f, indent=1, sort_keys=True)
    print(f"wrote {len(key)} items: pass 1 {zips[1]}, pass 2 {zips[2]} (hand out after the pass-1 sheet is back); "
          f"key {args.key} (sha256 {digest})")


def write_rating_pages(dirs, pass1, pass2, key_digest):
    """``RUBRIC.md`` and ``rate.html`` (``analysis/rating_page.py``) in each packet directory, from the blank sheets
    in packet order. The pass-1 page holds only the bare crops and label types; the pass-2 page adds the VLM's
    answer and reason as text and stays closed until a complete pass-1 sheet is loaded. Each page exports its sheet
    in the blank sheet's format, so ``agreement`` reads it unchanged. Answers are kept in the browser under a key
    that includes the start of ``key_digest``, so two packets never share saved answers."""
    from analysis.rating_page import render_page

    def items(sheet, subdir, extra=None):
        return [{"id": r.packet_id, "image": f"{subdir}/{i + 1:03d}_{r.packet_id}.jpg",
                 "alt": f"Item {i + 1}, label type {TYPE_NAMES[r.label_type]}",
                 "caption_html": f"{i + 1}. {r.packet_id} &middot; label type: <strong>{TYPE_NAMES[r.label_type]}"
                                 "</strong>",
                 "extra_html": extra(r) if extra else None}
                for i, r in enumerate(sheet.itertuples())]

    def fixed(sheet, cols):
        return {r["packet_id"]: {c: "" if pd.isna(r[c]) else r[c] for c in cols} for r in sheet.to_dict("records")}

    def vlm_text(r):
        return (f"<p><strong>Model's answer:</strong> {html.escape(str(r.vlm_answer))}</p>"
                f"<p><strong>Its reason:</strong> {html.escape(str(r.vlm_reason))}</p>")

    tag = key_digest[:8]
    pages = {
        1: render_page(
            title="Rate label crops, pass 1", heading="Rating label crops, pass 1 of 2",
            intro_html="<p>Each card is a label crop. The labelled point is the centre, marked with a cyan ring, "
                       "and the caption gives the label type. Do not look anything up; judge from the crop alone. "
                       "When every card is done, export the sheet and send it back; the material for pass 2 is "
                       "handed out after that.</p>",
            items=items(pass1, "items_bare"),
            fields=[{"name": "my_verdict", "kind": "radio",
                     "legend": "Is this a correct label of the stated type at the ringed point?",
                     "options": [("correct", "Correct", "c"), ("incorrect", "Incorrect", "i"),
                                 ("cant_tell", "Can't tell", "t")]},
                    {"name": "notes_pass1", "kind": "text", "legend": "Notes (optional)"}],
            sheet_columns=PASS1_COLUMNS, fixed=fixed(pass1, ["label_type"]),
            storage_key=f"vlm16_{tag}_pass1", export_name="pass1_sheet"),
        2: render_page(
            title="Rate VLM explanations, pass 2", heading="Rating the VLM explanations, pass 2 of 2",
            intro_html="<p>The same crops as pass 1 (same item ids), now with a vision-language model's yes/no "
                       "answer to &ldquo;Is this a correct &lt;type&gt; label?&rdquo; and its reason (&ldquo;In at "
                       "most 30 words, why?&rdquo;). Your pass-1 sheet is final; do not change it. Rate the reason "
                       "on its own: judge whether it describes the crop accurately, whatever you think of the "
                       "answer.</p>",
            items=items(pass2, "items_annotated", vlm_text),
            fields=[{"name": "reason_correct", "kind": "radio",
                     "legend": "Is the reason a true description of what the crop shows, and does it bear on the "
                               "question? (Partly: some of it is right, or it is right but misses the deciding "
                               "feature.)",
                     "options": [("yes", "Yes", "y"), ("partly", "Partly", "p"), ("no", "No", "n")]},
                    {"name": "would_change_my_vote", "kind": "radio",
                     "legend": "Had you been validating this label, would reading this answer and reason have "
                               "changed your vote?",
                     "options": [("yes", "Yes", "1"), ("no", "No", "0")]},
                    {"name": "notes_pass2", "kind": "text", "legend": "Notes (optional)"}],
            sheet_columns=PASS2_COLUMNS, fixed=fixed(pass2, ["label_type", "vlm_answer", "vlm_reason"]),
            storage_key=f"vlm16_{tag}_pass2", export_name="pass2_sheet",
            gate={"column": "my_verdict", "values": list(RUBRIC["my_verdict"]),
                  "html": "<p><strong>Load your filled pass-1 sheet to open the cards.</strong> The page only "
                          "checks that every item has a verdict; it does not store or show them.</p>"}),
    }
    for n, text in ((1, RUBRIC_PASS1_TEXT), (2, RUBRIC_PASS2_TEXT)):
        with open(os.path.join(dirs[n], "RUBRIC.md"), "w") as f:
            f.write(text)
        with open(os.path.join(dirs[n], "rate.html"), "w") as f:
            f.write(pages[n])


def zip_packet(dirs, out_dir):
    """Zip each packet directory into ``out_dir`` as ``PACKET_ZIPS``; returns ``{pass: zip path}``."""
    import shutil

    return {n: shutil.make_archive(os.path.join(out_dir, PACKET_ZIPS[n][:-4]), "zip", root_dir=d)
            for n, d in dirs.items()}


def run_gallery(args):
    """Add ``rate.html`` to an existing packet (``write_rating_pages``) and re-zip it, leaving the key, ids, sheets
    and crops unchanged. Refuses unless the key matches ``--key-sha256`` and both blank sheets hold the key's ids in
    the key's order with the expected columns."""
    if sha256_of(args.key) != open(args.key_sha256).read().split()[0]:
        raise SystemExit(f"{args.key} does not match {args.key_sha256}")
    key = pd.read_csv(args.key, dtype=str, keep_default_na=False)
    dirs = {1: os.path.join(args.packet_dir, "pass1"), 2: os.path.join(args.packet_dir, "pass2")}
    sheets = {}
    for n, cols, sub in ((1, PASS1_COLUMNS, "items_bare"), (2, PASS2_COLUMNS, "items_annotated")):
        s = pd.read_csv(os.path.join(dirs[n], f"pass{n}_sheet.csv"), dtype=str, keep_default_na=False)
        if list(s.columns) != cols:
            raise SystemExit(f"pass{n}_sheet.csv has columns {list(s.columns)}, expected {cols}")
        if list(s.packet_id) != list(key.packet_id):
            raise SystemExit(f"pass{n}_sheet.csv does not hold the key's packet ids in the key's order")
        fixed = [c for c in cols[1:] if c in key]
        if not (s[fixed] == key[fixed]).all().all():
            raise SystemExit(f"pass{n}_sheet.csv: {', '.join(fixed)} differ from the key")
        answers = [c for c in cols if c not in key]
        if (s[answers] != "").any().any():
            raise SystemExit(f"pass{n}_sheet.csv is not blank ({', '.join(answers)}); rebuild from a blank packet")
        missing = [p for i, p in enumerate(s.packet_id)
                   if not os.path.exists(os.path.join(dirs[n], sub, f"{i + 1:03d}_{p}.jpg"))]
        if missing:
            raise SystemExit(f"pass {n}: {len(missing)} crops missing, e.g. {missing[0]}")
        sheets[n] = s
    digest = sha256_of(args.key)
    write_rating_pages(dirs, sheets[1], sheets[2], digest)
    zips = zip_packet(dirs, args.out_dir)
    pp = os.path.join(args.doc_dir, "provenance.json") if args.doc_dir else None
    if pp and os.path.exists(pp):
        prov = json.load(open(pp))
        if prov.get("packet", {}).get("key_sha256") == digest:
            prov["packet"]["zips"] = {PACKET_ZIPS[n]: zip_contents(dirs[n]) for n in dirs}
            with open(pp, "w") as f:
                json.dump(prov, f, indent=1, sort_keys=True)
            print(f"refreshed the packet's zip contents in {pp}")
        else:
            print(f"left {pp} alone: its packet entry is for a different key")
    print(f"wrote rate.html to {dirs[1]} and {dirs[2]}; re-zipped {zips[1]} and {zips[2]} (key sha256 {digest})")


def zip_contents(d):
    """``{"files": [non-image files], "images": {subdir: count}}`` of one packet directory, for provenance."""
    files, images = [], {}
    for root, _, names in os.walk(d):
        rel = os.path.relpath(root, d)
        for n in names:
            if n.endswith(".jpg"):
                images[rel] = images.get(rel, 0) + 1
            else:
                files.append(os.path.normpath(os.path.join(rel, n)))
    return {"files": sorted(files), "images": images}


RUBRIC_PASS1_TEXT = """# Rating label crops, pass 1 of 2 (issue #16)

Each image in `items_bare/` is a label crop. The labelled point is the centre, marked with a cyan ring, and the
caption gives the label type. Do not look anything up; judge from the crop alone. Fill `pass1_sheet.csv` and send
it back; the material for pass 2 is handed out after that. To rate in a browser instead, extract the zip and open `rate.html`:
it saves as you go and exports the filled sheet.

| column | values | meaning |
|---|---|---|
| `my_verdict` | `correct` / `incorrect` / `cant_tell` | Is this a correct label of the stated type at the ringed point? |
| `notes_pass1` | free text | Optional. |
"""

RUBRIC_PASS2_TEXT = """# Rating the VLM explanations, pass 2 of 2 (issue #16)

The same crops as pass 1 (same item ids), now in `items_annotated/` with a vision-language model's yes/no answer
to "Is this a correct <type> label?" and its reason ("In at most 30 words, why?"). Your pass-1 sheet is final; do
not change it. Fill `pass2_sheet.csv`, or extract the zip and open `rate.html` in a browser (load your filled
pass-1 sheet there to open the cards; it exports the filled pass-2 sheet).

| column | values | meaning |
|---|---|---|
| `reason_correct` | `yes` / `partly` / `no` | Is the reason a true description of what the crop shows, and does it bear on the question? `partly`: some of it is right, or it is right but misses the deciding feature. |
| `would_change_my_vote` | `yes` / `no` | Had you been validating this label, would reading this answer and reason have changed your vote? |
| `notes_pass2` | free text | Optional. |

Rate the reason on its own: judge whether it describes the crop accurately, whatever you think of the answer.
"""
RUBRIC_TEXT = RUBRIC_PASS1_TEXT + "\n" + RUBRIC_PASS2_TEXT
"""Both rubrics, for reference; each zip carries only its own."""


def run_agreement(args):
    if args.key_sha256:
        expected = open(args.key_sha256).read().split()[0]
        if sha256_of(args.key) != expected:
            raise SystemExit(f"{args.key} does not match {args.key_sha256}")
    key = pd.read_csv(args.key, dtype={"packet_id": str})
    pass1, pass2 = pd.read_csv(args.pass1, dtype=str), pd.read_csv(args.pass2, dtype=str)
    validate_codes(pass1)
    validate_codes(pass2)
    table = agreement_tables(unblind(pass1, pass2, key))
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

    tm = sub.add_parser("timing", help="time N crops per type at batch 1 on an idle machine")
    common(tm)
    tm.add_argument("--model", required=True, choices=sorted(MODELS) + [PRODUCTION_NAME])
    tm.add_argument("--n-per-type", type=int, default=10)
    tm.add_argument("--with-reason", action="store_true", help="also time the second-turn explanation")
    tm.add_argument("--device", default="mps", help="production only: cpu or mps")

    g = sub.add_parser("greedy", help="greedy text vs the score's argmax, stratified by type")
    common(g)
    g.add_argument("--model", required=True, choices=sorted(MODELS))
    g.add_argument("--n-per-type", type=int, default=2)
    g.add_argument("--seed", type=int, default=16)

    b = sub.add_parser("baseline", help="score the sample with the five production validators")
    common(b)
    b.add_argument("--device", default="cpu", help="cpu, mps or cuda")
    b.add_argument("--split-root", default=None,
                   help="issue #63: score every held-out crop of this round-3 run root instead of the sample")

    st = sub.add_parser("score-torch", help="issue #63: score the held-out split with one transformers VLM (klone)")
    st.add_argument("--model", required=True, choices=sorted(TORCH_MODELS))
    st.add_argument("--split-root", required=True, help="round-3 run root with <Type>/split.csv and the crops")
    st.add_argument("--out-dir", required=True, help="output root, outside the repo; writes scores/<leg>/<Type>.csv")
    st.add_argument("--variant", default="definition", choices=VARIANTS)
    st.add_argument("--label-types", default=None, help="comma-separated subset (default all five)")
    st.add_argument("--limit", type=int, default=0, help="smoke run: this many crops per type, half per class")
    st.add_argument("--batch", type=int, default=8)
    st.add_argument("--chunk", type=int, default=64, help="rows per CSV append")
    st.add_argument("--fewshot", type=int, default=0, help="in-context train examples per prompt (even; 0 = none)")
    st.add_argument("--max-batch-diff", type=float, default=0.02,
                    help="abort if batch-of-N and batch-of-1 p_incorrect differ by more (logged either way)")

    tt = sub.add_parser("timing-torch", help="issue #63: batch-1 latency of one transformers VLM")
    tt.add_argument("--model", required=True, choices=sorted(TORCH_MODELS))
    tt.add_argument("--split-root", required=True)
    tt.add_argument("--out-dir", required=True)
    tt.add_argument("--n-per-type", type=int, default=10)

    ex = sub.add_parser("export-heldout", help="issue #63: split slices + round-3 held-out scores, outside the repo")
    ex.add_argument("--split-root", required=True)
    ex.add_argument("--out-dir", required=True)

    ts = sub.add_parser("timing-summary", help="issue #63: timing.csv from the timing runs and the full runs")
    ts.add_argument("--out-dir", required=True, help="root with scores/ and timing/; writes scores/timing.csv")

    r = sub.add_parser("report", help="tables, figure and provenance into --doc-dir")
    common(r)
    r.add_argument("--doc-dir", default="docs/vlm-baseline-2026-10")
    r.add_argument("--cities-json", required=True, help="saved /v3/api/cities response (decides public naming)")
    r.add_argument("--min-city-n", type=int, default=30)
    r.add_argument("--min-city-type-n", type=int, default=20, help="per city x type rows need this many rows")
    r.add_argument("--within-min-n", type=int, default=10, help="within-city AUC: minimum rows per city")
    r.add_argument("--subset", type=int, default=100,
                   help="headline tables on balanced_subset(per_type=N), the crops every VLM scored (0 = all)")
    r.add_argument("--n-boot", type=int, default=1000)
    r.add_argument("--seed", type=int, default=20261021, help="bootstrap seed")

    e = sub.add_parser("explain", help="pick the explanation items and ask the best VLM why")
    common(e)
    e.add_argument("--model", required=True, choices=sorted(MODELS))
    e.add_argument("--cities-json", required=True)
    e.add_argument("--per-type", type=int, default=20, help="items per label type, half VLM-right (default 20)")
    e.add_argument("--reuse", default=None,
                   help="an earlier explanations.csv: its reasons are reused for items selected again (same model)")
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

    g = sub.add_parser("gallery", help="add the rate.html rating pages to an existing packet and re-zip it")
    g.add_argument("--out-dir", default=os.path.expanduser("~/Data/sidewalk-validator-ai-vlm-2026-10"),
                   help="where the two zips are written (default: the packet's private output directory)")
    g.add_argument("--packet-dir", required=True, help="packet directory (with pass1/ and pass2/)")
    g.add_argument("--key", required=True, help="the packet's key CSV, outside the repo")
    g.add_argument("--key-sha256", default=os.path.join(REPO_DIR, "docs/vlm-baseline-2026-10/packet_key.sha256"),
                   help="committed sha256 the key must match")
    g.add_argument("--doc-dir", default=os.path.join(REPO_DIR, "docs/vlm-baseline-2026-10"),
                   help="where the packet's provenance entry is refreshed ('' to skip)")

    a = sub.add_parser("agreement", help="unblind a filled sheet; VLM-vs-rater and rater-vs-crowd agreement")
    a.add_argument("pass1", help="filled pass1_sheet.csv")
    a.add_argument("pass2", help="filled pass2_sheet.csv")
    a.add_argument("--key", required=True)
    a.add_argument("--key-sha256", default=None, help="committed hash file to check --key against")
    a.add_argument("--out", default=None, help="optional CSV copy of the table")

    args = parser.parse_args(argv)
    if args.command == "score":
        run_score(args)
    elif args.command == "timing":
        run_timing(args)
    elif args.command == "greedy":
        run_greedy(args)
    elif args.command == "baseline":
        run_baseline(args)
    elif args.command == "score-torch":
        run_score_torch(args)
    elif args.command == "timing-torch":
        run_timing_torch(args)
    elif args.command == "export-heldout":
        run_export_heldout(args)
    elif args.command == "timing-summary":
        t = timing_summary(os.path.join(args.out_dir, "scores"), os.path.join(args.out_dir, "timing"))
        t.to_csv(os.path.join(args.out_dir, "scores", "timing.csv"), index=False)
        print(t.to_string(index=False))
        b = batch_invariance(os.path.join(args.out_dir, "scores"), os.path.join(args.out_dir, "timing"))
        b.to_csv(os.path.join(args.out_dir, "scores", "batch_invariance.csv"), index=False)
        print(b.to_string(index=False))
    elif args.command == "report":
        run_report(args)
    elif args.command == "explain":
        run_explain(args)
    elif args.command == "packet":
        run_packet(args, parser)
    elif args.command == "gallery":
        run_gallery(args)
    else:
        run_agreement(args)


if __name__ == "__main__":
    main()
