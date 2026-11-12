"""Score the round-3 held-out crops with a Gemini model on Vertex AI as a label validator (issue #63, an API leg).

The open-weight legs read ``p_incorrect`` off the next-token distribution (``vlm_baseline.yes_no_score``). Gemini on
Vertex gives no usable first-token log-probabilities for this, so this leg's score is **elicited**: the prompt is
``vlm_baseline.build_prompt`` unchanged (image first, one user turn, as every other leg), followed by one fixed
sentence, ``JSON_SENTENCE``, asking for ``{"answer": "yes" | "no", "confidence": 0-100}``, the confidence being the
model's confidence that its answer is right. ``RESPONSE_SCHEMA`` enforces that shape (``response_mime_type`` JSON,
temperature 0.0). The score is

    ``p_incorrect = confidence / 100`` when the answer is ``no``, else ``1 - confidence / 100``.

It is **self-reported**: every row carries ``score_source = "self_reported"`` and the meta file says so. The
log-probability columns of the shared score layout (``LOGPROB_COLUMNS``) are left empty. A reply that is empty, not
JSON, or outside the schema (answer not yes/no, confidence not a number in 0-100) is written as an **unscored** row:
``p_incorrect`` empty and ``unscored_reason`` saying why (with the finish reason when it was not ``STOP``), never a
0.5. ``vlm_scoreboard.load_leg`` drops those rows and the coverage table counts them.

**Arms.** ``--fewshot 0`` (default) or ``--fewshot 8``, with ``vlm_baseline``'s in-context machinery: the same
seeded order of the round-3 ``train`` rows (``fewshot_pools``, ``FEWSHOT_SEED``), the same rule (the first four per
class not on the query's panorama, interleaved), and the same example turns (the unchanged question, answered
``Yes``/``No``). Only the query turn carries ``JSON_SENTENCE``. With a cities list, an example from a deployment not
on the list is skipped like a panorama clash, so the eight examples equal the open-weight arm's whenever all of those
are on the list; the meta file records whether they did (``fewshot_matches_open_weight_arm``).

**Public deployments only.** ``--cities-file`` (required; a JSON list, a saved ``/v3/api/cities`` response, or one
city id per line) limits both the scored crops and the few-shot examples to those deployments; nothing else is sent.
The meta file records the list's sha256 (over the sorted ids, one per line) and its size, never the names of the
deployments left out. The leg is then "partial" on the scoreboard; a ``notes.json`` entry can say why.

**Cost.** After each label type one JSON line goes to ``--usage-log`` (outside any git checkout, or ``none``): model,
leg, type, calls, input / output / thinking / cached tokens from the SDK's usage metadata, wall time, and an estimate
from ``PRICES_USD_PER_MTOK`` (null for a model without a price there).

Subcommand ``score`` writes ``<out-dir>/<leg>/<Type>.csv`` (columns ``SCORE_COLUMNS``) and ``<Type>.meta.json``, where
``<leg>`` is ``vlm_baseline.leg_name(--name or --model, --variant, --fewshot)``; point ``--out-dir`` at
``<scores-dir>/scores`` and the scoreboard picks the leg up. Resumable: rows already in the CSV (scored or unscored)
are skipped; a request that still fails after ``--retries`` (429, 5xx, timeouts) writes nothing, so the next run
retries it. A run refuses to append to a leg whose meta was written under different settings (``binding``).
No ``timing.csv`` is written: API latency is not a GPU cost and would be misread by the scoreboard's GPU-hour columns.

The client is the ``google-genai`` SDK on Vertex AI with Application Default Credentials (``gcloud auth
application-default login``), project from ``GOOGLE_CLOUD_PROJECT``, location from ``GOOGLE_CLOUD_LOCATION``
(default ``global``, where the newest model ids are served). The SDK is imported only when a request is about to be
sent, so ``--help`` and the tests never need it. Per-crop outputs carry ``label_uid``s: keep them outside the repo.

Usage (a 50-crop pilot: ``--limit 10`` is five crops per class per type)::

    D=~/Data/sidewalk-validator-ai-retrain-2026-10/round3; O=~/Data/sidewalk-validator-ai-vlm-scoreboard-2026-10/scores
    export GOOGLE_GENAI_USE_VERTEXAI=true GOOGLE_CLOUD_PROJECT=<project> GOOGLE_CLOUD_LOCATION=global
    python analysis/vlm_gemini.py score --model gemini-3.8-flash --data-root $D --out-dir $O \\
        --cities-file ~/Data/public_cities.txt --usage-log ~/Data/vlm-gemini-usage.jsonl --limit 10
    python analysis/vlm_gemini.py score ... --fewshot 8      # the in-context arm (leg gemini-3.8-flash-fewshot8)

Example (the score from a reply)::

    >>> s = score_response({"text": '{"answer": "no", "confidence": 80}', "finish_reason": "STOP"})
    >>> round(s["p_incorrect"], 2), s["unscored_reason"]
    (0.8, '')
    >>> score_response({"text": "", "finish_reason": "SAFETY"})["unscored_reason"]
    'empty answer (finish_reason SAFETY)'
"""
import argparse
import concurrent.futures
import datetime
import functools
import hashlib
import json
import math
import os
import platform
import random
import sys
import time
import pandas as pd

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_DIR)
from analysis import vlm_baseline as vb  # noqa: E402
from analysis import vlm_vllm as vv  # noqa: E402
from analysis.ai_disagree_relabel import inside_git_checkout  # noqa: E402

BACKEND = "vertex-gemini-genai"
JSON_SENTENCE = ('Reply with JSON only: {"answer": "yes" or "no", "confidence": an integer from 0 to 100 for how '
                 'confident you are that your answer is right}.')
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string", "enum": ["yes", "no"]},
                   "confidence": {"type": "integer", "minimum": 0, "maximum": 100}},
    "required": ["answer", "confidence"],
    "propertyOrdering": ["answer", "confidence"],   # answer first: the confidence is about that answer
}
TEMPERATURE = 0.0
THINKING_LEVELS = ("minimal", "low", "medium", "high")
DEFAULT_LOCATION = "global"
RETRY_CODES = (408, 429, 500, 502, 503, 504)
SCORE_SOURCE = "self_reported"
SCORE_RULE = "p_incorrect = confidence/100 if answer == 'no' else 1 - confidence/100"
LOGPROB_COLUMNS = ("p_yes", "p_no", "answer_mass", "log_odds_incorrect", "top_token")
SCORE_COLUMNS = ["label_uid", "model", "variant", "p_yes", "p_no", "answer_mass", "log_odds_incorrect", "p_incorrect",
                 "p_correct", "top_token", "prompt_tokens", "latency_s", "batch_size", "gpu", "score_source", "answer",
                 "confidence", "unscored_reason", "finish_reason", "output_tokens", "thinking_tokens"]
"""The transformers path's columns (logprob ones empty, ``gpu`` empty), then this leg's own."""

PRICES_AS_OF = "2026-11-08"
PRICES_USD_PER_MTOK = {
    # USD per 1M tokens; thinking tokens are billed as output. Copied from RampNet's
    # scripts/model_comparison/pricing.py, which read them on 2026-11-08 from
    # https://ai.google.dev/gemini-api/docs/pricing
    # (Standard tab, the Gemini Developer API page, NOT the Vertex rate card that bills this path).
    # TODO: confirm against the Vertex price sheet (https://cloud.google.com/vertex-ai/generative-ai/pricing).
    # Introductory prices: Flash becomes $1.50/$7.50 from 2027-01-01, Argon $4.00/$20.00 from 2026-12-01.
    # Implicit-cache discounts are not priced (cached tokens are logged), so an 8-shot estimate is an upper bound.
    "gemini-3.8-flash": {"input": 0.75, "output": 3.75},
    "gemini-4-argon": {"input": 2.00, "output": 10.00},
}


# --------------------------------------------------------------------------------------------------------------------
# Pure logic -- unit-tested without the SDK
# --------------------------------------------------------------------------------------------------------------------

def query_text(label_type, variant="definition"):
    """The query turn's text: ``vlm_baseline.build_prompt`` unchanged, a space, ``JSON_SENTENCE``."""
    return vb.build_prompt(label_type, variant) + " " + JSON_SENTENCE


def parse_elicited(text):
    """``(answer, confidence)`` from the model's JSON reply; ``ValueError`` with a short reason when unusable.

    >>> parse_elicited('{"answer": "Yes", "confidence": 90}')
    ('yes', 90.0)
    >>> parse_elicited('{"answer": "yes", "confidence": 101}')
    Traceback (most recent call last):
    ...
    ValueError: confidence outside 0-100
    """
    if text is None or not str(text).strip():
        raise ValueError("empty answer")
    try:
        d = json.loads(str(text))
    except json.JSONDecodeError:
        raise ValueError("not JSON") from None
    if not isinstance(d, dict):
        raise ValueError("not a JSON object")
    a = d.get("answer")
    if not isinstance(a, str) or a.strip().lower() not in ("yes", "no"):
        raise ValueError("answer not yes/no")
    c = d.get("confidence")
    if isinstance(c, bool) or not isinstance(c, (int, float)) or not math.isfinite(c):
        raise ValueError("confidence not a number")
    if not 0 <= c <= 100:
        raise ValueError("confidence outside 0-100")
    return a.strip().lower(), float(c)


def elicited_p_incorrect(answer, confidence):
    """``p_incorrect`` from a yes/no answer and its 0-100 confidence (``SCORE_RULE``).

    >>> elicited_p_incorrect("no", 80), elicited_p_incorrect("yes", 80), elicited_p_incorrect("yes", 50)
    (0.8, 0.19999999999999996, 0.5)
    """
    c = confidence / 100.0
    return c if answer == "no" else 1.0 - c


def score_response(resp):
    """The score columns from one response dict (``{"text", "finish_reason", "usage", "model_version"}``, the shape
    ``VertexGemini`` returns). The logprob columns stay empty; an unusable reply gets ``unscored_reason``."""
    usage = resp.get("usage") or {}
    finish = resp.get("finish_reason") or ""
    out = {c: None for c in LOGPROB_COLUMNS}
    out.update(gpu="", score_source=SCORE_SOURCE, finish_reason=finish, prompt_tokens=usage.get("prompt"),
               output_tokens=usage.get("output"), thinking_tokens=usage.get("thinking"))
    try:
        answer, conf = parse_elicited(resp.get("text"))
    except ValueError as e:
        reason = str(e) + (f" (finish_reason {finish})" if finish and finish != "STOP" else "")
        out.update(answer="", confidence=None, p_incorrect=None, p_correct=None, unscored_reason=reason)
        return out
    p = elicited_p_incorrect(answer, conf)
    out.update(answer=answer, confidence=conf, p_incorrect=p, p_correct=1.0 - p, unscored_reason="")
    return out


def build_request(model, label_type, image, variant="definition", examples=(), thinking_level=None):
    """The request for one crop, as plain data: in-context ``examples`` (``[(image_path, "Yes"|"No"), ...]``) as
    user/model turn pairs with the unchanged question, then the query turn (image first, ``query_text``).

    >>> r = build_request("m", "Obstacle", "q.webp", examples=[("e.webp", "No")])
    >>> [c["role"] for c in r["contents"]], r["contents"][-1]["parts"][0], r["config"]["temperature"]
    (['user', 'model', 'user'], {'image': 'q.webp'}, 0.0)
    """
    q = vb.build_prompt(label_type, variant)
    contents = []
    for ex_image, answer in examples:
        contents.append({"role": "user", "parts": [{"image": ex_image}, {"text": q}]})
        contents.append({"role": "model", "parts": [{"text": answer}]})
    contents.append({"role": "user", "parts": [{"image": image}, {"text": query_text(label_type, variant)}]})
    config = {"temperature": TEMPERATURE, "response_mime_type": "application/json",
              "response_schema": RESPONSE_SCHEMA, "thinking_level": thinking_level}
    return {"model": model, "contents": contents, "config": config}


def status_code(exc):
    """The HTTP status of an SDK or HTTP error (``code`` on google-genai's ``APIError``), or None."""
    for attr in ("code", "status_code"):
        v = getattr(exc, attr, None)
        if isinstance(v, int) and not isinstance(v, bool):
            return v
    return None


def is_retryable(exc):
    """429 / 5xx / 408, a connection or timeout error, or a transport error of the HTTP stack; never a 4xx.

    >>> class E(Exception):
    ...     def __init__(self, code): self.code = code
    >>> is_retryable(E(429)), is_retryable(E(503)), is_retryable(E(400)), is_retryable(TimeoutError())
    (True, True, False, True)
    """
    code = status_code(exc)
    if code is not None:
        return code in RETRY_CODES
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return True
    return type(exc).__module__.split(".")[0] in ("httpx", "httpcore", "requests", "urllib3")


def backoff_s(attempt, base=2.0, cap=60.0, rng=random):
    """Seconds before retry ``attempt`` (0-based): exponential, capped, with jitter in [0.5, 1) of the step."""
    return min(cap, base * 2 ** attempt) * (0.5 + 0.5 * rng.random())


def load_cities_file(path):
    """``(sorted city ids, sha256)`` from a JSON list, a saved ``/v3/api/cities`` response (public ones), or one id
    per line (``#`` comments allowed). The sha256 is over the sorted ids joined by newlines, so it does not depend
    on the file's format."""
    with open(path, encoding="utf-8") as f:
        text = f.read()
    try:
        d = json.loads(text)
    except json.JSONDecodeError:
        d = None
    if isinstance(d, list):
        ids = {str(x).strip() for x in d if str(x).strip()}
    elif isinstance(d, dict) and "cities" in d:
        ids = {c["city_id"] for c in d["cities"] if c.get("visibility") == "public"}
    else:
        ids = {ln.strip() for ln in text.splitlines() if ln.strip() and not ln.strip().startswith("#")}
    if not ids:
        raise SystemExit(f"{path}: no city ids")
    ids = sorted(ids)
    return ids, hashlib.sha256("\n".join(ids).encode()).hexdigest()


def rewrite_prefix(paths, old, new):
    """Replace the leading directory ``old`` with ``new`` in each path (whole components only); others unchanged.
    For a ``split.csv`` whose ``crop_path`` holds absolute paths of another machine.

    >>> rewrite_prefix(["/g/run/x.webp", "/g/runs/y.webp", "/h/z.webp"], "/g/run", "D:/data")
    ['D:/data/x.webp', '/g/runs/y.webp', '/h/z.webp']
    """
    if not old:
        return list(paths)
    old, new = old.rstrip("/\\"), str(new).rstrip("/\\")
    out = []
    for p in map(str, paths):
        # On Windows, joining the data root with a drive-less absolute crop_path keeps the root's drive ("D:/g/...")
        q = os.path.splitdrive(p)[1] if not os.path.splitdrive(old)[0] else p
        hit = q == old or (q.startswith(old) and q[len(old)] in "/\\")
        out.append(new + q[len(old):] if hit else p)
    return out


def with_city(df):
    """``df`` with a ``city`` column, from ``label_uid`` (``city:label_id``) when ``split.csv`` has none."""
    return df if "city" in df else df.assign(city=df.label_uid.astype(str).str.split(":").str[0])


def listed_pools(train, cities):
    """``vlm_baseline.fewshot_pools`` over EVERY train row (the open-weight arm's seeded order), then rows of
    deployments not in ``cities`` dropped in place, so the examples are the open-weight arm's whenever those are all
    listed, and otherwise the next listed ones in the same order."""
    return {c: p[p.city.isin(cities)].reset_index(drop=True) for c, p in vb.fewshot_pools(train).items()}


def estimate_cost_usd(model, input_tokens, output_tokens, thinking_tokens, prices=PRICES_USD_PER_MTOK):
    """USD estimate (thinking billed as output), or None for a model without a price.

    >>> round(estimate_cost_usd("gemini-3.8-flash", 1_000_000, 100_000, 100_000), 4), estimate_cost_usd("x", 1, 1, 1)
    (1.5, None)
    """
    p = prices.get(model)
    if not p or p.get("input") is None or p.get("output") is None:
        return None
    return (input_tokens * p["input"] + (output_tokens + thinking_tokens) * p["output"]) / 1e6


def usage_record(model, leg, label_type, results, wall_s):
    """One ``--usage-log`` line from the per-crop results of one type (scored, unscored and failed)."""
    ok = [r for r in results if "error" not in r]
    tok = {k: int(sum(r.get(f"_{k}") or 0 for r in ok)) for k in ("prompt", "output", "thinking", "cached", "total")}
    return {"timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"), "model": model,
            "model_versions": sorted({r["_model_version"] for r in ok if r.get("_model_version")}), "leg": leg,
            "label_type": label_type, "n_calls": len(ok),
            "n_attempts": int(sum(r.get("_attempts", 1) for r in results)),
            "n_scored": sum(not r["unscored_reason"] for r in ok),
            "n_unscored": sum(bool(r["unscored_reason"]) for r in ok),
            "n_failed": len(results) - len(ok), "input_tokens": tok["prompt"], "output_tokens": tok["output"],
            "thinking_tokens": tok["thinking"], "cached_tokens": tok["cached"], "total_tokens": tok["total"],
            "wall_s": round(float(wall_s), 2),
            "est_cost_usd": estimate_cost_usd(model, tok["prompt"], tok["output"], tok["thinking"]),
            "prices_usd_per_mtok": PRICES_USD_PER_MTOK.get(model), "prices_as_of": PRICES_AS_OF}


# --------------------------------------------------------------------------------------------------------------------
# The Vertex client -- the only code that needs google-genai; ``post`` is injectable so the tests never load it
# --------------------------------------------------------------------------------------------------------------------

def image_mime(path):
    ext = os.path.splitext(path)[1].lower().lstrip(".")
    return {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "webp": "image/webp"}.get(ext, "image/webp")


def _read_bytes(path):
    with open(path, "rb") as f:
        return f.read()


_example_bytes = functools.lru_cache(maxsize=64)(_read_bytes)
"""Example crops repeat in every 8-shot request; query crops are read once each."""


def to_sdk_contents(contents, types, read=_read_bytes):
    """``build_request`` contents as google-genai ``Content`` objects (image bytes inline)."""
    def part(p):
        if "image" in p:
            data = read(p["image"])
            return types.Part.from_bytes(data=data, mime_type=image_mime(p["image"]))
        return types.Part.from_text(text=p["text"])
    return [types.Content(role=c["role"], parts=[part(p) for p in c["parts"]]) for c in contents]


def to_sdk_config(config, types):
    """``build_request`` config as a ``GenerateContentConfig`` (thinking level only when set)."""
    kw = dict(temperature=config["temperature"], response_mime_type=config["response_mime_type"],
              response_schema=config["response_schema"])
    if config.get("thinking_level"):
        kw["thinking_config"] = types.ThinkingConfig(thinking_level=config["thinking_level"].upper())
    return types.GenerateContentConfig(**kw)


def from_sdk_response(resp):
    """A google-genai response as ``{"text", "finish_reason", "usage", "model_version"}``. The text joins the first
    candidate's non-thought parts; a blocked prompt (no candidate) reads ``BLOCKED_<reason>`` or ``NO_CANDIDATES``."""
    cands = getattr(resp, "candidates", None) or []
    if cands:
        fr = getattr(cands[0], "finish_reason", None)
        finish = (getattr(fr, "name", None) or str(fr)) if fr is not None else ""
        parts = getattr(getattr(cands[0], "content", None), "parts", None) or []
        text = "".join(getattr(p, "text", None) or "" for p in parts if not getattr(p, "thought", False)) or None
    else:
        br = getattr(getattr(resp, "prompt_feedback", None), "block_reason", None)
        finish = f"BLOCKED_{getattr(br, 'name', None) or br}" if br is not None else "NO_CANDIDATES"
        text = None
    um = getattr(resp, "usage_metadata", None)
    usage = {"prompt": getattr(um, "prompt_token_count", None) or 0,
             "output": getattr(um, "candidates_token_count", None) or 0,
             "thinking": getattr(um, "thoughts_token_count", None) or 0,
             "cached": getattr(um, "cached_content_token_count", None) or 0,
             "total": getattr(um, "total_token_count", None) or 0}
    return {"text": text, "finish_reason": finish, "usage": usage,
            "model_version": getattr(resp, "model_version", None)}


class VertexGemini:
    """The real ``post``: a ``build_request`` dict in, ``from_sdk_response`` out, via google-genai on Vertex AI.
    The SDK's own retries are off (``attempts=1``) so ``Scorer`` is the only retry layer."""

    def __init__(self, timeout_s=120):
        try:
            from google import genai
            from google.genai import types
        except ImportError as e:
            raise SystemExit("the Gemini leg needs google-genai (pip install google-genai)") from e
        project = os.environ.get("GOOGLE_CLOUD_PROJECT")
        if not project:
            raise SystemExit("set GOOGLE_CLOUD_PROJECT (and GOOGLE_GENAI_USE_VERTEXAI=true; ADC via "
                             "`gcloud auth application-default login`)")
        self.location = os.environ.get("GOOGLE_CLOUD_LOCATION") or DEFAULT_LOCATION
        self.sdk_version = getattr(genai, "__version__", None)
        self.types = types
        self.client = genai.Client(vertexai=True, project=project, location=self.location,
                                   http_options=types.HttpOptions(timeout=int(timeout_s * 1000),
                                                                  retry_options=types.HttpRetryOptions(attempts=1)))

    def __call__(self, request):
        contents = request["contents"]
        last = len(contents) - 1   # the query image is read fresh; example images repeat and are cached
        sdk_contents = []
        for i, c in enumerate(contents):
            sdk_contents += to_sdk_contents([c], self.types, _read_bytes if i == last else _example_bytes)
        resp = self.client.models.generate_content(model=request["model"], contents=sdk_contents,
                                                   config=to_sdk_config(request["config"], self.types))
        return from_sdk_response(resp)


class Scorer:
    """Sends one request with retries on retryable errors (``is_retryable``), exponential backoff with jitter."""

    def __init__(self, post, retries=6, sleep=time.sleep):
        self.post, self.retries, self.sleep = post, retries, sleep

    def call(self, request):
        """``(response, attempts)``; re-raises a non-retryable error at once and a retryable one after ``retries``."""
        for attempt in range(self.retries + 1):
            try:
                return self.post(request), attempt + 1
            except Exception as e:   # noqa: BLE001 -- classified below
                if not is_retryable(e) or attempt == self.retries:
                    try:
                        e.attempts = attempt + 1
                    except AttributeError:
                        pass
                    raise
                wait = backoff_s(attempt)
                print(f"  retry {attempt + 1} in {wait:.1f}s after {type(e).__name__} {status_code(e) or ''}: "
                      f"{str(e)[:200]}", flush=True)
                self.sleep(wait)

    def score(self, request):
        """One crop's result: the score columns plus private ``_``-prefixed usage fields, or ``{"error", ...}``."""
        t0 = time.perf_counter()
        try:
            resp, attempts = self.call(request)
        except Exception as e:   # noqa: BLE001 -- reported by the caller, which decides whether to stop
            return {"error": f"{type(e).__name__}: {str(e)[:300]}", "retryable": is_retryable(e),
                    "_attempts": getattr(e, "attempts", 1)}
        out = score_response(resp)
        usage = resp.get("usage") or {}
        out.update(latency_s=time.perf_counter() - t0, _attempts=attempts, _model_version=resp.get("model_version"),
                   **{f"_{k}": usage.get(k) for k in ("prompt", "output", "thinking", "cached", "total")})
        return out


# --------------------------------------------------------------------------------------------------------------------
# Files
# --------------------------------------------------------------------------------------------------------------------

def check_owned(path, meta_path, binding):
    """Refuse a score CSV or meta another backend wrote, or a leg whose meta was written under other settings."""
    if os.path.exists(path) and os.path.getsize(path) > 0:
        header = list(pd.read_csv(path, nrows=0).columns)
        if header != SCORE_COLUMNS:
            raise SystemExit(f"{path} was not written by vlm_gemini (columns {header}); "
                             f"use another --out-dir or --name")
    if os.path.exists(meta_path):
        meta = vb._read_meta(meta_path)
        if meta.get("backend") != BACKEND:
            raise SystemExit(f"{meta_path} was not written by vlm_gemini; use another --out-dir or --name")
        old = meta.get("binding")
        # A meta with no scored row beside it binds nothing: a run that failed on its first request (a setting the
        # API refused, say) must not block the retry with corrected settings.
        if old is not None and old != binding and vv.done_uids(path):
            diff = sorted(k for k in set(old) | set(binding) if old.get(k) != binding.get(k))
            raise SystemExit(f"{meta_path}: this leg was scored with other settings ({', '.join(diff)}); "
                             f"use another --out-dir or --name")


def append_rows(path, rows):
    """Append rows in ``SCORE_COLUMNS`` order, writing the header only for a new (or empty) file."""
    new_file = not os.path.exists(path) or os.path.getsize(path) == 0
    pd.DataFrame(rows, columns=SCORE_COLUMNS).to_csv(path, mode="a", header=new_file, index=False)


def summarize_scores(path):
    """Aggregates of one score CSV for the meta file (no ``label_uid``)."""
    df = pd.read_csv(path, keep_default_na=False, na_values=[""])
    reasons = df.unscored_reason.fillna("")
    conf = pd.to_numeric(df.confidence, errors="coerce").dropna()
    lat = pd.to_numeric(df.latency_s, errors="coerce").dropna()
    return {"n_rows": int(len(df)), "n_scored": int((reasons == "").sum()), "n_unscored": int((reasons != "").sum()),
            "unscored_reasons": {str(k): int(v) for k, v in reasons[reasons != ""].value_counts().items()},
            "finish_reasons": {str(k): int(v)
                               for k, v in df.finish_reason.fillna("").astype(str).value_counts().items()},
            "answers": {str(k): int(v) for k, v in df.answer.fillna("").astype(str).value_counts().items()},
            "confidence_top_values": {str(int(k)) if float(k).is_integer() else str(k): int(v)
                                      for k, v in conf.value_counts().head(8).items()},
            "s_per_crop": float(lat.median()) if len(lat) else None}


def append_usage(path, record):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


# --------------------------------------------------------------------------------------------------------------------
# score
# --------------------------------------------------------------------------------------------------------------------

def _rewrite(args):
    return lambda paths: rewrite_prefix(list(paths), args.crop_root_from, args.crop_root_to)


def run_score(args, post=None):
    for what, p in (("--out-dir", args.out_dir), ("--usage-log", None if args.usage_log == "none" else args.usage_log)):
        if p is not None and inside_git_checkout(p):
            raise SystemExit(f"refusing {what} inside a git checkout (per-crop data and costs stay outside): {p}")
    if args.usage_log != "none" and os.path.isdir(args.usage_log):
        raise SystemExit(f"--usage-log is a directory: {args.usage_log}")
    if bool(args.crop_root_from) != bool(args.crop_root_to):
        raise SystemExit("--crop-root-from and --crop-root-to go together")
    if args.fewshot % 2:
        raise SystemExit("--fewshot must be even (half per class)")
    cities, digest = load_cities_file(args.cities_file)
    leg = vb.leg_name(args.name or args.model, args.variant, args.fewshot)
    leg_dir = os.path.join(args.out_dir, leg)
    os.makedirs(leg_dir, exist_ok=True)
    binding = {"backend": BACKEND, "model": args.model, "variant": args.variant, "fewshot": args.fewshot,
               "json_sentence": JSON_SENTENCE, "response_schema": RESPONSE_SCHEMA, "temperature": TEMPERATURE,
               "thinking_level": args.thinking_level, "cities_sha256": digest}
    rewrite = _rewrite(args)
    scorer = None
    for t in args.label_types.split(","):
        held = with_city(vv.load_heldout(args.data_root, t))
        listed = held[held.city.isin(cities)]
        df = vv.limit_rows(listed, args.limit).copy()
        df["image"] = rewrite(df.image)
        path = os.path.join(leg_dir, f"{t}.csv")
        meta_path = path.replace(".csv", ".meta.json")
        check_owned(path, meta_path, binding)
        done = vv.done_uids(path)
        todo = df[~df.label_uid.isin(done)]
        print(f"{leg} {t}: {len(todo)} to score, {len(df) - len(todo)} done ({len(listed)} of {len(held)} held-out "
              f"crops on the cities list), concurrency {args.concurrency}", flush=True)
        meta = vb._read_meta(meta_path)
        meta.update({"backend": BACKEND, "model": args.model, "leg": leg, "variant": args.variant, "label_type": t,
                     "prompt": vb.build_prompt(t, args.variant), "json_sentence": JSON_SENTENCE,
                     "query_text": query_text(t, args.variant), "response_schema": RESPONSE_SCHEMA,
                     "response_mime_type": "application/json", "temperature": TEMPERATURE,
                     "thinking_level": args.thinking_level, "score_source": SCORE_SOURCE, "score_rule": SCORE_RULE,
                     "logprob_columns_empty": list(LOGPROB_COLUMNS), "fewshot": args.fewshot,
                     "concurrency": args.concurrency, "batch_size": args.concurrency,
                     "cities_sha256": digest, "n_cities": len(cities), "n_crops_heldout": int(len(held)),
                     "n_crops_listed": int(len(listed)), "n_crops": int(len(df)), "limit": args.limit,
                     "crop_root_from": args.crop_root_from, "crop_root_to": args.crop_root_to, "binding": binding,
                     "prices_usd_per_mtok": PRICES_USD_PER_MTOK.get(args.model), "prices_as_of": PRICES_AS_OF})
        pools = base = None
        if args.fewshot:
            train = with_city(vb.load_split_rows(args.data_root, t, ("train",)))
            train["image"] = rewrite(train.image)
            pools = listed_pools(train, cities)
            base = vb.fewshot_examples(pools, None, args.fewshot)
            open_arm = vb.fewshot_examples(vb.fewshot_pools(train), None, args.fewshot)
            meta.update(fewshot_seed=vb.FEWSHOT_SEED, fewshot_examples=base.label_uid.tolist(),
                        fewshot_matches_open_weight_arm=base.label_uid.tolist() == open_arm.label_uid.tolist(),
                        fewshot_example_turns="the unchanged build_prompt question, answered Yes/No "
                                              "(vlm_baseline.answer_word); JSON_SENTENCE on the query turn only")

        def examples(r):
            if not args.fewshot:
                return []
            ex = base if r.pano_id not in set(base.pano_id) else vb.fewshot_examples(pools, r.pano_id, args.fewshot)
            return [(e.image, vb.answer_word(e.label_class)) for e in ex.itertuples()]

        needed = list(todo.image) + (list(base.image) if base is not None else [])
        missing = [p for p in needed if not os.path.exists(p)]
        if missing:
            raise SystemExit(f"{t}: {len(missing)} crop files missing, e.g. {missing[0]} "
                             f"(see --crop-root-from / --crop-root-to)")
        run = {"start": datetime.datetime.now().isoformat(timespec="seconds"), "n_todo": int(len(todo)),
               "python": platform.python_version(), "host": platform.node(),
               "git_commit": vb._git(REPO_DIR, "rev-parse", "HEAD")}
        meta.setdefault("runs", []).append(run)
        vb._write_meta(meta_path, meta)
        results, t0 = [], time.perf_counter()
        if len(todo):
            if scorer is None:
                post = post or VertexGemini()
                scorer = Scorer(post, retries=args.retries)
            meta.update(location=getattr(post, "location", None), sdk_version=getattr(post, "sdk_version", None))
            results = score_frame(scorer, todo, args, t, leg, path, examples)
        wall = time.perf_counter() - t0
        failed = [r for r in results if "error" in r]
        run.update(end=datetime.datetime.now().isoformat(timespec="seconds"), n_failed=len(failed),
                   n_written=len(results) - len(failed),
                   model_versions=sorted({r["_model_version"] for r in results if r.get("_model_version")}))
        if os.path.exists(path) and os.path.getsize(path) > 0:
            meta.update(summarize_scores(path))
        vb._write_meta(meta_path, meta)
        if results and args.usage_log != "none":
            append_usage(args.usage_log, usage_record(args.model, leg, t, results, wall))
        fatal = [r for r in failed if not r["retryable"]]
        if fatal:
            raise SystemExit(f"{t}: stopped on a non-retryable error ({fatal[0]['error']}); rows scored before it "
                             f"are written")
        if failed:
            print(f"  {t}: {len(failed)} requests still failing after {args.retries} retries (not written; the next "
                  f"run retries them), e.g. {failed[0]['error']}", flush=True)


def score_frame(scorer, todo, args, label_type, leg, path, examples):
    """Score ``todo`` with ``args.concurrency`` requests in flight, appending each chunk's written rows in input
    order. Returns every result (failed ones carry ``error``). Stops after the first chunk with a non-retryable
    error, once that chunk's good rows are on disk."""
    chunk = args.chunk or max(4 * args.concurrency, 16)
    rows = list(todo.itertuples())
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        for start in range(0, len(rows), chunk):
            part = rows[start:start + chunk]
            reqs = [build_request(args.model, label_type, r.image, args.variant, examples(r), args.thinking_level)
                    for r in part]
            got = list(pool.map(scorer.score, reqs))
            out = [{"label_uid": r.label_uid, "model": leg, "variant": args.variant, "batch_size": args.concurrency,
                    **{k: v for k, v in s.items() if not k.startswith("_")}}
                   for r, s in zip(part, got) if "error" not in s]
            if out:
                append_rows(path, out)
            results += got
            print(f"  {label_type} {start + len(part)}/{len(rows)}  written {len(out)}  "
                  f"unscored {sum(bool(o['unscored_reason']) for o in out)}  "
                  f"failed {sum('error' in s for s in got)}", flush=True)
            if any("error" in s and not s["retryable"] for s in got):
                break
    return results


def main(argv=None, post=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score", help="score the held-out crops of the listed deployments")
    s.add_argument("--model", required=True, help="Vertex model id, e.g. gemini-3.8-flash")
    s.add_argument("--name", default=None, help="leg name instead of the model id (directory <out-dir>/<leg>/)")
    s.add_argument("--data-root", required=True, help="round-3 run root with <Type>/split.csv and crops")
    s.add_argument("--out-dir", required=True, help="<scores-dir>/scores of the scoreboard; outside any git checkout")
    s.add_argument("--cities-file", required=True,
                   help="deployments whose crops may be sent: JSON list, saved /v3/api/cities response, "
                        "or one id per line")
    s.add_argument("--usage-log", required=True, help="JSONL cost ledger outside any git checkout, or 'none'")
    s.add_argument("--label-types", default=",".join(vb.LABEL_TYPES))
    s.add_argument("--variant", default="definition", choices=vb.VARIANTS)
    s.add_argument("--fewshot", type=int, default=0, help="in-context train examples per prompt (0 or 8)")
    s.add_argument("--thinking-level", default="minimal", choices=THINKING_LEVELS + ("default",),
                   help="Gemini thinking level; 'default' sends no thinking config (the model's own default)")
    s.add_argument("--concurrency", type=int, default=6)
    s.add_argument("--retries", type=int, default=6, help="retries per request on 429 / 5xx / timeouts")
    s.add_argument("--chunk", type=int, default=0, help="rows per CSV append (default max(4 x concurrency, 16))")
    s.add_argument("--limit", type=int, default=0, help="pilot: first limit//2 rows per class per type")
    s.add_argument("--crop-root-from", default=None, help="leading directory of crop paths in split.csv to replace")
    s.add_argument("--crop-root-to", default=None, help="its replacement on this machine")
    args = p.parse_args(argv)
    if args.thinking_level == "default":
        args.thinking_level = None
    run_score(args, post=post)


if __name__ == "__main__":
    main()
