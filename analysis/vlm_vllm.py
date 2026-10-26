"""Score the round-3 held-out crops with a VLM served by vLLM (issue #63, the GLM-5.3-Flash leg).

The transformers path (``vlm_baseline.py``) reads the full next-token distribution. A model too large for it (GLM-5.3-Flash,
320B MoE in FP8 on four H200s) is served by vLLM's OpenAI-compatible server instead, and this client reads the score
from the **first generated token's top-20 log-probabilities** (``max_tokens=1``, ``logprobs=true``,
``top_logprobs=20``, greedy). The prompt is ``vlm_baseline.build_prompt``, image first, one user turn, so rows are
comparable with the MLX and transformers legs. The score is ``vlm_baseline.yes_no_score`` over the yes/no spellings
(``YES_VARIANTS`` / ``NO_VARIANTS``) that appear among the 20 returned tokens:

- ``top20_mass_outside`` = 1 - (sum of the 20 returned probabilities): the mass vLLM did not show. It bounds how much
  yes/no mass could have been missed; it is near 0 when the answer tokens dominate (#16 found ~0.98 on yes/no).
- ``answer_floor``: empty when both answers appeared; ``yes``, ``no`` or ``both`` when a side had no spelling in the
  top 20. A missing side is given the 20th token's log-prob (an upper bound on any one unseen token), so the score is
  still defined and conservative; ``both`` leaves ``p_incorrect`` at 0.5 and the row should be read as unscored.

**Thinking.** GLM-5.3-Flash's chat template always opens a ``<think>`` block in the generation prompt and has no
switch to turn it off (the vLLM recipe: "Thinking is always on"). The template renders an assistant turn as
``<|assistant|><think>{reasoning}</think>{content}``, so ``--prefill '<think></think>'`` sends an assistant message with
that content and ``continue_final_message=true`` (``add_generation_prompt=false``): the prompt then ends with an empty,
closed think block and the next token is the answer. ``score`` records the last tokens of the rendered prompt (vLLM's
``/tokenize``) in the meta file so the prefill can be checked. Models with a real switch (Qwen3-VL) use
``--template-kwargs '{"enable_thinking": false}'`` and no prefill, as the transformers path does.

Subcommands:

``score``
    Score every held-out crop (``split`` in test / test_heldout of ``<data-root>/<Type>/split.csv``) of the given types
    against a running server, ``--concurrency`` requests in flight. Writes ``<out-dir>/<model>/<Type>.csv`` (columns
    ``SCORE_COLUMNS``: the transformers path's columns, then ``batch_size`` = concurrency, ``gpu``,
    ``top20_mass_outside``, ``answer_floor``) and ``<Type>.meta.json`` (repo, revision, vLLM version, image tag, TP size,
    request shape, rendered-prompt tail, per-chunk wall times). Resumable: rows already in the CSV are skipped, and the
    CSV is appended every chunk. Refreshes the run-concurrency row of ``<out-dir>/<model>/timing.csv`` at the end.
``timing``
    Latency at concurrency 1: one warm-up request, then 10 crops per type (``vlm_baseline.stratified_crops``, the Mac
    protocol), one at a time. Writes the per-crop latencies to ``<out-dir>/<model>/timing_c1.csv`` and the
    concurrency-1 row of ``timing.csv`` (columns ``TIMING_COLUMNS``, the scoreboard's layout).

The client needs only the standard library plus this repo's CPU requirements (numpy, pandas; ``vlm_baseline`` imports
``evaluate``, hence scikit-learn and matplotlib); it never imports vllm. ``analysis/vlm_vllm.slurm`` starts the server
in the ``vllm/vllm-openai`` Apptainer image, waits for ``/health``, runs this, and stops the server.

Usage (on a GPU node, with a server on port 8000)::

    D=/gscratch/makelab/jfroehli/sidewalk-validator-ai/retrain-2026-10; O=/gscratch/scrubbed/jfroehli/vlm-scoreboard/scores
    python analysis/vlm_vllm.py score --server http://127.0.0.1:8000 --model glm5.3-flash --repo zai-org/GLM-5.3-Flash \\
        --data-root $D --out-dir $O --concurrency 32 --gpu 4xH200 --tp 4 --image-tag vllm/vllm-openai:v0.32.0-x86_64-cu129 \\
        --prefill '<think></think>' --template-kwargs '{"reasoning_effort": "low"}'
    python analysis/vlm_vllm.py timing ... (same server and model arguments)

Example (the score from a top-20 list)::

    >>> import math
    >>> top = [("No", math.log(0.7)), ("Yes", math.log(0.2)), ("no", math.log(0.05)), ("The", math.log(0.04))]
    >>> s = score_top_logprobs(top)
    >>> round(s["p_incorrect"], 4), round(s["top20_mass_outside"], 4), s["answer_floor"]
    (0.7895, 0.01, '')
"""
import argparse
import base64
import concurrent.futures
import json
import math
import os
import sys
import time
import urllib.error
import urllib.request

import numpy as np
import pandas as pd

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_DIR)
from analysis import vlm_baseline as vb  # noqa: E402

HELDOUT_SPLITS = ("test", "test_heldout")
TOP_K = 20
SCORE_COLUMNS = ["label_uid", "model", "variant", "p_yes", "p_no", "answer_mass", "log_odds_incorrect", "p_incorrect",
                 "p_correct", "top_token", "prompt_tokens", "latency_s", "batch_size", "gpu", "top20_mass_outside",
                 "answer_floor"]
TIMING_COLUMNS = ["model", "gpu", "batch_size", "n_crops", "median_s_per_crop", "p90_s_per_crop", "source"]


# --------------------------------------------------------------------------------------------------------------------
# Pure logic -- unit-tested without a server
# --------------------------------------------------------------------------------------------------------------------

def score_top_logprobs(top, yes_variants=vb.YES_VARIANTS, no_variants=vb.NO_VARIANTS):
    """The yes/no score from a top-k list of ``(token_string, logprob)`` pairs (one generated position).

    Every entry is a distinct token id, so two entries that decode to the same spelling are both counted. A side with
    no spelling in the list gets the smallest returned log-prob as a stand-in (``answer_floor`` names the side).

    >>> import math
    >>> s = score_top_logprobs([("Yes", math.log(0.9)), (" yes", math.log(0.05)), ("Maybe", math.log(0.01))])
    >>> s["answer_floor"], round(s["p_incorrect"], 4), round(s["top20_mass_outside"], 4)
    ('no', 0.0104, 0.04)
    """
    if not top:
        raise ValueError("empty top-logprobs list")
    lp = [float(x) for _, x in top]
    toks = [t for t, _ in top]
    yes_ids = [i for i, t in enumerate(toks) if t in yes_variants]
    no_ids = [i for i, t in enumerate(toks) if t in no_variants]
    floor = min(lp)
    missing = [side for side, ids in (("yes", yes_ids), ("no", no_ids)) if not ids]
    if "yes" in missing:
        lp.append(floor)
        yes_ids = [len(lp) - 1]
    if "no" in missing:
        lp.append(floor)
        no_ids = [len(lp) - 1]
    out = vb.yes_no_score(np.array(lp), yes_ids, no_ids)
    seen = sum(math.exp(x) for x in lp[:len(top)])
    out["top20_mass_outside"] = max(0.0, 1.0 - seen)
    out["answer_floor"] = "both" if len(missing) == 2 else (missing[0] if missing else "")
    out["top_token"] = toks[int(np.argmax(lp[:len(top)]))]
    return out


def parse_chat_response(resp):
    """``(top, prompt_tokens)`` from an OpenAI-style chat-completion response dict with ``logprobs`` requested.

    ``top`` is the first generated position's ``top_logprobs`` as ``(token, logprob)`` pairs.

    >>> r = {"choices": [{"logprobs": {"content": [{"token": "No", "logprob": -0.1,
    ...       "top_logprobs": [{"token": "No", "logprob": -0.1}, {"token": "Yes", "logprob": -2.4}]}]}}],
    ...      "usage": {"prompt_tokens": 812, "completion_tokens": 1}}
    >>> parse_chat_response(r)
    ([('No', -0.1), ('Yes', -2.4)], 812)
    """
    try:
        first = resp["choices"][0]["logprobs"]["content"][0]
    except (KeyError, IndexError, TypeError) as e:
        raise ValueError(f"response has no first-token logprobs: {str(resp)[:300]}") from e
    top = [(d["token"], float(d["logprob"])) for d in first["top_logprobs"]]
    prompt_tokens = int(resp.get("usage", {}).get("prompt_tokens", -1))
    return top, prompt_tokens


def image_data_url(path):
    """A base64 ``data:`` URL for an image file (the server needs no access to the crop directory)."""
    ext = os.path.splitext(path)[1].lower().lstrip(".") or "png"
    mime = {"jpg": "jpeg"}.get(ext, ext)
    with open(path, "rb") as f:
        return f"data:image/{mime};base64," + base64.b64encode(f.read()).decode("ascii")


def build_messages(image_url, question, prefill=None):
    """The chat messages: image first, then the question, in one user turn; optional assistant prefill turn."""
    msgs = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": image_url}},
                                         {"type": "text", "text": question}]}]
    if prefill is not None:
        msgs.append({"role": "assistant", "content": prefill})
    return msgs


def build_request(served_name, image_url, question, prefill=None, template_kwargs=None, top_k=TOP_K):
    """The ``/v1/chat/completions`` body: greedy, one token, top-``top_k`` log-probs of that token.

    With ``prefill`` the assistant turn is continued (``continue_final_message``) instead of a fresh generation prompt.

    >>> b = build_request("m", "data:,", "Q?", prefill="<think></think>", template_kwargs={"reasoning_effort": "low"})
    >>> b["max_tokens"], b["top_logprobs"], b["continue_final_message"], b["add_generation_prompt"]
    (1, 20, True, False)
    >>> b["messages"][-1]
    {'role': 'assistant', 'content': '<think></think>'}
    >>> "continue_final_message" in build_request("m", "data:,", "Q?")
    False
    """
    body = {"model": served_name, "messages": build_messages(image_url, question, prefill), "max_tokens": 1,
            "temperature": 0.0, "logprobs": True, "top_logprobs": top_k, "n": 1}
    if prefill is not None:
        body.update(continue_final_message=True, add_generation_prompt=False)
    if template_kwargs:
        body["chat_template_kwargs"] = dict(template_kwargs)
    return body


def load_heldout(data_root, label_type):
    """Held-out rows of one type's ``split.csv`` with an absolute ``image`` column; ``label_uid`` must be unique."""
    df = pd.read_csv(os.path.join(data_root, label_type, "split.csv"), dtype={"label_id": str})
    df = df[df.split.isin(HELDOUT_SPLITS)].copy()
    if df.label_uid.duplicated().any():
        raise ValueError(f"{label_type}: duplicate label_uid in held-out rows")
    df["label_type"] = label_type
    df["image"] = [os.path.join(data_root, label_type, p) for p in df.crop_path]
    return df


def limit_rows(df, limit):
    """The first ``limit // 2`` rows per class by ``label_uid`` (the transformers path's ``--limit`` pilot)."""
    if not limit:
        return df
    return (df.sort_values(["label_type", "label_class", "label_uid"])
            .groupby(["label_type", "label_class"]).head(max(1, limit // 2)))


def done_uids(path):
    """``label_uid`` values already in a score CSV (empty set when it does not exist)."""
    return set(pd.read_csv(path, usecols=["label_uid"]).label_uid) if os.path.exists(path) else set()


def append_rows(path, rows):
    """Append rows to a score CSV in ``SCORE_COLUMNS`` order, writing the header only for a new file."""
    pd.DataFrame(rows, columns=SCORE_COLUMNS).to_csv(path, mode="a", header=not os.path.exists(path), index=False)


def timing_row(model, gpu, batch_size, s_per_crop, source):
    """One ``timing.csv`` row from per-crop seconds (latencies at concurrency 1, or chunk wall / n at concurrency N)."""
    s = np.asarray(s_per_crop, dtype=float)
    return {"model": model, "gpu": gpu, "batch_size": int(batch_size), "n_crops": int(len(s)),
            "median_s_per_crop": float(np.median(s)), "p90_s_per_crop": float(np.percentile(s, 90)), "source": source}


def upsert_timing(path, row):
    """Write ``row`` into ``timing.csv``, replacing any row with the same model, gpu and batch_size."""
    df = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame(columns=TIMING_COLUMNS)
    keep = ~((df.model == row["model"]) & (df.gpu == row["gpu"]) & (df.batch_size == row["batch_size"]))
    df = pd.concat([df[keep], pd.DataFrame([row])], ignore_index=True)[TIMING_COLUMNS]
    df.to_csv(path, index=False)


def chunk_s_per_crop(chunks):
    """Per-crop seconds at the run concurrency, one value per crop: each chunk's wall time / its size, repeated.

    >>> chunk_s_per_crop([[4, 2.0], [2, 3.0]])
    [0.5, 0.5, 0.5, 0.5, 1.5, 1.5]
    """
    return [w / n for n, w in chunks for _ in range(int(n)) if n]


def hf_revision(repo, hf_home):
    """The commit ``refs/main`` points at in a local HF cache, or None."""
    p = os.path.join(hf_home, "hub", "models--" + repo.replace("/", "--"), "refs", "main")
    return open(p).read().strip() if os.path.exists(p) else None


# --------------------------------------------------------------------------------------------------------------------
# HTTP -- ``post``/``get`` are injectable so the tests need no server
# --------------------------------------------------------------------------------------------------------------------

def http_post(url, body, timeout=600):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def http_get(url, timeout=30):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read() or b"null")


class Client:
    """Scores one crop per request against ``server``; retries a failed request twice before raising."""

    def __init__(self, server, served_name, prefill=None, template_kwargs=None, post=http_post, retries=2):
        self.server, self.served_name = server.rstrip("/"), served_name
        self.prefill, self.template_kwargs = prefill, template_kwargs
        self.post, self.retries = post, retries

    def score(self, image_path, question):
        body = build_request(self.served_name, image_data_url(image_path), question, self.prefill,
                             self.template_kwargs)
        for attempt in range(self.retries + 1):
            t0 = time.perf_counter()
            try:
                resp = self.post(self.server + "/v1/chat/completions", body)
                break
            except (urllib.error.URLError, OSError, ValueError) as e:
                if attempt == self.retries:
                    raise
                print(f"  retry {attempt + 1} after {type(e).__name__}: {e}", flush=True)
                time.sleep(5 * (attempt + 1))
        latency = time.perf_counter() - t0
        top, prompt_tokens = parse_chat_response(resp)
        out = score_top_logprobs(top)
        out.update(prompt_tokens=prompt_tokens, latency_s=latency)
        return out

    def rendered_tail(self, image_path, question, n=16):
        """The last ``n`` token strings of the rendered prompt (vLLM ``/tokenize``), to check the prefill landed."""
        body = {"model": self.served_name, "messages": build_messages(image_data_url(image_path), question,
                                                                         self.prefill),
                "return_token_strs": True}
        if self.prefill is not None:
            body.update(continue_final_message=True, add_generation_prompt=False)
        else:
            body["add_generation_prompt"] = True
        if self.template_kwargs:
            body["chat_template_kwargs"] = dict(self.template_kwargs)
        r = self.post(self.server + "/tokenize", body)
        return {"count": r.get("count"), "tail": (r.get("token_strs") or [])[-n:]}


def score_frame(client, todo, model, variant, concurrency, gpu, path, chunk=None, on_chunk=None):
    """Score ``todo`` rows with ``concurrency`` requests in flight, appending each chunk to ``path`` in input order.

    Returns ``[[n, wall_s], ...]`` per chunk (the run-concurrency throughput). ``on_chunk`` is called after each.
    """
    chunk = chunk or max(4 * concurrency, 16)
    rows = list(todo.itertuples())
    chunks = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        for start in range(0, len(rows), chunk):
            part = rows[start:start + chunk]
            t0 = time.perf_counter()
            results = list(pool.map(lambda r: client.score(r.image, vb.build_prompt(r.label_type, variant)), part))
            wall = time.perf_counter() - t0
            out = [{"label_uid": r.label_uid, "model": model, "variant": variant, **s, "batch_size": concurrency,
                    "gpu": gpu} for r, s in zip(part, results)]
            append_rows(path, out)
            chunks.append([len(part), wall])
            if on_chunk:
                on_chunk(start + len(part), len(rows), chunks)
    return chunks


# --------------------------------------------------------------------------------------------------------------------
# Subcommands
# --------------------------------------------------------------------------------------------------------------------

def _server_version(server, get=http_get):
    try:
        return (get(server.rstrip("/") + "/version") or {}).get("version")
    except Exception as e:   # noqa: BLE001 -- provenance only
        return f"unavailable ({type(e).__name__})"


def _base_meta(args, version):
    return {"model": args.model, "repo": args.repo, "revision": hf_revision(args.repo, args.hf_home),
            "variant": args.variant, "backend": "vllm-openai-server", "vllm_version": version,
            "image_tag": args.image_tag, "tensor_parallel_size": args.tp, "gpu": args.gpu,
            "batch_size": args.concurrency, "concurrency": args.concurrency,
            "request": {"max_tokens": 1, "temperature": 0.0, "top_logprobs": TOP_K, "prefill": args.prefill,
                        "chat_template_kwargs": args.template_kwargs},
            "thinking": args.thinking_note, "serve_args": os.environ.get("VLLM_SERVE_ARGS"),
            "yes_variants": list(vb.YES_VARIANTS), "no_variants": list(vb.NO_VARIANTS),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID")}


def _read_json(path):
    return json.load(open(path)) if os.path.exists(path) else {}


def _write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def summarize_scores(path):
    """Aggregates of one score CSV for the meta file: rows, floors, the outside-top-20 mass, answer mass."""
    df = pd.read_csv(path, keep_default_na=False)
    m = pd.to_numeric(df.top20_mass_outside)
    return {"n_rows": int(len(df)), "n_answer_floor": {k: int(v) for k, v in
                                                      df.answer_floor[df.answer_floor != ""].value_counts().items()},
            "top20_mass_outside_mean": float(m.mean()), "top20_mass_outside_p99": float(m.quantile(0.99)),
            "top20_mass_outside_max": float(m.max()), "answer_mass_mean": float(pd.to_numeric(df.answer_mass).mean()),
            "top_tokens": {str(k): int(v) for k, v in df.top_token.astype(str).value_counts().head(8).items()}}


def run_score(args):
    client = Client(args.server, args.served_name or args.repo, args.prefill, args.template_kwargs)
    version = _server_version(args.server)
    model_dir = os.path.join(args.out_dir, args.model)
    os.makedirs(model_dir, exist_ok=True)
    tail = None
    for t in args.label_types.split(","):
        df = limit_rows(load_heldout(args.data_root, t), args.limit)
        path = os.path.join(model_dir, f"{t}.csv")
        meta_path = path.replace(".csv", ".meta.json")
        done = done_uids(path)
        todo = df[~df.label_uid.isin(done)]
        print(f"{args.model} {t}: {len(todo)} to score, {len(done)} done, concurrency {args.concurrency}", flush=True)
        meta = {**_read_json(meta_path), **_base_meta(args, version), "label_type": t}
        if tail is None and len(df):
            r = df.iloc[0]
            try:
                tail = client.rendered_tail(r.image, vb.build_prompt(t, args.variant))
            except Exception as e:   # noqa: BLE001 -- provenance only
                tail = {"error": f"{type(e).__name__}: {e}"}
            print(f"  rendered prompt tail: {tail}", flush=True)
        meta["rendered_prompt_tail"] = tail
        meta.setdefault("chunks", [])
        _write_json(meta_path, meta)
        if len(todo):
            def progress(k, n, chunks, _t0=time.time()):
                print(f"  {t} {k}/{n}  {time.time() - _t0:.0f}s  last chunk {chunks[-1][1] / chunks[-1][0]:.3f} "
                      f"s/crop", flush=True)
                meta["chunks"] = meta["chunks"][:n_prev] + chunks
                _write_json(meta_path, meta)
            n_prev = len(meta["chunks"])
            score_frame(client, todo, args.model, args.variant, args.concurrency, args.gpu, path, on_chunk=progress)
        if os.path.exists(path):
            meta.update(summarize_scores(path))
            sp = chunk_s_per_crop(meta["chunks"])
            if sp:
                meta["s_per_crop"] = float(np.median(sp))
            _write_json(meta_path, meta)
    # run-concurrency timing row from every type's chunks scored at this concurrency and GPU
    sp = []
    for mp in sorted(os.listdir(model_dir)):
        if mp.endswith(".meta.json"):
            m = _read_json(os.path.join(model_dir, mp))
            if m.get("gpu") == args.gpu and m.get("concurrency") == args.concurrency:
                sp += chunk_s_per_crop(m.get("chunks", []))
    if sp:
        upsert_timing(os.path.join(model_dir, "timing.csv"),
                      timing_row(args.model, args.gpu, args.concurrency, sp,
                                 f"vlm_vllm score: chunk wall / chunk size at concurrency {args.concurrency}"))


def run_timing(args):
    client = Client(args.server, args.served_name or args.repo, args.prefill, args.template_kwargs)
    sample = pd.concat([load_heldout(args.data_root, t) for t in args.label_types.split(",")])
    crops = vb.stratified_crops(sample, args.per_type)
    model_dir = os.path.join(args.out_dir, args.model)
    os.makedirs(model_dir, exist_ok=True)
    first = crops.iloc[0]
    client.score(first.image, vb.build_prompt(first.label_type, args.variant))   # warm-up, not timed
    rows = []
    for r in crops.itertuples():
        s = client.score(r.image, vb.build_prompt(r.label_type, args.variant))
        rows.append({"label_uid": r.label_uid, "label_type": r.label_type, "latency_s": s["latency_s"],
                     "p_incorrect": s["p_incorrect"]})
    pd.DataFrame(rows).to_csv(os.path.join(model_dir, "timing_c1.csv"), index=False)
    row = timing_row(args.model, args.gpu, 1, [r["latency_s"] for r in rows],
                     f"vlm_vllm timing: {args.per_type} crops per type, concurrency 1, after one warm-up")
    upsert_timing(os.path.join(model_dir, "timing.csv"), row)
    print(row)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("score", "timing"):
        s = sub.add_parser(name)
        s.add_argument("--server", default="http://127.0.0.1:8000")
        s.add_argument("--model", required=True, help="short name, the scores/<model>/ directory")
        s.add_argument("--repo", required=True, help="HF repo id (provenance; also the served name by default)")
        s.add_argument("--served-name", default=None)
        s.add_argument("--data-root", required=True, help="retrain-2026-10 root with <Type>/split.csv and crops")
        s.add_argument("--out-dir", required=True)
        s.add_argument("--label-types", default=",".join(vb.LABEL_TYPES))
        s.add_argument("--variant", default="definition", choices=vb.VARIANTS)
        s.add_argument("--concurrency", type=int, default=1 if name == "timing" else 16)
        s.add_argument("--gpu", required=True, help='label, e.g. "4xH200"')
        s.add_argument("--tp", type=int, default=1)
        s.add_argument("--image-tag", default=None)
        s.add_argument("--prefill", default=None, help="assistant prefill continued by the server, e.g. '<think></think>'")
        s.add_argument("--template-kwargs", type=json.loads, default=None, help="JSON chat_template_kwargs")
        s.add_argument("--thinking-note", default=None, help="how thinking was disabled, for the meta file")
        s.add_argument("--hf-home", default=os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")))
        if name == "score":
            s.add_argument("--limit", type=int, default=0, help="pilot: first limit//2 rows per class per type")
        else:
            s.add_argument("--per-type", type=int, default=10)
    args = p.parse_args(argv)
    if args.cmd == "timing":
        args.concurrency = 1
        run_timing(args)
    else:
        run_score(args)


if __name__ == "__main__":
    main()
