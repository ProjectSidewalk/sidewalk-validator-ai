"""Export a validator checkpoint to ONNX, quantize it to int8, and benchmark the variants on the same crops.

Issue #20 asks whether a cheaper validator could replace the five 300M-parameter ViT-L models the API serves (about
1.2 GB each). Before any ViT-S fine-tune or distillation (GPU work, written up in docs/distillation-2026-10.md), this
script answers the CPU-side questions that need no training: how much do ONNX and int8 alone save, do they change any
served vote, and how do the existing ``projectsidewalk/quantized_*_dinov2_tiny_onnx`` models compare on the same
held-out crops.

Three subcommands:

``export``
    A checkpoint (local directory, or Hugging Face id plus ``--revision``) to ``<out>/model.onnx`` with the legacy
    TorchScript exporter (``torch.onnx.export(..., dynamo=False)``), opset 17, dynamic batch axis, fixed 224 x 224
    input (the preprocessor's crop size, so the position-embedding interpolation is traced for that size), then
    ``onnx.checker``. ``config.json`` and ``preprocessor_config.json`` are copied next to it so a consumer reproduces
    the exact preprocessing, and ``export_info.json`` records the source, revision, exporter, versions and sha256.
``quantize``
    int8 without calibration data, recorded in the same ``export_info.json``. ``--method dynamic`` (default):
    ``onnxruntime.quantization.quantize_dynamic`` (int8 weights, uint8 activations quantized at run time) to
    ``model_int8.onnx``. ``--method weight-only``: 8-bit block-quantized weights with fp32 activations
    (``MatMulNBits``) to ``model_int8_wo.onnx``.
``bench``
    Scores every crop with the PyTorch reference (fp32, on CPU and optionally MPS) and each ``--onnx`` variant, then
    writes one row per variant: size on disk, parameter count, median latency at each ``--batch-sizes``, accuracy,
    macro-F1, AUC and recall at 95% precision (``evaluate.slice_metrics``, so the same definitions as evaluate.py),
    and parity against the PyTorch CPU reference: max |logit difference|, max |softmax difference|, verdict flips,
    and how many crops change their served vote. A served vote is the API's rule: verdict, confidence looked up in
    the type's accuracy map (``--accuracy-map``, sidewalk-ai-api's ``accuracy_mappings/<type>.json``), vote only if
    the looked-up value is >= 0.92. Without a map, the raw confidence >= 0.92 is used as a proxy and the CSV says so.

    Crops are scored one per call by default (``--score-batch 1``), as the API does. This matters for dynamically
    quantized graphs (``DynamicQuantizeLinear``, e.g. ``quantize --method dynamic`` and the tiny repos): they pick one
    activation scale over the whole batch, so a crop's output changes with its batch neighbours. They are always
    scored at batch 1, and the row's ``b16_vs_b1_max_abs_dprob`` reports how far batch-16 scoring would move them.
    ``--cities-json`` (a saved ``/v3/api/cities`` response) keeps only crops from public deployments
    (``public_city_rule``: the schema-name match of ``calibration.per_city`` or the subdomain of
    ``data_generation.discover_cities``); the published test splits also contain crops from deployments that
    are private today, and only their count is recorded.

Latency is the median of ``--runs`` timed calls after ``--warmup`` untimed ones, on preprocessed tensors (image
decoding and preprocessing are excluded, and are the same for every variant), with ``--threads`` intra-op threads for
both PyTorch and onnxruntime. Numbers are comparable only within one machine and one thread setting; the CSV records
the platform, the thread count and the load average at the start of each variant.

Optional dependencies (not in requirements.txt, which the training path pins; install into your own venv)::

    pip install onnx onnxruntime huggingface_hub    # tested with onnx 1.23.2, onnxruntime 1.31.0, torch 2.14.1
    pip install onnx-ir                             # only for quantize --method weight-only (tested 1.0.0)

Usage::

    # ViT-L production Obstacle validator -> ONNX fp32 and int8
    python export_onnx.py export --model projectsidewalk/sidewalk-validator-ai-obstacle \\
        --revision 8f0fe8fc48748fac35098e74c1848d015126deef --out onnx/obstacle
    python export_onnx.py quantize onnx/obstacle/model.onnx                        # -> model_int8.onnx
    python export_onnx.py quantize onnx/obstacle/model.onnx --method weight-only   # -> model_int8_wo.onnx

    # held-out crops laid out like the HF dataset: <dir>/test/<class>/<city>_<label_id>.webp
    hf download projectsidewalk/sidewalk-validator-ai-dataset-obstacle --repo-type dataset \\
        --include "test/*" --local-dir data/obstacle

    python export_onnx.py bench --label-type Obstacle --crops data/obstacle \\
        --model projectsidewalk/sidewalk-validator-ai-obstacle --revision 8f0fe8fc48748fac35098e74c1848d015126deef \\
        --onnx onnx-fp32=onnx/obstacle/model.onnx --onnx onnx-int8=onnx/obstacle/model_int8.onnx \\
        --onnx tiny-int8=tiny/obstacle/onnx/model_quantized.onnx \\
        --accuracy-map ../sidewalk-ai-api/accuracy_mappings/obstacle.json --cities-json cities.json \
        --mps --threads 4 --out bench/

    # all bench CSVs -> one markdown table
    python export_onnx.py table bench/bench_*.csv > table.md
"""
import argparse
import glob
import hashlib
import json
import os
import platform
import shutil
import statistics
import time
import warnings

import numpy as np
import pandas as pd

from calibration._isotonic_map import GATE, api_lookup_many, predict_verdicts
from calibration.per_city import public_name
from evaluate import CLASSES, slice_metrics

OPSET = 17
INPUT_NAME = "pixel_values"
OUTPUT_NAME = "logits"
IMAGE_EXTENSIONS = (".webp", ".png", ".jpg", ".jpeg")
BATCH_CHECK = 16
"""Batch size at which a dynamically quantized graph is re-scored, to report how much its output depends on the batch."""


# ---------------------------------------------------------------------------------------------------------------------
# Small pure helpers (unit-tested)
# ---------------------------------------------------------------------------------------------------------------------

def sha256_file(path, chunk=1 << 20):
    """Hex sha256 of a file, read in 1 MB chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def softmax(logits):
    """Row-wise softmax in float32, as the API's ``F.softmax`` on float32 logits (stable).

    The precision matters: the maps step every 1e-6 near confidence 1.0, about the size of float32 rounding there.
    """
    z = np.asarray(logits, dtype=np.float32)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def served_votes(p_correct, p_incorrect, accuracy_map=None, gate=GATE):
    """The verdict and whether the API would turn it into a vote.

    With ``accuracy_map`` (the API's ``{"correct": {threshold: value}, "incorrect": {...}}``), the confidence of the
    predicted class is looked up exactly as sidewalk-ai-api does and compared with ``gate``. Without it, the raw
    confidence is compared with ``gate`` (a proxy: the served gate is on the mapped value, not on the confidence).

    Returns:
        ``(verdict, votes)``: string array and boolean array.

    Example::

        >>> v, ok = served_votes([0.95, 0.40], [0.05, 0.60])
        >>> v.tolist(), ok.tolist()
        (['correct', 'incorrect'], [True, False])
    """
    verdict, confidence = predict_verdicts(p_correct, p_incorrect)
    if accuracy_map is None:
        served = confidence
    else:
        served = np.zeros(len(verdict))
        for cls in CLASSES:
            m = verdict == cls
            if m.any():
                served[m] = api_lookup_many(accuracy_map[cls], confidence[m])
    return verdict, served >= gate


def parity(ref_logits, logits, accuracy_map=None, gate=GATE):
    """How far one variant's logits are from the reference's, on the same crops in the same order.

    Columns are ``[correct, incorrect]``. ``vote_changes`` counts crops whose served vote differs: the verdict
    changed while either side votes, or the vote turned on or off (see ``served_votes``).

    Example::

        >>> r = parity(np.array([[2.0, 0.0], [0.0, 1.0]]), np.array([[2.0, 0.1], [1.0, 0.0]]))
        >>> r["verdict_flips"], round(r["max_abs_dlogit"], 3)
        (1, 1.0)
    """
    ref_logits = np.asarray(ref_logits, dtype=np.float64)
    logits = np.asarray(logits, dtype=np.float64)
    if ref_logits.shape != logits.shape:
        raise ValueError(f"shape mismatch {ref_logits.shape} vs {logits.shape}")
    pr, pv = softmax(ref_logits), softmax(logits)
    vr, okr = served_votes(pr[:, 0], pr[:, 1], accuracy_map, gate)
    vv, okv = served_votes(pv[:, 0], pv[:, 1], accuracy_map, gate)
    changed = (okr != okv) | ((vr != vv) & (okr | okv))
    return {"n": int(len(logits)),
            "max_abs_dlogit": float(np.abs(ref_logits - logits).max()) if len(logits) else 0.0,
            "max_abs_dprob": float(np.abs(pr - pv).max()) if len(logits) else 0.0,
            "verdict_flips": int((vr != vv).sum()),
            "votes_ref": int(okr.sum()), "votes": int(okv.sum()), "vote_changes": int(changed.sum())}


def accuracy_row(y_true, logits):
    """acc, macro-F1, AUC and recall at 95% precision per class, via evaluate.slice_metrics (one source of truth)."""
    p = softmax(logits)
    m = slice_metrics(pd.DataFrame({"true": y_true, "p_correct": p[:, 0], "p_incorrect": p[:, 1]}))
    rap = m.get("recall_at_precision", {})
    return {"accuracy": m["accuracy"], "balanced_accuracy": m["balanced_accuracy"],
            "macro_f1": float(np.mean([m["per_class"][c]["f1-score"] for c in CLASSES])), "auc": m["auc"],
            "r_at_p95_correct": rap.get("correct", {}).get("0.95", {}).get("recall"),
            "r_at_p95_incorrect": rap.get("incorrect", {}).get("0.95", {}).get("recall")}


def list_crops(crops_dir, split="test"):
    """Crops laid out like the HF dataset repos: ``<crops_dir>/<split>/<class>/<city>_<label_id>.<ext>``.

    ``<crops_dir>/<class>/...`` (no split level) is also accepted. ``label_uid`` is ``<city>:<label_id>``, with the
    city taken up to the LAST underscore, since city slugs may contain ``_`` but label ids never do.

    Returns:
        DataFrame ``label_uid, label_class, crop_path`` sorted by ``label_uid`` (a stable order for parity).
    """
    base = os.path.join(crops_dir, split) if os.path.isdir(os.path.join(crops_dir, split)) else crops_dir
    rows = []
    for cls in CLASSES:
        for f in sorted(glob.glob(os.path.join(base, cls, "*"))):
            if not f.lower().endswith(IMAGE_EXTENSIONS):
                continue
            city, _, label_id = os.path.splitext(os.path.basename(f))[0].rpartition("_")
            rows.append({"label_uid": f"{city}:{label_id}", "label_class": cls, "crop_path": f})
    if not rows:
        raise FileNotFoundError(f"no crops under {base}/{{correct,incorrect}}/")
    df = pd.DataFrame(rows).sort_values("label_uid", kind="stable").reset_index(drop=True)
    if not df.label_uid.is_unique:
        raise ValueError(f"{crops_dir}: label_uid is not unique")
    return df


def public_city_rule(listing):
    """A predicate ``is_public(city)`` for the city part of a crop name, from a ``/v3/api/cities`` listing.

    Crop names use either the schema's city (``new_york``), matched to a ``city_id`` with
    ``calibration.per_city.public_name``, or the deployment's subdomain (``sea`` for
    ``https://sidewalk-sea.cs.washington.edu``), from ``data_generation.discover_cities``. A name is public if either
    rule says so and it is not the subdomain of a deployment that is not public. Everything else (private, unlisted,
    ambiguous) is not public.

    Example::

        >>> rule = public_city_rule({"cities": [
        ...     {"city_id": "seattle-wa", "visibility": "public", "url": "https://sidewalk-sea.cs.washington.edu"},
        ...     {"city_id": "hidden-xx", "visibility": "private", "url": "https://sidewalk-hid.cs.washington.edu"}]})
        >>> [rule(c) for c in ("seattle", "sea", "hidden", "hid", "nowhere")]
        [True, True, False, False, False]
    """
    from data_generation import discover_cities
    entries = listing.get("cities", listing) if isinstance(listing, dict) else listing
    cities = {c["city_id"]: {"visibility": c.get("visibility"), "country_id": c.get("country_id")} for c in entries}
    public_subs = set(discover_cities(listing, public_only=True).values())
    other_subs = set(discover_cities(listing, public_only=False).values()) - public_subs
    return lambda city: city not in other_subs and (city in public_subs or public_name(city, cities) is not None)


def public_crops(crops, listing):
    """Keep only crops from public deployments (``public_city_rule``); the city is ``label_uid`` before ``:``.

    Returns:
        ``(kept, n_dropped)``. Only the count of dropped crops is returned, never which deployments they came from.

    Example::

        >>> df = pd.DataFrame({"label_uid": ["seattle:1", "hidden:2"], "label_class": ["correct", "incorrect"]})
        >>> kept, n = public_crops(df, {"cities": [
        ...     {"city_id": "seattle-wa", "visibility": "public", "url": "https://sidewalk-sea.cs.washington.edu"},
        ...     {"city_id": "hidden-xx", "visibility": "private", "url": None}]})
        >>> kept.label_uid.tolist(), n
        (['seattle:1'], 1)
    """
    rule = public_city_rule(listing)
    keep = crops.label_uid.str.rpartition(":")[0].map(rule).astype(bool)
    return crops[keep].reset_index(drop=True), int((~keep).sum())


def markdown_table(df, columns=None, floatfmt=".4g"):
    """A GitHub-markdown table of ``df`` (no tabulate dependency). ``None``/NaN print as an em dash."""
    columns = list(columns or df.columns)

    def cell(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "—"
        if isinstance(v, (float, np.floating)):
            return format(float(v), floatfmt)
        return str(v)
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(cell(r[c]) for c in columns) + " |")
    return "\n".join(lines) + "\n"


def median_ms(fn, warmup, runs):
    """Median wall-clock milliseconds of ``fn()`` over ``runs`` calls after ``warmup`` untimed ones."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    return statistics.median(times)


# ---------------------------------------------------------------------------------------------------------------------
# Export and quantization
# ---------------------------------------------------------------------------------------------------------------------

def _versions():
    import onnx
    import onnxruntime
    import torch
    import transformers
    return {"torch": torch.__version__, "transformers": transformers.__version__, "onnx": onnx.__version__,
            "onnxruntime": onnxruntime.__version__, "python": platform.python_version()}


def _update_info(out_dir, update):
    path = os.path.join(out_dir, "export_info.json")
    info = {}
    if os.path.exists(path):
        with open(path) as f:
            info = json.load(f)
    for k, v in update.items():
        if isinstance(v, dict) and isinstance(info.get(k), dict):
            info[k].update(v)
        else:
            info[k] = v
    with open(path, "w") as f:
        json.dump(info, f, indent=2)
    return info


def export_model(model, out_dir, revision=None, opset=OPSET, image_size=None):
    """Export ``model`` (local dir or HF id) to ``<out_dir>/model.onnx``; return the path.

    The dummy input is ``1 x 3 x H x W`` with H, W from the preprocessor's ``crop_size`` (or ``image_size``); only
    the batch axis is dynamic.
    """
    import onnx
    import torch
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    os.makedirs(out_dir, exist_ok=True)
    processor = AutoImageProcessor.from_pretrained(model, revision=revision, use_fast=False)
    net = AutoModelForImageClassification.from_pretrained(model, revision=revision, attn_implementation="eager").eval()
    if image_size is None:
        crop = getattr(processor, "crop_size", None) or {}
        image_size = (crop["height"], crop["width"]) if crop else (net.config.image_size,) * 2
    h, w = (image_size, image_size) if isinstance(image_size, int) else image_size
    dummy = torch.randn(1, net.config.num_channels, h, w)

    class LogitsOnly(torch.nn.Module):
        """Return the logits tensor alone, so the graph has one named output."""
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, pixel_values):
            return self.m(pixel_values=pixel_values).logits

    path = os.path.join(out_dir, "model.onnx")
    with torch.no_grad():
        torch.onnx.export(LogitsOnly(net), (dummy,), path, input_names=[INPUT_NAME], output_names=[OUTPUT_NAME],
                          dynamic_axes={INPUT_NAME: {0: "batch"}, OUTPUT_NAME: {0: "batch"}},
                          opset_version=opset, do_constant_folding=True, dynamo=False)
    onnx.checker.check_model(path)   # path form: works for models near the 2 GB protobuf limit
    processor.save_pretrained(out_dir)
    net.config.save_pretrained(out_dir)
    n_params = int(sum(p.numel() for p in net.parameters()))
    _update_info(out_dir, {
        "source": model, "revision": revision, "exporter": "torch.onnx.export (TorchScript, dynamo=False)",
        "attn_implementation": "eager", "opset": opset, "input": {INPUT_NAME: ["batch", net.config.num_channels, h, w]},
        "output": {OUTPUT_NAME: ["batch", net.config.num_labels]}, "id2label": net.config.id2label,
        "parameters": n_params, "preprocessing": "AutoImageProcessor(use_fast=False), i.e. the slow PIL "
                                                 "BitImageProcessor sidewalk-ai-api uses (backend='pil')",
        "versions": _versions(),
        "files": {"model.onnx": {"sha256": sha256_file(path), "bytes": os.path.getsize(path)}}})
    return path


QUANT_METHODS = ("dynamic", "weight-only")


def quantize_model(onnx_path, out_path=None, per_channel=False, method="dynamic", block_size=128):
    """Quantize ``onnx_path`` to int8; return the output path (default ``model_int8.onnx`` beside it).

    ``method="dynamic"``: ``quantize_dynamic``, int8 weights and uint8 activations quantized per call, integer
    matmuls (``MatMulInteger``). ``method="weight-only"``: ``MatMulNBitsQuantizer(bits=8)``, int8 weights in blocks of
    ``block_size`` with one scale per block, dequantized inside the ``MatMulNBits`` kernel, so activations stay fp32.
    Weight-only needs the ``onnx-ir`` package as well (onnxruntime imports it for this quantizer).
    """
    if method not in QUANT_METHODS:
        raise ValueError(f"method must be one of {QUANT_METHODS}")
    default = "model_int8.onnx" if method == "dynamic" else "model_int8_wo.onnx"
    out_path = out_path or os.path.join(os.path.dirname(onnx_path), default)
    if method == "dynamic":
        from onnxruntime.quantization import QuantType, quantize_dynamic
        quantize_dynamic(onnx_path, out_path, weight_type=QuantType.QInt8, per_channel=per_channel)
        details = {"method": "onnxruntime.quantization.quantize_dynamic", "weight_type": "QInt8",
                   "activations": "QUInt8 (dynamic)", "per_channel": per_channel}
    else:
        import onnx
        from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer
        q = MatMulNBitsQuantizer(onnx.load(onnx_path), bits=8, block_size=block_size, is_symmetric=True)
        q.process()
        onnx.save_model(q.model.model, out_path)
        details = {"method": "onnxruntime.quantization.matmul_nbits_quantizer.MatMulNBitsQuantizer",
                   "bits": 8, "block_size": block_size, "symmetric": True, "activations": "fp32"}
    out_dir = os.path.dirname(out_path)
    for name in ("config.json", "preprocessor_config.json"):   # keep the int8 file self-describing if moved
        src = os.path.join(os.path.dirname(onnx_path), name)
        if os.path.exists(src) and os.path.abspath(out_dir) != os.path.abspath(os.path.dirname(onnx_path)):
            shutil.copy(src, out_dir)
    _update_info(out_dir, {"files": {os.path.basename(out_path): {
        "sha256": sha256_file(out_path), "bytes": os.path.getsize(out_path), "from": os.path.basename(onnx_path),
        "quantization": details, "versions": _versions()}}})
    return out_path


# ---------------------------------------------------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------------------------------------------------

def find_beside(onnx_path, name):
    """``name`` in the ONNX file's directory or its parent (the HF ``onnx/`` subfolder layout); None if absent."""
    for d in (os.path.dirname(onnx_path), os.path.dirname(os.path.dirname(onnx_path))):
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    return None


def preprocess(processor, paths):
    """``N x 3 x H x W`` float32 array of ``paths`` through ``processor`` (one image at a time, as the API does)."""
    from PIL import Image
    out = [processor(Image.open(p).convert("RGB"), return_tensors="np")["pixel_values"][0] for p in paths]
    return np.stack(out).astype(np.float32)


def onnx_param_count(path):
    """Elements in the graph's initializers (int8 weights count as elements, like their fp32 originals)."""
    import onnx
    m = onnx.load(path, load_external_data=False)
    return int(sum(int(np.prod(t.dims)) for t in m.graph.initializer))


def id2label_order(config_path):
    """Column order that puts logits as ``[correct, incorrect]``, from a config.json's ``id2label``."""
    with open(config_path) as f:
        id2label = {int(k): v for k, v in json.load(f)["id2label"].items()}
    if set(id2label.values()) != set(CLASSES):
        raise ValueError(f"{config_path}: unexpected id2label {id2label}")
    inv = {v: k for k, v in id2label.items()}
    return [inv[c] for c in CLASSES]


def has_dynamic_quantization(path):
    """True if the ONNX graph quantizes activations at run time (``DynamicQuantizeLinear``).

    Such a graph computes one activation scale per tensor over the WHOLE input, batch included, so a crop's output
    depends on which other crops share its batch. The API scores one crop per call, so these graphs are scored at
    batch 1.
    """
    import onnx
    m = onnx.load(path, load_external_data=False)
    return any(n.op_type == "DynamicQuantizeLinear" for n in m.graph.node)


def effective_score_batch(path, requested):
    """The batch size to score ``path`` with: ``requested``, except 1 for a dynamically quantized graph (warns)."""
    if requested != 1 and has_dynamic_quantization(path):
        warnings.warn(f"{path} quantizes activations per call (DynamicQuantizeLinear), so its scores depend on the "
                      f"batch; scoring at batch 1 as the API does instead of --score-batch {requested}")
        return 1
    return requested


def make_session(path, threads):
    """onnxruntime CPU session with ``threads`` intra-op threads, 1 inter-op thread and all graph optimizations."""
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])


def _batches(x, batch):
    for i in range(0, len(x), batch):
        yield x[i:i + batch]


def _sample(pixels, b):
    """The first ``b`` crops, tiled if there are fewer: a fixed input for latency."""
    reps = -(-b // len(pixels))
    return np.concatenate([pixels] * reps)[:b] if reps > 1 else pixels[:b]


def bench_torch(model, revision, pixels, device, batch_sizes, warmup, runs, score_batch=1):
    """Logits on every crop and median latency per batch size for the PyTorch fp32 model on ``device``."""
    import torch
    from transformers import AutoModelForImageClassification
    net = AutoModelForImageClassification.from_pretrained(model, revision=revision)   # fp32, as the API loads it
    net = net.to(device).eval()
    order = [net.config.label2id[c] for c in CLASSES]
    sync = torch.mps.synchronize if device == "mps" else (lambda: None)

    def run(x):
        with torch.no_grad():
            out = net(pixel_values=torch.from_numpy(x).to(device)).logits
        sync()
        return out
    logits = np.concatenate([run(b).cpu().numpy() for b in _batches(pixels, score_batch)])[:, order]
    lat = {b: median_ms(lambda x=_sample(pixels, b): run(x), warmup, runs) for b in batch_sizes}
    n_params = int(sum(p.numel() for p in net.parameters()))
    del net
    return logits, lat, n_params


def bench_onnx(path, pixels, threads, batch_sizes, warmup, runs, order, score_batch=1):
    """Logits on every crop, median latency per batch size, and (dynamic graphs only) batch dependence.

    Returns:
        ``(logits, latency_by_batch, score_batch_used, batch_check)``. ``batch_check`` is the max |Δ softmax| between
        scoring at ``score_batch_used`` (1 for a dynamically quantized graph) and at ``BATCH_CHECK``, or ``None`` for
        a graph without run-time activation quantization.
    """
    sess = make_session(path, threads)
    name = sess.get_inputs()[0].name

    def run(x):
        return sess.run(None, {name: x})[0]

    def score(batch):
        return np.concatenate([run(b) for b in _batches(pixels, batch)])[:, order]
    used = effective_score_batch(path, score_batch)
    logits = score(used)
    check = None
    if has_dynamic_quantization(path):
        check = float(np.abs(softmax(score(BATCH_CHECK)) - softmax(logits)).max())
    lat = {b: median_ms(lambda x=_sample(pixels, b): run(x), warmup, runs) for b in batch_sizes}
    del sess
    return logits, lat, used, check


def _model_file_size(model, revision):
    """Bytes of the model's weight file(s): local dir, or the HF cache snapshot for (id, revision)."""
    d = model if os.path.isdir(model) else None
    if d is None:
        from huggingface_hub import snapshot_download
        d = snapshot_download(model, revision=revision, allow_patterns=["config.json", "preprocessor_config.json",
                                                                        "model.safetensors"])
    files = glob.glob(os.path.join(d, "*.safetensors")) or glob.glob(os.path.join(d, "*.bin"))
    return sum(os.path.getsize(f) for f in files)


def bench(args):
    """Run the benchmark described in the module docstring; write CSV + markdown + per-crop scores to ``--out``."""
    import gc

    import torch
    from transformers import AutoImageProcessor

    torch.set_num_threads(args.threads)
    crops = list_crops(args.crops, args.split)
    n_private = 0
    if args.cities_json:
        with open(args.cities_json) as f:
            crops, n_private = public_crops(crops, json.load(f))
        print(f"{len(crops)} crops from public deployments; {n_private} from other deployments dropped", flush=True)
    else:
        warnings.warn("no --cities-json: scoring every crop, including any from deployments that are not public")
    accuracy_map = None
    if args.accuracy_map:
        with open(args.accuracy_map) as f:
            accuracy_map = json.load(f)
    batch_sizes = [int(b) for b in args.batch_sizes.split(",")]
    os.makedirs(args.out, exist_ok=True)
    common = {"label_type": args.label_type, "n_crops": len(crops), "n_crops_dropped_not_public": n_private,
              "crop_filter": "public" if args.cities_json else "none", "threads": args.threads,
              "machine": f"{platform.machine()} {_cpu_name()}", "gate_rule": "map" if accuracy_map else "raw>=0.92"}

    cache = {}

    def pixels_for(processor):
        key = json.dumps(processor.to_dict(), sort_keys=True, default=str)
        if key not in cache:
            print(f"preprocessing {len(crops)} crops ({type(processor).__name__})", flush=True)
            cache[key] = preprocess(processor, crops.crop_path)
        return cache[key]

    ref_proc = AutoImageProcessor.from_pretrained(args.model, revision=args.revision, use_fast=False)
    ref_pixels = pixels_for(ref_proc)
    size = _model_file_size(args.model, args.revision)
    rows, scores, ref_logits = [], {"label_uid": crops.label_uid, "true": crops.label_class}, None
    devices = ["cpu"] + (["mps"] if args.mps else [])
    variants = [(f"torch-fp32-{d}", "torch", d) for d in devices] + [(n, "onnx", p) for n, p in args.onnx]
    for name, kind, spec in variants:
        load = os.getloadavg()[0]
        print(f"[{args.label_type}] {name} (load {load:.1f})", flush=True)
        if kind == "torch":
            logits, lat, n_params = bench_torch(args.model, args.revision, ref_pixels, spec, batch_sizes,
                                                args.warmup, args.runs, args.score_batch)
            nbytes, path = size, f"{args.model}@{(args.revision or 'main')[:7]}"
            used, check = args.score_batch, None
        else:
            pre = find_beside(spec, "preprocessor_config.json")
            cfg = find_beside(spec, "config.json")
            if not (pre and cfg):
                raise FileNotFoundError(f"{spec}: need config.json and preprocessor_config.json beside it or one up")
            proc = AutoImageProcessor.from_pretrained(os.path.dirname(pre), use_fast=False)
            logits, lat, used, check = bench_onnx(spec, pixels_for(proc), args.threads, batch_sizes, args.warmup,
                                                  args.runs, id2label_order(cfg), args.score_batch)
            n_params, nbytes, path = onnx_param_count(spec), os.path.getsize(spec), spec
        if ref_logits is None:
            ref_logits = logits
        row = {**common, "variant": name, "path": path, "size_mb": nbytes / 1e6, "params_m": n_params / 1e6,
               "load_avg_1m": round(load, 2), "score_batch": used, f"b{BATCH_CHECK}_vs_b1_max_abs_dprob": check}
        row.update({f"lat_b{b}_ms": v for b, v in lat.items()})
        row.update({f"ms_per_crop_b{b}": v / b for b, v in lat.items()})
        if args.no_accuracy:
            row.update(dict.fromkeys(["accuracy", "balanced_accuracy", "macro_f1", "auc", "r_at_p95_correct",
                                      "r_at_p95_incorrect"]))
        else:
            row.update(accuracy_row(crops.label_class.values, logits))
        row.update({f"vs_ref_{k}": v for k, v in parity(ref_logits, logits, accuracy_map).items() if k != "n"})
        rows.append(row)
        p = softmax(logits)
        scores[f"{name}_p_incorrect"] = p[:, 1]
        pd.DataFrame(rows).to_csv(os.path.join(args.out, f"bench_{args.label_type}.csv"), index=False)  # each step
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    df = pd.DataFrame(rows)
    pd.DataFrame(scores).to_csv(os.path.join(args.out, f"scores_{args.label_type}.csv"), index=False)
    with open(os.path.join(args.out, f"bench_{args.label_type}.md"), "w") as f:
        f.write(markdown_table(df, TABLE_COLUMNS_SHORT + [c for c in df.columns if c.startswith("lat_")]))
    print(markdown_table(df, TABLE_COLUMNS_SHORT))
    return df


TABLE_COLUMNS_SHORT = ["label_type", "variant", "size_mb", "params_m", "accuracy", "macro_f1", "auc",
                       "r_at_p95_correct", "r_at_p95_incorrect", "vs_ref_max_abs_dprob", "vs_ref_verdict_flips",
                       "vs_ref_vote_changes"]


def _cpu_name():
    if platform.system() == "Darwin":
        import subprocess
        try:
            return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True,
                                  check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            pass
    return platform.processor() or "unknown CPU"


def table(paths, columns=None):
    """Concatenate bench CSVs into one markdown table (``columns`` default: the short set plus latency columns)."""
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    columns = columns or TABLE_COLUMNS_SHORT + sorted(c for c in df.columns if c.startswith("lat_"))
    return markdown_table(df, columns)


# ---------------------------------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------------------------------

def _named_path(s):
    name, sep, path = s.partition("=")
    if not sep or not name or not path:
        raise argparse.ArgumentTypeError(f"expected NAME=PATH, got {s!r}")
    return name, path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("export", help="checkpoint -> model.onnx (+ config, preprocessor, export_info.json)")
    p.add_argument("--model", required=True, help="local checkpoint dir or Hugging Face model id")
    p.add_argument("--revision", default=None, help="Hugging Face revision (pin a full commit sha)")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--opset", type=int, default=OPSET, help=f"ONNX opset (default {OPSET})")

    p = sub.add_parser("quantize", help="model.onnx -> model_int8.onnx (dynamic int8)")
    p.add_argument("onnx", help="fp32 ONNX file written by export")
    p.add_argument("--out", default=None, help="output file (default: model_int8.onnx beside the input)")
    p.add_argument("--method", choices=QUANT_METHODS, default="dynamic",
                   help="dynamic: int8 weights + run-time uint8 activations, integer matmuls (model_int8.onnx); "
                        "weight-only: int8 weights in blocks, fp32 activations (model_int8_wo.onnx; needs onnx-ir)")
    p.add_argument("--per-channel", action="store_true",
                   help="dynamic only: per-output-channel weight scales (default: one scale per tensor)")
    p.add_argument("--block-size", type=int, default=128, help="weight-only only: weights per scale (default 128)")

    p = sub.add_parser("bench", help="score + time the PyTorch reference and ONNX variants on the same crops")
    p.add_argument("--label-type", required=True, help="label type name, used in output file names and rows")
    p.add_argument("--crops", required=True,
                   help="directory laid out like the HF dataset: <crops>/<split>/<class>/<city>_<label_id>.webp")
    p.add_argument("--split", default="test", help="split subdirectory of --crops (default test)")
    p.add_argument("--model", required=True, help="PyTorch reference: local checkpoint dir or HF model id")
    p.add_argument("--revision", default=None, help="Hugging Face revision of --model")
    p.add_argument("--onnx", action="append", type=_named_path, default=[], metavar="NAME=PATH",
                   help="an ONNX variant; config.json and preprocessor_config.json must sit beside it or one "
                        "directory up (repeatable)")
    p.add_argument("--accuracy-map", default=None,
                   help="sidewalk-ai-api accuracy_mappings/<type>.json, for served-vote parity (else raw >= 0.92)")
    p.add_argument("--mps", action="store_true", help="also time and score the PyTorch model on Apple MPS")
    p.add_argument("--threads", type=int, default=4, help="intra-op threads for torch and onnxruntime (default 4)")
    p.add_argument("--batch-sizes", default="1,16", help="comma list of batch sizes to time (default 1,16)")
    p.add_argument("--score-batch", type=int, default=1,
                   help="batch size for scoring the crops (default 1, the API's one crop per call). Graphs with "
                        "run-time activation quantization (DynamicQuantizeLinear) are always scored at 1, with a "
                        "warning, because their outputs depend on the other crops in the batch")
    p.add_argument("--cities-json", default=None,
                   help="saved /v3/api/cities response: keep only crops from public deployments (strongly "
                        "recommended; the published test splits include crops from deployments that are private)")
    p.add_argument("--warmup", type=int, default=3, help="untimed calls before timing (default 3)")
    p.add_argument("--runs", type=int, default=10, help="timed calls; the median is reported (default 10)")
    p.add_argument("--no-accuracy", action="store_true",
                   help="parity and latency only (e.g. CurbRamp, whose published dataset is the crosswalk one)")
    p.add_argument("--out", required=True, help="directory for bench_<type>.csv/.md and scores_<type>.csv")

    p = sub.add_parser("table", help="bench CSVs -> one markdown table on stdout")
    p.add_argument("csvs", nargs="+")
    p.add_argument("--columns", default=None,
                   help="comma list of CSV columns to print (default: size, accuracy, parity and latency columns)")

    args = parser.parse_args(argv)
    if args.cmd == "export":
        print(export_model(args.model, args.out, args.revision, args.opset))
    elif args.cmd == "quantize":
        print(quantize_model(args.onnx, args.out, args.per_channel, args.method, args.block_size))
    elif args.cmd == "bench":
        return bench(args)
    else:
        print(table(args.csvs, args.columns.split(",") if args.columns else None), end="")


if __name__ == "__main__":
    main()
