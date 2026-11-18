"""Static guard for issue #114: every ``read_csv`` whose ``usecols`` names ``pano_id`` must type it as a string.

Mapillary pano_ids are all digits, so a reader that lets pandas infer the column gets ``int64`` (or ``float64`` when a
cell is blank, which then prints as ``'<digits>.0'``), and the store path ``pano_id[:2]``, cross-frame merges and the
panorama bootstrap groups all go wrong. The fix is a dtype at the reader: ``dtype=PANO_ID_DTYPE`` (from
``data_generation``) or ``"pano_id": str`` in the reader's dtype dict.

What counts as typed: ``dtype=str`` / ``object``; a dict with a ``"pano_id"`` key; a dict spreading a safe constant
(``{**PANO_ID_DTYPE, ...}``); or a safe constant itself (``PANO_ID_DTYPE``, ``SPLIT_DTYPES``).

What counts as naming ``pano_id`` in ``usecols``: a literal list/tuple/set, a name bound to one anywhere in the module,
``lambda c: c in NAME`` (or a literal), and a comprehension or ``+`` expression over such names or literals.

Limits, on purpose: reads **without** ``usecols`` are not checked, because the AST cannot tell which files have a
``pano_id`` column (most do not). For those, ``data_generation.pano_store_path`` raises ``TypeError`` on a non-str
pano_id, which catches the failure that matters (a store path built from a number). Tests are not scanned.
"""
import ast
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
SKIP = ("tests/", "Depth-Anything-V2/")
SAFE_CONSTS = {"PANO_ID_DTYPE", "SPLIT_DTYPES"}


def _literal_strs(node):
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return {e.value for e in node.elts if isinstance(e, ast.Constant) and isinstance(e.value, str)}
    return None


def _bound_lists(tree):
    """name -> the strings of every literal list/tuple/set assigned to that name anywhere in the module."""
    out = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
            s = _literal_strs(n.value)
            if s is not None:
                out.setdefault(n.targets[0].id, set()).update(s)
    return out


def _usecols_names(node, names):
    """The column names a ``usecols`` argument can select, as far as the AST shows."""
    s = _literal_strs(node)
    if s is not None:
        return s
    if isinstance(node, ast.Name):
        return names.get(node.id, set())
    if isinstance(node, ast.Lambda) and isinstance(node.body, ast.Compare):
        for c in node.body.comparators:
            if isinstance(c, ast.Name):
                return names.get(c.id, set())
            s = _literal_strs(c)
            if s is not None:
                return s
    if isinstance(node, (ast.ListComp, ast.BinOp)):
        found = set()
        for sub in ast.walk(node):
            if isinstance(sub, ast.Name):
                found |= names.get(sub.id, set())
            s = _literal_strs(sub)
            if s:
                found |= s
        return found
    return set()


def _dtype_types_pano_id(node):
    if node is None:
        return False
    if isinstance(node, ast.Name):
        return node.id in ("str", "object") or node.id in SAFE_CONSTS
    if isinstance(node, ast.Attribute):
        return node.attr in SAFE_CONSTS
    if isinstance(node, ast.Dict):
        for k, v in zip(node.keys, node.values):
            if k is None and _dtype_types_pano_id(v):  # {**PANO_ID_DTYPE, ...}
                return True
            if isinstance(k, ast.Constant) and k.value == "pano_id":
                return True
    return False


def untyped_pano_id_reads(root=ROOT):
    """``path:line`` of every non-test ``read_csv`` whose ``usecols`` names ``pano_id`` without a dtype for it."""
    bad = []
    for p in sorted(pathlib.Path(root).rglob("*.py")):
        rel = p.relative_to(root).as_posix()
        if rel.startswith(SKIP) or "/." in "/" + rel:
            continue
        tree = ast.parse(p.read_text(encoding="utf-8"))
        names = _bound_lists(tree)
        for n in ast.walk(tree):
            if isinstance(n, ast.Call) and getattr(n.func, "attr", getattr(n.func, "id", "")) == "read_csv":
                kw = {k.arg: k.value for k in n.keywords if k.arg}
                if ("usecols" in kw and "pano_id" in _usecols_names(kw["usecols"], names)
                        and not _dtype_types_pano_id(kw.get("dtype"))):
                    bad.append(f"{rel}:{n.lineno}")
    return bad


def test_every_usecols_read_of_pano_id_types_it():
    bad = untyped_pano_id_reads()
    assert not bad, ("these read_csv calls select pano_id without a dtype for it; all-digit Mapillary ids would come "
                     "back as numbers. Pass dtype=PANO_ID_DTYPE (data_generation) or add '\"pano_id\": str' to the "
                     "dtype dict: " + ", ".join(bad))


def _check(tmp_path, src):
    (tmp_path / "m.py").write_text(src)
    return untyped_pano_id_reads(tmp_path)


def test_the_guard_sees_each_usecols_form(tmp_path):
    for usecols in ('["label_uid", "pano_id"]', "COLS", "lambda c: c in COLS", "[c for c in COLS]",
                    'COLS + ["x"]'):
        src = f'COLS = ["label_uid", "pano_id"]\nimport pandas as pd\npd.read_csv("f", usecols={usecols})\n'
        assert _check(tmp_path, src) == ["m.py:3"], usecols


def test_the_guard_accepts_each_dtype_form(tmp_path):
    for dtype in ("str", "object", '{"pano_id": str}', "PANO_ID_DTYPE", "dg.PANO_ID_DTYPE",
                  '{"label_id": str, **dg.PANO_ID_DTYPE}', "SPLIT_DTYPES"):
        src = f'import pandas as pd\npd.read_csv("f", usecols=["pano_id"], dtype={dtype})\n'
        assert _check(tmp_path, src) == [], dtype
    assert _check(tmp_path, 'import pandas as pd\npd.read_csv("f", usecols=["pano_id"], dtype={"x": str})\n') \
        == ["m.py:2"]
    assert _check(tmp_path, 'import pandas as pd\npd.read_csv("f", usecols=["label_uid"])\n') == []
