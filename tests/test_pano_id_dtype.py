"""Static guard for issue #114: every ``read_csv`` whose ``usecols`` can select ``pano_id`` must type it as a string.

Mapillary pano_ids are all digits, so a reader that lets pandas infer the column gets ``int64`` (or ``float64`` when a
cell is blank, which then prints as ``'<digits>.0'``), and the store path ``pano_id[:2]``, cross-frame merges and the
panorama bootstrap groups all go wrong. The fix is a dtype at the reader: ``dtype=PANO_ID_DTYPE`` (from
``data_generation``) or ``"pano_id": str`` in the reader's dtype dict.

What counts as typed: ``dtype=str`` / ``object`` / ``"str"`` / ``"object"`` / ``"string"``; ``PANO_ID_DTYPE`` (pinned
below to ``data_generation``'s); a dict literal whose **last** binding of ``pano_id`` is a string type, as a
``"pano_id"`` key, a ``{c: str for c in X}`` over names that include it, or a spread (``{**PANO_ID_DTYPE, ...}``); or a
name the module binds exactly once to such a dict.

The guard fails closed. It reads ``usecols`` as a literal list/tuple/set, a name the module binds only with plain
``NAME = expr`` to expressions it can read, ``list``/``tuple``/``set``/``sorted`` of one, ``+`` or ``|`` of two, a
comprehension (its iterables), or ``lambda c: c in X``. Any other form (``lambda c: c not in X``, ``mod.COLS``, a
function parameter, an annotated or unpacked binding, ``**kwargs``) counts as able to select ``pano_id``, so the read
must type it; so does a dtype it cannot read. ``read_table`` is checked like ``read_csv``.

Limits, on purpose: reads **without** ``usecols`` (or ``**kwargs``) are not checked, because the AST cannot tell which
files have a ``pano_id`` column (most do not). For those, ``data_generation.pano_store_path`` raises ``TypeError`` on
a non-str pano_id with a message that names the cause, and ``ai_labels_eval crops`` stops after
``MAX_CONSECUTIVE_ERRORS`` failed labels in a row. Tests are not scanned. The files scanned are the ones git tracks
(every ``.py`` file when ``root`` is not the top of a git checkout), so a local venv or scratch copy is never parsed.
"""
import ast
import pathlib
import subprocess

import data_generation as dg

ROOT = pathlib.Path(__file__).resolve().parent.parent
SKIP = ("tests/", "Depth-Anything-V2/")
READERS = ("read_csv", "read_table")
SAFE_CONSTS = {"PANO_ID_DTYPE"}
STR_TYPES = {"str", "object", "string"}
UNKNOWN = None  # a form the guard cannot read


def _mark_unknown(out, target):
    for sub in ast.walk(target):
        if isinstance(sub, ast.Name):
            out[sub.id] = UNKNOWN


def _bindings(tree):
    """name -> every value expression assigned to it anywhere in the module; UNKNOWN when any binding is not a plain
    ``NAME = expr`` (annotated, augmented, unpacked, a loop or ``with`` target, a parameter, an import...)."""
    out = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Assign):
            for t in n.targets:
                if isinstance(t, ast.Name):
                    if out.get(t.id, []) is not UNKNOWN:
                        out.setdefault(t.id, []).append(n.value)
                else:
                    _mark_unknown(out, t)
        elif isinstance(n, (ast.AnnAssign, ast.AugAssign, ast.NamedExpr, ast.For, ast.AsyncFor, ast.comprehension)):
            _mark_unknown(out, n.target)
        elif isinstance(n, ast.withitem) and n.optional_vars is not None:
            _mark_unknown(out, n.optional_vars)
        elif isinstance(n, ast.arg):
            out[n.arg] = UNKNOWN
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            for a in n.names:
                out[(a.asname or a.name).split(".")[0]] = UNKNOWN
    return out


def _names(node, names, seen=frozenset()):
    """The column names a ``usecols`` argument (or a dict comprehension's iterable) can hold, or UNKNOWN."""
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        found = set()
        for e in node.elts:
            if isinstance(e, ast.Constant) and isinstance(e.value, str):
                found.add(e.value)
                continue
            sub = _names(e.value if isinstance(e, ast.Starred) else e, names, seen)
            if sub is UNKNOWN:
                return UNKNOWN
            found |= sub
        return found
    if isinstance(node, ast.Name):
        values = names.get(node.id)
        if not values or node.id in seen:  # unbound here (a builtin), bound unreadably, or recursive
            return UNKNOWN
        found = set()
        for v in values:
            sub = _names(v, names, seen | {node.id})
            if sub is UNKNOWN:
                return UNKNOWN
            found |= sub
        return found
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id in ("list", "tuple", "set", "sorted") and len(node.args) == 1 and not node.keywords):
        return _names(node.args[0], names, seen)
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.BitOr)):
        a, b = _names(node.left, names, seen), _names(node.right, names, seen)
        return UNKNOWN if a is UNKNOWN or b is UNKNOWN else a | b
    if isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp)):
        found = set()
        for gen in node.generators:  # every name an iterable holds (a filter only narrows it)
            sub = _names(gen.iter, names, seen)
            if sub is UNKNOWN:
                return UNKNOWN
            found |= sub
        return found
    if (isinstance(node, ast.Lambda) and isinstance(node.body, ast.Compare) and len(node.body.ops) == 1
            and isinstance(node.body.ops[0], ast.In)):
        return _names(node.body.comparators[0], names, seen)
    return UNKNOWN


def _is_str_type(node):
    if isinstance(node, ast.Name):
        return node.id in ("str", "object")
    return isinstance(node, ast.Constant) and node.value in STR_TYPES


def _pano_id_type(node, names, seen=frozenset()):
    """What ``dtype=node`` does to ``pano_id``: ``"str"``, ``"other"`` (another type), ``"absent"`` (a dict without
    it: inferred) or UNKNOWN."""
    if node is None:
        return "absent"
    if _is_str_type(node):
        return "str"
    if isinstance(node, ast.Name) and node.id in SAFE_CONSTS or isinstance(node, ast.Attribute) and \
            node.attr in SAFE_CONSTS:
        return "str"
    if isinstance(node, ast.Name):  # a dict the module binds exactly once
        values = names.get(node.id)
        if not values or len(values) != 1 or node.id in seen:
            return UNKNOWN
        return _pano_id_type(values[0], names, seen | {node.id})
    if isinstance(node, ast.Dict):
        state = "absent"  # the last binding of pano_id wins, as in the dict itself
        for k, v in zip(node.keys, node.values):
            if k is None:
                sub = _pano_id_type(v, names, seen)
                if sub != "absent":
                    state = sub
            elif isinstance(k, ast.Constant):
                if k.value == "pano_id":
                    state = "str" if _is_str_type(v) else "other"
            else:
                state = UNKNOWN  # a computed key
        return state
    if isinstance(node, ast.DictComp) and len(node.generators) == 1 and isinstance(node.key, ast.Name) \
            and isinstance(node.generators[0].target, ast.Name) and node.key.id == node.generators[0].target.id:
        cols = _names(node.generators[0].iter, names, seen)
        if cols is UNKNOWN:
            return UNKNOWN
        return ("str" if _is_str_type(node.value) else "other") if "pano_id" in cols else "absent"
    return UNKNOWN if not isinstance(node, (ast.Name, ast.Attribute, ast.Constant)) else "other"


def _py_files(root):
    """Paths (relative to ``root``) of the ``.py`` files git tracks; every ``.py`` file when ``root`` is not the top
    of a git checkout (the self-tests below)."""
    root = pathlib.Path(root).resolve()
    top = subprocess.run(["git", "-C", str(root), "rev-parse", "--show-toplevel"], capture_output=True, text=True)
    if top.returncode == 0 and pathlib.Path(top.stdout.strip()).resolve() == root:
        ls = subprocess.run(["git", "-C", str(root), "ls-files", "-z", "--", "*.py"], capture_output=True, text=True,
                            check=True)
        return sorted(p for p in ls.stdout.split("\0") if p and (root / p).is_file())
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*.py"))


def untyped_pano_id_reads(root=ROOT):
    """``path:line`` of every non-test ``read_csv``/``read_table`` whose ``usecols`` (or ``**kwargs``) can select
    ``pano_id`` without a dtype that reads it as a string."""
    bad = []
    for rel in _py_files(root):
        if rel.startswith(SKIP) or "/." in "/" + rel:
            continue
        tree = ast.parse((pathlib.Path(root) / rel).read_text(encoding="utf-8"))
        names = _bindings(tree)
        for n in ast.walk(tree):
            if isinstance(n, ast.Call) and getattr(n.func, "attr", getattr(n.func, "id", "")) in READERS:
                kw = {k.arg: k.value for k in n.keywords if k.arg}
                if "usecols" in kw:
                    cols = _names(kw["usecols"], names)
                    selects = cols is UNKNOWN or "pano_id" in cols
                else:
                    selects = any(k.arg is None for k in n.keywords)  # **kwargs may carry usecols
                if selects and _pano_id_type(kw.get("dtype"), names) != "str":
                    bad.append(f"{rel}:{n.lineno}")
    return bad


def test_every_usecols_read_of_pano_id_types_it():
    bad = untyped_pano_id_reads()
    assert not bad, ("these read_csv calls can select pano_id without a str dtype for it; all-digit Mapillary ids "
                     "would come back as numbers. Pass dtype=PANO_ID_DTYPE (data_generation) or add "
                     "'\"pano_id\": str' to the dtype dict: " + ", ".join(bad))


def test_pano_id_dtype_is_str():
    # the guard trusts the name PANO_ID_DTYPE, so pin what data_generation binds to it
    assert dg.PANO_ID_DTYPE == {"pano_id": str}


def _check(tmp_path, src):
    (tmp_path / "m.py").write_text(src)
    return untyped_pano_id_reads(tmp_path)


def test_the_guard_sees_each_usecols_form(tmp_path):
    for usecols in ('["label_uid", "pano_id"]', "COLS", "lambda c: c in COLS", "[c for c in COLS]",
                    'COLS + ["x"]', "list(COLS)", "lambda c: c in set(COLS)", "ALIAS", "[*COLS, 'x']"):
        src = (f'COLS = ["label_uid", "pano_id"]\nALIAS = COLS + ["y"]\nimport pandas as pd\n'
               f'pd.read_csv("f", usecols={usecols})\n')
        assert _check(tmp_path, src) == ["m.py:4"], usecols


def test_the_guard_fails_closed_on_forms_it_cannot_read(tmp_path):
    # PR #115 review R3: each of these can select pano_id; the guard cannot tell, so it asks for the dtype
    cases = {"lambda c: c not in DROP": 'DROP = ["x"]\n', "mod.COLS": "import mod\n",
             "COLS": 'COLS: list = ["pano_id"]\n', "A": 'A, B = ["pano_id"], ["x"]\n',
             "cols": "def f(cols):\n    pass\n", "lambda c: c.startswith('pano')": "", "WHAT": ""}
    for usecols, pre in cases.items():
        src = f'{pre}import pandas as pd\npd.read_csv("f", usecols={usecols})\n'
        assert _check(tmp_path, src) == [f"m.py:{src.count(chr(10))}"], usecols
    src = 'import pandas as pd\nkw = {}\npd.read_csv("f", **kw)\npd.read_table("f", usecols=["pano_id"])\n'
    assert _check(tmp_path, src) == ["m.py:3", "m.py:4"]
    assert _check(tmp_path, 'import pandas as pd\nkw = {}\npd.read_csv("f", dtype=str, **kw)\n') == []
    assert _check(tmp_path, 'import pandas as pd\npd.read_csv("f", usecols=["label_uid"])\n') == []


def test_the_guard_accepts_each_dtype_form(tmp_path):
    for dtype in ("str", "object", '"str"', '"string"', '{"pano_id": str}', '{"pano_id": "string"}', "PANO_ID_DTYPE",
                  "dg.PANO_ID_DTYPE", '{"label_id": str, **dg.PANO_ID_DTYPE}', "MY_DTYPES", '{"x": int, **MY_DTYPES}',
                  "{c: str for c in COLS}"):
        src = (f'import pandas as pd\nMY_DTYPES = {{"pano_id": str}}\nCOLS = ["pano_id", "x"]\n'
               f'pd.read_csv("f", usecols=["pano_id"], dtype={dtype})\n')
        assert _check(tmp_path, src) == [], dtype


def test_the_guard_refuses_dtypes_that_do_not_type_pano_id_as_str(tmp_path):
    # PR #115 review R3: a "pano_id" key is not enough; its value must be a string type, and the last binding wins
    for dtype in ('{"x": str}', '{"pano_id": int}', '{"pano_id": "float64"}', '{**PANO_ID_DTYPE, "pano_id": int}',
                  '{"pano_id": str, **UNSEEN}', "SPLIT_DTYPES", "other.SPLIT_DTYPES", "TWICE", "int", '"int64"',
                  "{c: str for c in ['x']}", "{c: int for c in COLS}"):
        src = (f'import pandas as pd\nTWICE = {{"pano_id": str}}\nTWICE = {{"pano_id": int}}\nCOLS = ["pano_id"]\n'
               f'pd.read_csv("f", usecols=["pano_id"], dtype={dtype})\n')
        assert _check(tmp_path, src) == ["m.py:5"], dtype


def test_the_guard_reads_only_files_git_tracks(tmp_path):
    # PR #115 review R6: a local venv or scratch copy inside the checkout is not parsed
    src = 'import pandas as pd\npd.read_csv("f", usecols=["pano_id"])\n'
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "tracked.py").write_text(src)
    (tmp_path / "venv").mkdir()
    (tmp_path / "venv" / "lib.py").write_text(src)
    (tmp_path / "venv" / "broken.py").write_text("def (:\n")  # would fail ast.parse if it were read
    subprocess.run(["git", "-C", str(tmp_path), "add", "tracked.py"], check=True)
    assert untyped_pano_id_reads(tmp_path) == ["tracked.py:2"]
