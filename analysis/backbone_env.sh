#!/bin/bash
# Issue #119 (docs/backbone-2026-11.md): build the study env on klone's login node, an overlay venv over the round-3
# conda env that never writes into that env.
#
#   analysis/backbone_env.sh            # from the repo root of a checkout of this branch
#
# The venv sees the base env's packages (--system-site-packages) and adds requirements-backbone.txt
# (transformers 4.56.2, tokenizers, timm). Every other base package is held at its installed version by a
# constraints file built from `pip list --format=freeze` of the new venv, which lists the base env's packages too
# (transformers, tokenizers and timm left out).
# timm goes in with --no-deps when the base env has no torchvision, so nothing pulls a second torch in.
# The check at the end requires torch 2.7.1 and pandas 2.3.1 imported from $BASE_ENV and transformers 4.56.2,
# tokenizers 0.22.2 and timm 1.0.31 from $VENV; it exits 1 otherwise. Idempotent: an env that already passes the check
# is left alone (all five are imported, so a venv that scrubbed's purge has partly deleted fails it and is rebuilt).
#
# Variables (defaults in brackets):
#   OUT       [/gscratch/scrubbed/$USER/backbone-comparison-2026-11] constraints.txt, pip_freeze.txt
#   BASE_ENV  [/gscratch/makelab/$USER/envs/sidewalk-validator-ai] the round-3 env (read only)
#   VENV      [/gscratch/scrubbed/$USER/envs/sva-bb] the overlay
#   PIP_CACHE_DIR [/gscratch/scrubbed/$USER/pip_cache] (home is a 10 GB quota)
set -euo pipefail
OUT="${OUT:-/gscratch/scrubbed/$USER/backbone-comparison-2026-11}"
BASE_ENV="${BASE_ENV:-/gscratch/makelab/$USER/envs/sidewalk-validator-ai}"
VENV="${VENV:-/gscratch/scrubbed/$USER/envs/sva-bb}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/gscratch/scrubbed/$USER/pip_cache}" PIP_DISABLE_PIP_VERSION_CHECK=1
REPO="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$OUT"

check() {
  [ -x "$VENV/bin/python" ] || return 1
  "$VENV/bin/python" - "$BASE_ENV" "$VENV" <<'PY'
import os, sys
base, venv = (os.path.realpath(p) for p in sys.argv[1:3])
import pandas, timm, tokenizers, torch, transformers
want = {"torch": ("2.7.1", torch, base), "pandas": ("2.3.1", pandas, base), "transformers": ("4.56.2", transformers, venv),
        "tokenizers": ("0.22.2", tokenizers, venv), "timm": ("1.0.31", timm, venv)}
bad = []
for name, (version, mod, root) in want.items():
    got = mod.__version__.split("+")[0]
    where = os.path.realpath(mod.__file__)
    print(f"{name} {mod.__version__} from {where}")
    if got != version:
        bad.append(f"{name} {got} != {version}")
    if not where.startswith(root + os.sep):
        bad.append(f"{name} is imported from {where}, not under {root}")
if bad:
    print("; ".join(bad), file=sys.stderr)
    sys.exit(1)
PY
}

if check 2>/dev/null; then
  echo "$VENV already passes the version check"
else
  [ -x "$BASE_ENV/bin/python" ] || { echo "no $BASE_ENV/bin/python" >&2; exit 2; }
  "$BASE_ENV/bin/python" -m venv --system-site-packages "$VENV"
  # The venv's own pip lists the base env's packages too (system site packages), whether or not the base env has pip.
  "$VENV/bin/pip" list --format=freeze \
      | grep -v -i -E '^(transformers|tokenizers|timm|pip|setuptools)==' > "$OUT/constraints.txt"
  grep -v -E '^(timm)' "$REPO/requirements-backbone.txt" > "$OUT/requirements-backbone.notimm.txt"
  "$VENV/bin/pip" install -c "$OUT/constraints.txt" -r "$OUT/requirements-backbone.notimm.txt"
  TIMM="$(grep -E '^timm==' "$REPO/requirements-backbone.txt")"
  if "$BASE_ENV/bin/python" -c "import torchvision" 2>/dev/null; then
    "$VENV/bin/pip" install -c "$OUT/constraints.txt" "$TIMM"
  else
    "$VENV/bin/pip" install --no-deps "$TIMM"
  fi
  check
fi
"$VENV/bin/pip" freeze > "$OUT/pip_freeze.txt"
sha256sum "$OUT/pip_freeze.txt"
