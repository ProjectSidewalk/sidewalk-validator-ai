#!/usr/bin/env bash
# Round-2 recount (issue #85): build, run (once) and accept the author-exclusion pull. Runs LOCALLY; the only
# remote entry point of the recount.
#
# Modes:
#   (default)   build only: write $D/remote_bundle.sh (pull_author_excluded.sh + pull_exclusion_changes.sh, each in
#               a subshell, with section markers and a status trailer) and print its sha256. No ssh.
#   --connect   build, then ONE ssh call to makelab1 that runs the bundle, then accept (below). No loop, no retry,
#               no fallback host: on a non-zero ssh status it prints "STOP: do not retry; tell Jon" and exits with
#               that status, leaving $D/pull.out.tmp and $D/pull.err in place and accepting nothing.
#   --accept    accept an existing $D/pull.out.tmp (e.g. after fixing the local Python env). No ssh.
#
# Accepting: require the "@@@ status s1 s2" trailer as the last line (a missing one means a truncated stream) and
# exactly one "@@@ exclusion_changes" marker; split into $D/author_excluded.csv.tmp and $D/exclusion_changes.csv;
# require s1 = 0 and no "query failed" / "ERROR" line in the flags part of pull.err (other stderr lines, such as a
# login banner, are printed for a human to read); report s2 (a failed exclusion-log query does not block the flags,
# but the doc must then say the log was not read); run analysis/check_author_excluded.py. Only if it exits 0 is the
# file moved to $D/author_excluded.csv. An existing $D/author_excluded.csv is never overwritten (checked before
# connecting too). Prints the sha256 of every file written and the time in UTC and Pacific.
#
# Usage (Phase B of the #85 plan; Jon must have cleared makelab1 first):
#
#   PYTHON="uv run --no-project --python 3.10 --with pandas<3 python" \
#     bash analysis/round2_flags_pull.sh --connect --check-out docs/ai-disagree-relabel-2026-10/round2_flags_check.json
#
# Options: --dir D (default ~/Data/sidewalk-validator-ai-relabel-2026-10/round2), --pool, --key, --unmasked (default:
# the committed round-2 files), --check-out FILE (the checker's counts-only JSON; default: none). Env: SCHEMAS
# (default "sidewalk_seattle sidewalk_chicago"), SINCE (default 2026-10-18), PYTHON (an interpreter with pandas;
# default python3; word-split, so a "uv run ... python" command works).
#
# Nothing here names a user id; the pulled files carry label ids and stay in $D, outside the repo.
set -u
HOST=makelab1
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
R="$ROOT/data/relabel-2026-10/round2"
D="$HOME/Data/sidewalk-validator-ai-relabel-2026-10/round2"
POOL="$R/pool.csv.gz"
KEY="$R/key/key.csv"
UNMASKED="$R/pre-consensus-2026-11-02/unmasked.csv"
CHECK_OUT=""
MODE=build
SCHEMAS="${SCHEMAS:-sidewalk_seattle sidewalk_chicago}"
SINCE="${SINCE:-2026-10-18}"
PYTHON="${PYTHON:-python3}"

die() { echo "round2_flags_pull: $*" >&2; exit 1; }
while [ $# -gt 0 ]; do
  case "$1" in
    --connect) MODE=connect ;;
    --accept) MODE=accept ;;
    --dir) D="$2"; shift ;;
    --pool) POOL="$2"; shift ;;
    --key) KEY="$2"; shift ;;
    --unmasked) UNMASKED="$2"; shift ;;
    --check-out) CHECK_OUT="$2"; shift ;;
    -h|--help) sed -n '2,32p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
  shift
done
# The schema names and SINCE go into the remote command line: validate them here, as the host scripts do.
for S in $SCHEMAS; do
  case "$S" in sidewalk_[a-z0-9_]*) ;; *) die "$S: not a city schema name" ;; esac
  case "$S" in *[!a-z0-9_]*) die "$S: not a city schema name" ;; esac
done
case "$SINCE" in [0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]) ;; *) die "SINCE=$SINCE: not YYYY-MM-DD" ;; esac

sha() { if command -v sha256sum >/dev/null; then sha256sum "$@"; else shasum -a 256 "$@"; fi; }
stamp() { echo "UTC $(date -u '+%Y-%m-%d %H:%M:%S')  Pacific $(TZ=America/Los_Angeles date '+%Y-%m-%d %H:%M:%S %Z')"; }

mkdir -p "$D" || die "cannot create $D"
BUNDLE="$D/remote_bundle.sh"
OUT_TMP="$D/pull.out.tmp"
ERR="$D/pull.err"
FLAGS_TMP="$D/author_excluded.csv.tmp"
FLAGS="$D/author_excluded.csv"
CHANGES="$D/exclusion_changes.csv"

build_bundle() {
  local f1="$HERE/pull_author_excluded.sh" f2="$HERE/pull_exclusion_changes.sh"
  [ -r "$f1" ] && [ -r "$f2" ] || die "host scripts not found next to $0"
  {
    echo "# Built by analysis/round2_flags_pull.sh ($(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)); runs on the DB host, read-only."
    echo "echo '@@@ stderr flags' >&2"
    echo "("
    cat "$f1"
    echo "); s1=\$?"
    echo "echo '@@@ exclusion_changes'"
    echo "echo '@@@ stderr exclusion_changes' >&2"
    echo "("
    cat "$f2"
    echo "); s2=\$?"
    echo "echo \"@@@ status \$s1 \$s2\""
  } > "$BUNDLE.new" || die "cannot write $BUNDLE"
  mv "$BUNDLE.new" "$BUNDLE"
  bash -n "$BUNDLE" || die "the bundle does not parse"
  sha "$BUNDLE"
}

accept() {
  [ -s "$OUT_TMP" ] || die "$OUT_TMP is missing or empty"
  [ -e "$FLAGS" ] && die "$FLAGS exists; refusing to overwrite an accepted pull"
  local trailer s1 s2 n_mark
  trailer="$(tail -n 1 "$OUT_TMP")"
  [[ "$trailer" =~ ^@@@\ status\ [0-9]+\ [0-9]+$ ]] \
    || die "no '@@@ status' trailer at the end of $OUT_TMP: truncated stream; accepting nothing"
  read -r _ _ s1 s2 <<< "$trailer"
  n_mark="$(grep -c '^@@@ exclusion_changes$' "$OUT_TMP")"
  [ "$n_mark" = 1 ] || die "expected one '@@@ exclusion_changes' marker in $OUT_TMP, found $n_mark"
  awk '/^@@@ exclusion_changes$/ {exit} {print}' "$OUT_TMP" > "$FLAGS_TMP"
  awk '/^@@@ exclusion_changes$/ {on=1; next} /^@@@ status / {exit} on {print}' "$OUT_TMP" > "$CHANGES"
  [ "$s1" = 0 ] || die "pull_author_excluded.sh exited $s1 on the host (a schema's query failed); accepting nothing"
  if [ -e "$ERR" ]; then
    # Only the flags part of stderr can block (everything before the exclusion-log marker).
    if awk '/^@@@ stderr exclusion_changes$/ {exit} {print}' "$ERR" | grep -E 'query failed|ERROR' >&2; then
      die "the flags part of $ERR names a failure (above); accepting nothing"
    fi
    if grep -v '^@@@ stderr ' "$ERR" | grep -q .; then
      echo "--- stderr from the host (read it; not treated as failure unless above):" >&2
      grep -v '^@@@ stderr ' "$ERR" >&2
      echo "---" >&2
    fi
  fi
  if [ "$s2" != 0 ]; then
    echo "WARNING: pull_exclusion_changes.sh exited $s2: the exclusion log was not (fully) read; the flags are" \
         "still checked. The doc must say the log was not read." >&2
  fi
  # shellcheck disable=SC2206  # SCHEMAS is split on purpose; validated above, so no glob characters
  local check_args=(--flags "$FLAGS_TMP" --pool "$POOL" --key "$KEY" --unmasked "$UNMASKED" --schemas $SCHEMAS)
  [ -n "$CHECK_OUT" ] && check_args+=(--out "$CHECK_OUT")
  # shellcheck disable=SC2086  # PYTHON may be a multi-word command (uv run ... python)
  if ! $PYTHON "$HERE/check_author_excluded.py" "${check_args[@]}"; then
    die "check_author_excluded.py rejected the flags; $FLAGS_TMP kept for inspection, nothing accepted"
  fi
  [ -e "$FLAGS" ] && die "$FLAGS appeared meanwhile; refusing to overwrite"
  mv "$FLAGS_TMP" "$FLAGS"
  echo "accepted: $FLAGS"
  sha "$BUNDLE" "$OUT_TMP" "$FLAGS" "$CHANGES" 2>/dev/null
  [ -n "$CHECK_OUT" ] && sha "$CHECK_OUT"
  stamp
  return 0
}

case "$MODE" in
  build)
    build_bundle
    ;;
  accept)
    accept
    ;;
  connect)
    # Everything that can fail locally fails BEFORE the one connection.
    [ -e "$FLAGS" ] && die "$FLAGS exists; refusing to overwrite an accepted pull (nothing sent)"
    [ -e "$OUT_TMP" ] && die "$OUT_TMP exists from an earlier pull; run --accept on it or move it away (nothing sent)"
    for f in "$POOL" "$KEY" "$UNMASKED"; do [ -r "$f" ] || die "$f not readable (nothing sent)"; done
    # shellcheck disable=SC2086
    $PYTHON -c 'import pandas' 2>/dev/null || die "PYTHON='$PYTHON' cannot import pandas (nothing sent)"
    build_bundle
    echo "control masters (informational):"
    ls ~/.ssh/cm-* 2>/dev/null || echo "  none"
    ssh -O check "$HOST" || true
    echo "pull started: $(stamp)"
    ssh -o BatchMode=yes "$HOST" "SCHEMAS='$SCHEMAS' SINCE='$SINCE' bash -s" < "$BUNDLE" > "$OUT_TMP" 2> "$ERR"
    st=$?
    echo "pull ended:   $(stamp)  ssh status $st"
    if [ "$st" != 0 ]; then
      echo "STOP: do not retry; tell Jon. ssh exited $st. $OUT_TMP and $ERR kept; nothing accepted." >&2
      cat "$ERR" >&2
      exit "$st"
    fi
    ( accept ); a=$?   # a subshell, so a rejection still reaches the master's close below
    ssh -O exit "$HOST" 2>/dev/null || true
    exit "$a"
    ;;
esac
