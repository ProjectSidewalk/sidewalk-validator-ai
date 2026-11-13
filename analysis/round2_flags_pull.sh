#!/usr/bin/env bash
# Round-2 recount (issue #85): build, run (once) and accept the author-exclusion pull. Runs LOCALLY; the only
# remote entry point of the recount.
#
# Modes:
#   (default)   build only: write $D/remote_bundle.sh (pull_author_excluded.sh + pull_exclusion_changes.sh, each in
#               a subshell, with section markers and a status trailer) and print its sha256. No ssh.
#   --connect   build, then ONE ssh call to makelab1 that runs the bundle, then accept (below). No loop, no retry,
#               no fallback host, one connection attempt with a 30 s connect timeout: on a non-zero ssh status it
#               prints "STOP: do not retry; tell Jon" and exits with that status, leaving $D/pull.out.tmp and
#               $D/pull.err in place and accepting nothing. Everything that can fail locally is checked before the
#               connection. A control master that was already live is used and left open; one that this call
#               opened is closed afterwards.
#   --accept    accept an existing $D/pull.out.tmp (e.g. after fixing the local Python env). No ssh.
#
# Accepting: require the "@@@ status s1 s2" trailer as the last line (a missing one means a truncated stream) and
# exactly one "@@@ exclusion_changes" marker; split into $D/author_excluded.csv.tmp and
# $D/exclusion_changes.csv.tmp; require s1 = 0 and no "query failed" / "ERROR:" / "FATAL:" line in the flags part
# of pull.err (other stderr lines, such as a login banner, are printed for a human to read); run
# analysis/check_author_excluded.py. Only if it exits 0 are the files moved to their final names, so a rejected
# pull leaves only .tmp files. The exclusion log does not block the flags: if its script failed (s2 != 0) or a
# line of it is not a count row, the flags are still accepted and the log is kept as
# $D/exclusion_changes.csv.incomplete (never under the accepted name; the doc must then say the log was not read).
# An existing accepted file is never overwritten (checked before connecting too).
#
# Provenance: --connect records the sha256 of the bundle it sends in $D/pull.bundle.sha256. Build mode refuses to
# replace the bundle while a pulled stream is in $D, and --accept refuses a bundle that differs from the record, so
# the sha256 printed at acceptance is of the script that ran. Prints the sha256 of every file written and the time
# in UTC and Pacific.
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
# In other locales a range such as [a-z] can match capitals and accented letters, which would loosen the name checks
# below. The bundle sets the same on the host.
export LC_ALL=C
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
    -h|--help) awk 'NR > 1 && !/^#/ {exit} NR > 1' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
  shift
done
# The schema names and SINCE go into the remote command line: validate them here, as the host scripts do.
N_SCHEMAS=0
for S in $SCHEMAS; do
  case "$S" in sidewalk_[a-z0-9_]*) ;; *) die "$S: not a city schema name" ;; esac
  case "$S" in *[!a-z0-9_]*) die "$S: not a city schema name" ;; esac
  N_SCHEMAS=$((N_SCHEMAS + 1))
done
[ "$N_SCHEMAS" -gt 0 ] || die "SCHEMAS names no schema"
DATE_RE='^[0-9]{4}-(0[1-9]|1[0-2])-(0[1-9]|[12][0-9]|3[01])$'
[[ "$SINCE" =~ $DATE_RE ]] || die "SINCE=$SINCE: not YYYY-MM-DD"

sha() { if command -v sha256sum >/dev/null; then sha256sum "$@"; else shasum -a 256 "$@"; fi; }
sha_of() { sha "$1" | awk '{print $1}'; }
stamp() { echo "UTC $(date -u '+%Y-%m-%d %H:%M:%S')  Pacific $(TZ=America/Los_Angeles date '+%Y-%m-%d %H:%M:%S %Z')"; }

mkdir -p "$D" || die "cannot create $D"
BUNDLE="$D/remote_bundle.sh"
OUT_TMP="$D/pull.out.tmp"
ERR="$D/pull.err"
FLAGS_TMP="$D/author_excluded.csv.tmp"
FLAGS="$D/author_excluded.csv"
CHANGES_TMP="$D/exclusion_changes.csv.tmp"
CHANGES="$D/exclusion_changes.csv"
CHANGES_BAD="$D/exclusion_changes.csv.incomplete"
BUNDLE_SHA="$D/pull.bundle.sha256"
# The only lines pull_exclusion_changes.sh may print: counts, a day, a schema, true/false. Anything else (an id, a
# stray message) keeps the log out of its accepted name.
CHANGE_ROW='^(change,sidewalk_[a-z0-9_]+,[0-9]{4}-[0-9]{2}-[0-9]{2},(true|false)?,[0-9]+,[0-9]+|author_mismatch,sidewalk_[a-z0-9_]+,[0-9]+)$'

# The checker writes --check-out only after every check passes: make sure it can, before anything else happens.
check_out_writable() {
  [ -z "$CHECK_OUT" ] && return 0
  local dir
  dir="$(dirname "$CHECK_OUT")"
  [ -d "$dir" ] && [ -w "$dir" ] || die "--check-out $CHECK_OUT: $dir is not a writable directory (nothing sent)"
}

build_bundle() {
  local f1="$HERE/pull_author_excluded.sh" f2="$HERE/pull_exclusion_changes.sh"
  [ -r "$f1" ] && [ -r "$f2" ] || die "host scripts not found next to $0"
  {
    echo "# Built by analysis/round2_flags_pull.sh ($(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)); runs on the DB host, read-only."
    echo "export LC_ALL=C  # byte-wise [a-z] in the scripts' name checks; untranslated psql messages"
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
  if [ -e "$OUT_TMP" ] && [ -e "$BUNDLE" ] && ! cmp -s "$BUNDLE.new" "$BUNDLE"; then
    rm -f "$BUNDLE.new"
    die "$OUT_TMP was pulled with the bundle now in $D, and a fresh build differs from it; refusing to replace" \
        "the script that ran (build into another --dir instead)"
  fi
  mv "$BUNDLE.new" "$BUNDLE"
  bash -n "$BUNDLE" || die "the bundle does not parse"
  sha "$BUNDLE"
}

accept() {
  [ -s "$OUT_TMP" ] || die "$OUT_TMP is missing or empty"
  [ -e "$FLAGS" ] && die "$FLAGS exists; refusing to overwrite an accepted pull"
  [ -e "$CHANGES" ] && die "$CHANGES exists; refusing to overwrite it (move it away)"
  check_out_writable
  if [ -e "$BUNDLE_SHA" ]; then
    [ -r "$BUNDLE" ] && [ "$(sha_of "$BUNDLE")" = "$(cat "$BUNDLE_SHA")" ] \
      || die "$BUNDLE is not the bundle that was sent (sha256 recorded in $BUNDLE_SHA); accepting nothing"
  else
    echo "note: no $BUNDLE_SHA, so nothing ties $OUT_TMP to the bundle in $D" >&2
  fi
  local trailer s1 s2 n_mark changes_ok=1 n_bad
  trailer="$(tail -n 1 "$OUT_TMP")"
  [[ "$trailer" =~ ^@@@\ status\ [0-9]+\ [0-9]+$ ]] \
    || die "no '@@@ status' trailer at the end of $OUT_TMP: truncated stream; accepting nothing"
  read -r _ _ s1 s2 <<< "$trailer"
  n_mark="$(grep -c '^@@@ exclusion_changes$' "$OUT_TMP")"
  [ "$n_mark" = 1 ] || die "expected one '@@@ exclusion_changes' marker in $OUT_TMP, found $n_mark"
  awk '/^@@@ exclusion_changes$/ {exit} {print}' "$OUT_TMP" > "$FLAGS_TMP"
  awk '/^@@@ exclusion_changes$/ {on=1; next} /^@@@ status / {exit} on {print}' "$OUT_TMP" > "$CHANGES_TMP"
  [ "$s1" = 0 ] || die "pull_author_excluded.sh exited $s1 on the host (a schema's query failed); accepting nothing"
  if [ -e "$ERR" ]; then
    # Only the flags part of stderr can block (everything before the exclusion-log marker).
    if awk '/^@@@ stderr exclusion_changes$/ {exit} {print}' "$ERR" \
        | grep -E 'query failed|(^|[^A-Za-z])(ERROR|FATAL|PANIC):' >&2; then
      die "the flags part of $ERR names a failure (above); accepting nothing"
    fi
    if grep -v '^@@@ stderr ' "$ERR" | grep -q .; then
      echo "--- stderr from the host (read it; not treated as failure unless above):" >&2
      grep -v '^@@@ stderr ' "$ERR" >&2
      echo "---" >&2
    fi
  fi
  if [ "$s2" != 0 ]; then
    changes_ok=0
    echo "WARNING: pull_exclusion_changes.sh exited $s2: the exclusion log was not (fully) read; the flags are" \
         "still checked. The doc must say the log was not read." >&2
  fi
  n_bad="$(grep -Evc "$CHANGE_ROW" "$CHANGES_TMP")"
  if [ "$n_bad" != 0 ]; then
    changes_ok=0
    echo "WARNING: $n_bad line(s) of the exclusion log are not count rows (not shown: they may not be counts)." \
         "Do not paste the file anywhere; the flags are still checked." >&2
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
  if [ "$changes_ok" = 1 ]; then
    mv "$CHANGES_TMP" "$CHANGES"
    echo "accepted: $CHANGES"
  else
    mv "$CHANGES_TMP" "$CHANGES_BAD"
    echo "NOT accepted (see the warnings): the exclusion log, kept as $CHANGES_BAD"
    CHANGES="$CHANGES_BAD"
  fi
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
    [ -e "$CHANGES" ] && die "$CHANGES exists; refusing to overwrite it (nothing sent)"
    [ -e "$OUT_TMP" ] && die "$OUT_TMP exists from an earlier pull; run --accept on it or move it away (nothing sent)"
    check_out_writable
    for f in "$POOL" "$KEY" "$UNMASKED"; do [ -r "$f" ] || die "$f not readable (nothing sent)"; done
    # shellcheck disable=SC2086
    $PYTHON -c 'import pandas' 2>/dev/null || die "PYTHON='$PYTHON' cannot import pandas (nothing sent)"
    build_bundle
    sha_of "$BUNDLE" > "$BUNDLE_SHA" || die "cannot write $BUNDLE_SHA (nothing sent)"
    # "-O" only talks to a local control socket; it never opens a connection.
    echo "control sockets (informational):"
    ls ~/.ssh/cm-* 2>/dev/null || echo "  none"
    if ssh -O check "$HOST" 2>/dev/null; then
      had_master=1; echo "a master to $HOST is live: using it and leaving it open"
    else
      had_master=0; echo "no live master to $HOST: this call opens the one connection"
    fi
    close_own_master() { [ "$had_master" = 0 ] && ssh -O exit "$HOST" 2>/dev/null; return 0; }
    echo "pull started: $(stamp)"
    # The one connection. ConnectionAttempts=1 and ConnectTimeout override any ssh config: one try, and a host
    # that never answers is given up on after 30 s.
    ssh -o BatchMode=yes -o ConnectionAttempts=1 -o ConnectTimeout=30 "$HOST" \
        "SCHEMAS='$SCHEMAS' SINCE='$SINCE' bash -s" < "$BUNDLE" > "$OUT_TMP" 2> "$ERR"
    st=$?
    echo "pull ended:   $(stamp)  ssh status $st"
    if [ "$st" != 0 ]; then
      echo "STOP: do not retry; tell Jon. ssh exited $st. $OUT_TMP and $ERR kept; nothing accepted." >&2
      cat "$ERR" >&2
      close_own_master
      exit "$st"
    fi
    ( accept ); a=$?   # a subshell, so a rejection still reaches the master's close below
    close_own_master
    exit "$a"
    ;;
esac
