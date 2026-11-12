"""Tests for analysis/round2_flags_pull.sh and analysis/pull_exclusion_changes.sh (#85), with no host.

A stub ``psql`` first on PATH answers the three kinds of query with canned COPY output; a stub ``ssh`` records every
call and runs the remote command locally (so the bundle really runs against the stub psql). Nothing here connects
anywhere. The real SQL was run once against a synthetic schema in a local postgres:16 container (PR notes), not here.
"""
import os
import re
import stat
import subprocess
import sys

import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DRIVER = os.path.join(ROOT, "analysis", "round2_flags_pull.sh")
EXCL = os.path.join(ROOT, "analysis", "pull_exclusion_changes.sh")
SCHEMAS = "sidewalk_a sidewalk_b"
N = 200
UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.I)

STUB_PSQL = r"""#!/usr/bin/env bash
echo "PGOPTIONS=$PGOPTIONS ARGS=${*:1:$#-1}" >> "$STUB_LOG"
q="${@: -1}"
S=$(printf '%s' "$q" | grep -o 'FROM sidewalk_[a-z0-9_]*\.' | head -1 | sed 's/^FROM //; s/\.$//')
case "$q" in
  *user_stat*) kind=flags ;;
  *webpage_activity*) kind=changes ;;
  *audit_task*) kind=mismatch ;;
  *) echo "stub psql: unknown query" >&2; exit 3 ;;
esac
if [ "${STUB_FAIL:-}" = "$kind:$S" ]; then echo "ERROR:  relation \"$S.x\" does not exist" >&2; exit 1; fi
case $kind in
  flags) i=1; while [ $i -le ${STUB_N:-200} ]; do v=f; case " ${STUB_T:-} " in *" $S:$i "*) v=t ;; esac
         echo "$S,$i,$v"; i=$((i+1)); done ;;
  changes) echo "change,$S,2026-10-30,true,1,42" ;;
  mismatch) echo "author_mismatch,$S,0" ;;
esac
"""

STUB_SSH = r"""#!/usr/bin/env bash
echo "$*" >> "$SSH_LOG"
for a in "$@"; do [ "$a" = "-O" ] && exit 0; done
if [ -n "${STUB_SSH_STATUS:-}" ]; then cat > /dev/null; echo "Connection closed by 192.0.2.1 port 22" >&2
  exit "$STUB_SSH_STATUS"; fi
echo "Welcome to a test host" >&2
if [ -n "${STUB_SSH_TRUNCATE:-}" ]; then bash -c "${@: -1}" | sed '$d'; exit 0; fi
exec bash -c "${@: -1}"
"""


def _exe(path, text):
    path.write_text(text)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


@pytest.fixture
def env(tmp_path):
    stubs = tmp_path / "bin"
    stubs.mkdir()
    _exe(stubs / "psql", STUB_PSQL)
    _exe(stubs / "ssh", STUB_SSH)
    pd.DataFrame([{"schema": s, "label_id": i} for s in SCHEMAS.split() for i in range(1, N + 1)]).to_csv(
        tmp_path / "pool.csv.gz", index=False)
    pd.DataFrame([{"schema": s, "label_id": i} for s in SCHEMAS.split() for i in range(1, 9)]).to_csv(
        tmp_path / "key.csv", index=False)
    pd.DataFrame({"label_uid": ["sidewalk_a:5", "sidewalk_a:6"]}).to_csv(tmp_path / "unmasked.csv", index=False)
    e = dict(os.environ, PATH=f"{stubs}{os.pathsep}{os.environ['PATH']}", STUB_LOG=str(tmp_path / "psql.log"),
             SSH_LOG=str(tmp_path / "ssh.log"), SCHEMAS=SCHEMAS, PYTHON=sys.executable,
             STUB_T="sidewalk_a:5 sidewalk_a:6 sidewalk_b:150")
    for k in ("STUB_FAIL", "STUB_SSH_STATUS", "STUB_SSH_TRUNCATE"):
        e.pop(k, None)
    return tmp_path, e


def _driver(tmp_path, e, *args, **extra):
    d = tmp_path / "D"
    argv = ["bash", DRIVER, "--dir", str(d), "--pool", str(tmp_path / "pool.csv.gz"), "--key",
            str(tmp_path / "key.csv"), "--unmasked", str(tmp_path / "unmasked.csv"), *args]
    return subprocess.run(argv, env={**e, **extra}, capture_output=True, text=True), d


def _ssh_calls(tmp_path):
    p = tmp_path / "ssh.log"
    lines = p.read_text().splitlines() if p.exists() else []
    return [ln for ln in lines if "-O" not in ln.split()], [ln for ln in lines if "-O" in ln.split()]


def _run_bundle(tmp_path, e, **extra):
    r, d = _driver(tmp_path, e)
    assert r.returncode == 0, r.stderr
    return subprocess.run(["bash", str(d / "remote_bundle.sh")], env={**e, **extra}, capture_output=True, text=True)


def test_build_mode_writes_the_bundle_and_never_calls_ssh(env):
    tmp_path, e = env
    r, d = _driver(tmp_path, e)
    assert r.returncode == 0, r.stderr
    assert (d / "remote_bundle.sh").exists()
    assert re.search(r"^[0-9a-f]{64}\s+\S*remote_bundle\.sh$", r.stdout, re.M)
    assert _ssh_calls(tmp_path) == ([], [])


def test_the_bundle_runs_both_scripts_read_only_with_markers_and_trailer(env):
    tmp_path, e = env
    r = _run_bundle(tmp_path, e)
    lines = r.stdout.splitlines()
    assert lines[0] == "schema,label_id,author_excluded"
    assert lines[-1] == "@@@ status 0 0"
    i = lines.index("@@@ exclusion_changes")
    assert len(lines[1:i]) == 2 * N
    assert lines[i + 1:-1] == ["change,sidewalk_a,2026-10-30,true,1,42", "author_mismatch,sidewalk_a,0",
                               "change,sidewalk_b,2026-10-30,true,1,42", "author_mismatch,sidewalk_b,0"]
    assert r.stderr.splitlines() == ["@@@ stderr flags", "@@@ stderr exclusion_changes"]
    log = (tmp_path / "psql.log").read_text().splitlines()
    assert len(log) == 6  # 2 schemas x (flags, changes, mismatch)
    for ln in log:
        assert "default_transaction_read_only=on" in ln and "-d sidewalk_prod -p 5434" in ln


def test_a_failed_flags_query_shows_in_the_trailer(env):
    tmp_path, e = env
    r = _run_bundle(tmp_path, e, STUB_FAIL="flags:sidewalk_b")
    assert r.stdout.splitlines()[-1] == "@@@ status 1 0"
    assert not any(ln.startswith("sidewalk_b,") for ln in r.stdout.splitlines())
    assert "sidewalk_b: query failed" in r.stderr


def test_a_failed_exclusion_query_shows_in_the_trailer(env):
    tmp_path, e = env
    r = _run_bundle(tmp_path, e, STUB_FAIL="changes:sidewalk_a")
    assert r.stdout.splitlines()[-1] == "@@@ status 0 1"


def test_connect_pulls_once_splits_checks_and_accepts(env):
    tmp_path, e = env
    out = tmp_path / "check.json"
    r, d = _driver(tmp_path, e, "--connect", "--check-out", str(out))
    assert r.returncode == 0, r.stdout + r.stderr
    data, control = _ssh_calls(tmp_path)
    assert len(data) == 1 and "BatchMode=yes" in data[0] and "makelab1" in data[0]
    assert "SCHEMAS='sidewalk_a sidewalk_b'" in data[0]
    assert any("-O check" in c for c in control) and any("-O exit" in c for c in control)
    flags = (d / "author_excluded.csv").read_text().splitlines()
    assert flags[0] == "schema,label_id,author_excluded" and len(flags) == 1 + 2 * N
    assert not (d / "author_excluded.csv.tmp").exists()
    changes = (d / "exclusion_changes.csv").read_text().splitlines()
    assert sorted(len(ln.split(",")) for ln in changes) == [3, 3, 6, 6]
    assert not UUID.search("\n".join(changes))
    assert "Welcome to a test host" in r.stderr  # banner shown, not treated as failure
    assert out.exists() and "accepted:" in r.stdout


def test_connect_stops_without_retry_when_ssh_fails(env):
    tmp_path, e = env
    r, d = _driver(tmp_path, e, "--connect", STUB_SSH_STATUS="255")
    assert r.returncode == 255
    assert "STOP: do not retry; tell Jon" in r.stderr
    data, _ = _ssh_calls(tmp_path)
    assert len(data) == 1
    assert not (d / "author_excluded.csv").exists() and (d / "pull.out.tmp").exists()


def test_connect_refuses_a_truncated_stream(env):
    tmp_path, e = env
    r, d = _driver(tmp_path, e, "--connect", STUB_SSH_TRUNCATE="1")
    assert r.returncode != 0
    assert "truncated stream" in r.stderr
    assert not (d / "author_excluded.csv").exists()
    data, control = _ssh_calls(tmp_path)
    assert len(data) == 1 and any("-O exit" in c for c in control)  # master closed even on rejection


def test_connect_refuses_a_failed_flags_query(env):
    tmp_path, e = env
    r, d = _driver(tmp_path, e, "--connect", STUB_FAIL="flags:sidewalk_b")
    assert r.returncode != 0 and "exited 1 on the host" in r.stderr
    assert not (d / "author_excluded.csv").exists()


def test_connect_accepts_the_flags_when_only_the_exclusion_log_fails(env):
    tmp_path, e = env
    r, d = _driver(tmp_path, e, "--connect", STUB_FAIL="changes:sidewalk_b")
    assert r.returncode == 0, r.stderr
    assert "WARNING: pull_exclusion_changes.sh exited 1" in r.stderr
    assert (d / "author_excluded.csv").exists()


def test_connect_refuses_flags_the_checker_rejects(env):
    tmp_path, e = env
    r, d = _driver(tmp_path, e, "--connect", STUB_T="sidewalk_a:5")  # an unmasked item flagged f
    assert r.returncode != 0 and "check 7" in r.stderr
    assert not (d / "author_excluded.csv").exists() and (d / "author_excluded.csv.tmp").exists()


def test_connect_never_overwrites_an_accepted_pull_and_sends_nothing(env):
    tmp_path, e = env
    (tmp_path / "D").mkdir()
    (tmp_path / "D" / "author_excluded.csv").write_text("x\n")
    r, _ = _driver(tmp_path, e, "--connect")
    assert r.returncode != 0 and "refusing to overwrite" in r.stderr
    assert _ssh_calls(tmp_path) == ([], [])


def test_bad_schema_name_stops_before_any_ssh(env):
    tmp_path, e = env
    r, _ = _driver(tmp_path, e, "--connect", SCHEMAS="sidewalk_a; touch x")
    assert r.returncode != 0 and "not a city schema name" in r.stderr
    assert _ssh_calls(tmp_path) == ([], [])


def test_accept_mode_on_an_existing_stream_never_calls_ssh(env):
    tmp_path, e = env
    r = _run_bundle(tmp_path, e)
    d = tmp_path / "D"
    (d / "pull.out.tmp").write_text(r.stdout)
    (d / "pull.err").write_text(r.stderr)
    r2, _ = _driver(tmp_path, e, "--accept")
    assert r2.returncode == 0, r2.stderr
    assert (d / "author_excluded.csv").exists()
    assert _ssh_calls(tmp_path) == ([], [])


def _select_list(sql_after_select):
    """Top-level comma-separated expressions of a SELECT list (parentheses respected)."""
    items, depth, cur = [], 0, ""
    for ch in sql_after_select:
        depth += ch == "("
        depth -= ch == ")"
        if ch == "," and depth == 0:
            items.append(" ".join(cur.split()))
            cur = ""
        else:
            cur += ch
    return items + [" ".join(cur.split())]


def test_exclusion_queries_select_only_aggregates():
    """The activity string embeds a user_id: the outer SELECT lists may hold only literals, the day, the new value
    and counts."""
    text = open(EXCL).read()
    m = re.search(r"SELECT ('change'.*?)\n\s*FROM ev e GROUP BY", text, re.S)
    items = _select_list(m.group(1))
    assert items[:4] == ["'change'", "'$S'", "e.day", "e.new_value"]
    assert items[4] == "count(DISTINCT e.uid)"
    assert items[5].startswith("(SELECT count(*) FROM $S.label l") and len(items) == 6
    m = re.search(r"SELECT ('author_mismatch'.*?)\n\s*FROM", text, re.S)
    assert _select_list(m.group(1)) == ["'author_mismatch'", "'$S'", "count(*)"]
    assert "default_transaction_read_only=on" in text and "-p 5434" in text


@pytest.mark.parametrize("since", ["2026-10-18'; select 1; --", "yesterday", "2026-1-18"])
def test_exclusion_script_rejects_a_bad_since_before_any_query(env, since):
    tmp_path, e = env
    r = subprocess.run(["bash", EXCL], env={**e, "SINCE": since}, capture_output=True, text=True)
    assert r.returncode == 2 and "not YYYY-MM-DD" in r.stderr
    assert not (tmp_path / "psql.log").exists()
