"""Tests for analysis/round2_flags_pull.sh and analysis/pull_exclusion_changes.sh (#85), with no host.

A stub ``psql`` first on PATH answers the three kinds of query with canned COPY output; a stub ``ssh`` records every
call and runs the remote command locally (so the bundle really runs against the stub psql). Nothing here connects
anywhere. The real SQL was run once against a synthetic schema in a local postgres:16 container (PR notes), not here.
"""
import hashlib
import os
import re
import shutil
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
  changes) echo "${STUB_CHANGE_ROW:-change,$S,2026-10-30,true,1,42}" ;;
  mismatch) echo "author_mismatch,$S,0" ;;
esac
"""

STUB_SSH = r"""#!/usr/bin/env bash
echo "$*" >> "$SSH_LOG"
# "-O check" succeeds only when the test says a master is already live; "-O exit" always does.
if [ "$1" = "-O" ]; then [ "$2" = check ] && [ -z "${STUB_MASTER:-}" ] && exit 255; exit 0; fi
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
    for k in ("STUB_FAIL", "STUB_SSH_STATUS", "STUB_SSH_TRUNCATE", "STUB_MASTER", "STUB_CHANGE_ROW", "SINCE"):
        e.pop(k, None)
    return tmp_path, e


def _driver(tmp_path, e, *args, **extra):
    d = tmp_path / "D"
    argv = ["bash", DRIVER, "--dir", str(d), "--pool", str(tmp_path / "pool.csv.gz"), "--key",
            str(tmp_path / "key.csv"), "--unmasked", str(tmp_path / "unmasked.csv"), *args]
    return subprocess.run(argv, env={**e, **extra}, capture_output=True, text=True), d


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def test_the_bundle_checks_names_bytewise_whatever_the_host_locale(env):
    """pull_author_excluded.sh has the same [a-z] patterns and no locale line of its own: the bundle pins it."""
    tmp_path, e = env
    r = _run_bundle(tmp_path, e, SCHEMAS="sidewalk_a sidewalk_B", LC_ALL="en_US.UTF-8")
    assert r.stdout.splitlines()[-1] == "@@@ status 2 2"
    assert r.stderr.count("sidewalk_B: not a city schema name") == 2
    assert not (tmp_path / "psql.log").exists()


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
    assert "ConnectionAttempts=1" in data[0] and "ConnectTimeout=30" in data[0]
    assert "SCHEMAS='sidewalk_a sidewalk_b'" in data[0]
    assert control == ["-O check makelab1", "-O exit makelab1"]  # no master was live: the one it opened is closed
    flags = (d / "author_excluded.csv").read_text().splitlines()
    assert flags[0] == "schema,label_id,author_excluded" and len(flags) == 1 + 2 * N
    assert not list(d.glob("*.csv.tmp")) and not list(d.glob("*.incomplete"))
    assert (d / "pull.bundle.sha256").read_text().strip() == _sha(d / "remote_bundle.sh")
    assert _sha(d / "remote_bundle.sh") in r.stdout
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
    data, control = _ssh_calls(tmp_path)
    assert len(data) == 1 and control == ["-O check makelab1", "-O exit makelab1"]
    assert sorted(p.name for p in d.iterdir()) == ["pull.bundle.sha256", "pull.err", "pull.out.tmp",
                                                   "remote_bundle.sh"]  # nothing split, nothing accepted
    assert (d / "pull.out.tmp").read_text() == "" and "Connection closed" in (d / "pull.err").read_text()
    # the next attempt, and an --accept, both stop locally: still one data call in the log
    r2, _ = _driver(tmp_path, e, "--connect")
    assert r2.returncode != 0 and "exists from an earlier pull" in r2.stderr and "nothing sent" in r2.stderr
    r3, _ = _driver(tmp_path, e, "--accept")
    assert r3.returncode != 0 and "missing or empty" in r3.stderr
    assert len(_ssh_calls(tmp_path)[0]) == 1


def test_connect_uses_a_live_master_and_leaves_it_open(env):
    """Another session's master is reused (no new handshake) and must not be closed under it."""
    tmp_path, e = env
    r, d = _driver(tmp_path, e, "--connect", STUB_MASTER="1")
    assert r.returncode == 0, r.stderr
    data, control = _ssh_calls(tmp_path)
    assert len(data) == 1 and control == ["-O check makelab1"]
    assert "using it and leaving it open" in r.stdout


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
    assert not (d / "author_excluded.csv").exists() and not (d / "exclusion_changes.csv").exists()


def test_connect_accepts_the_flags_when_only_the_exclusion_log_fails(env):
    tmp_path, e = env
    r, d = _driver(tmp_path, e, "--connect", STUB_FAIL="changes:sidewalk_b")
    assert r.returncode == 0, r.stderr
    assert "WARNING: pull_exclusion_changes.sh exited 1" in r.stderr
    assert (d / "author_excluded.csv").exists()
    # the partial log never gets the accepted name
    assert not (d / "exclusion_changes.csv").exists() and (d / "exclusion_changes.csv.incomplete").exists()
    assert "NOT accepted" in r.stdout


def test_an_exclusion_log_line_that_is_not_a_count_row_keeps_the_log_unaccepted(env):
    """Backstop for "counts only": if the log ever carried anything else (here a user id), the file must not land
    under its accepted name, and the driver must not echo the line."""
    tmp_path, e = env
    uid = "0f8fad5b-d9cb-469f-a165-70867728950e"
    r, d = _driver(tmp_path, e, "--connect", STUB_CHANGE_ROW=f"change,sidewalk_a,2026-10-30,true,{uid},42")
    assert r.returncode == 0, r.stderr  # the flags are still accepted
    assert (d / "author_excluded.csv").exists()
    assert not (d / "exclusion_changes.csv").exists() and (d / "exclusion_changes.csv.incomplete").exists()
    assert "2 line(s) of the exclusion log are not count rows" in r.stderr
    assert uid not in r.stdout + r.stderr


def test_a_change_row_with_an_unparsed_value_is_still_a_count_row(env):
    """An activity that matches the LIKE but not the pattern gives an empty new_value and zero counts."""
    tmp_path, e = env
    r, d = _driver(tmp_path, e, "--connect", STUB_CHANGE_ROW="change,sidewalk_a,2026-10-29,,0,0")
    assert r.returncode == 0, r.stderr
    assert (d / "exclusion_changes.csv").exists()


def test_connect_refuses_flags_the_checker_rejects(env):
    tmp_path, e = env
    r, d = _driver(tmp_path, e, "--connect", STUB_T="sidewalk_a:5")  # an unmasked item flagged f
    assert r.returncode != 0 and "check 7" in r.stderr
    assert not (d / "author_excluded.csv").exists() and (d / "author_excluded.csv.tmp").exists()
    # a rejected pull leaves nothing under an accepted name
    assert not (d / "exclusion_changes.csv").exists() and (d / "exclusion_changes.csv.tmp").exists()


def test_connect_never_overwrites_an_accepted_pull_and_sends_nothing(env):
    tmp_path, e = env
    (tmp_path / "D").mkdir()
    (tmp_path / "D" / "author_excluded.csv").write_text("x\n")
    r, _ = _driver(tmp_path, e, "--connect")
    assert r.returncode != 0 and "refusing to overwrite" in r.stderr
    assert _ssh_calls(tmp_path) == ([], [])


# "public" fails only the prefix pattern; the quoted one passes the prefix (its * matches anything) and fails only
# the character pattern, so each of the two patterns is needed. The capital passes [a-z] in a UTF-8 locale unless
# the scripts pin LC_ALL=C.
BAD_SCHEMAS = ["sidewalk_a; touch x", "public", "sidewalk_a';id;'", "sidewalk_a sidewalk_B"]


@pytest.mark.parametrize("schemas", BAD_SCHEMAS)
def test_bad_schema_name_stops_before_any_ssh(env, schemas):
    tmp_path, e = env
    r, _ = _driver(tmp_path, e, "--connect", SCHEMAS=schemas)
    assert r.returncode != 0 and "not a city schema name" in r.stderr
    assert _ssh_calls(tmp_path) == ([], [])


@pytest.mark.parametrize("schemas", BAD_SCHEMAS)
def test_exclusion_script_rejects_a_bad_schema_before_any_query(env, schemas):
    tmp_path, e = env
    r = subprocess.run(["bash", EXCL], env={**e, "SCHEMAS": schemas}, capture_output=True, text=True)
    assert r.returncode == 2 and "not a city schema name" in r.stderr
    assert not (tmp_path / "psql.log").exists()


BAD_SINCE = ["2026-10-18'; select 1; --", "yesterday", "2026-1-18", "2026-13-45", "2026-10-18\n", "2026-10-32"]


@pytest.mark.parametrize("since", BAD_SINCE)
def test_bad_since_stops_before_any_ssh(env, since):
    tmp_path, e = env
    r, _ = _driver(tmp_path, e, "--connect", SINCE=since)
    assert r.returncode != 0 and "not YYYY-MM-DD" in r.stderr
    assert _ssh_calls(tmp_path) == ([], [])


def test_local_mistakes_stop_before_the_one_connection(env):
    """Each of these used to be found, or could be found, only after the connection was spent."""
    tmp_path, e = env
    cases = [
        (("--connect",), {"SCHEMAS": " "}, "SCHEMAS names no schema"),
        (("--connect", "--check-out", str(tmp_path / "no-such-dir" / "c.json")), {}, "not a writable directory"),
        (("--connect",), {"PYTHON": "/nonexistent/python"}, "cannot import pandas"),
        (("--connect", "--key", str(tmp_path / "no-key.csv")), {}, "not readable"),
    ]
    for args, extra, message in cases:
        r, d = _driver(tmp_path, e, *args, **extra)
        assert r.returncode != 0 and message in r.stderr, (message, r.stderr)
        assert _ssh_calls(tmp_path) == ([], [])
        assert not (d / "pull.out.tmp").exists()


def test_connect_refuses_when_an_earlier_stream_is_waiting(env):
    tmp_path, e = env
    (tmp_path / "D").mkdir()
    (tmp_path / "D" / "pull.out.tmp").write_text("")
    r, _ = _driver(tmp_path, e, "--connect")
    assert r.returncode != 0 and "exists from an earlier pull" in r.stderr
    assert _ssh_calls(tmp_path) == ([], [])


def _stream(tmp_path, e):
    """A good stream and its stderr placed in D by hand (as after a pull whose local check could not run)."""
    r = _run_bundle(tmp_path, e)
    d = tmp_path / "D"
    (d / "pull.out.tmp").write_text(r.stdout)
    (d / "pull.err").write_text(r.stderr)
    return d, r


def test_accept_never_overwrites_an_accepted_pull(env):
    tmp_path, e = env
    d, _ = _stream(tmp_path, e)
    (d / "author_excluded.csv").write_text("x\n")
    r, _ = _driver(tmp_path, e, "--accept")
    assert r.returncode != 0 and "refusing to overwrite" in r.stderr
    assert (d / "author_excluded.csv").read_text() == "x\n"


def test_accept_refuses_two_markers(env):
    tmp_path, e = env
    d, r0 = _stream(tmp_path, e)
    (d / "pull.out.tmp").write_text(r0.stdout.replace("@@@ exclusion_changes\n", "@@@ exclusion_changes\n" * 2))
    r, _ = _driver(tmp_path, e, "--accept")
    assert r.returncode != 0 and "expected one '@@@ exclusion_changes' marker" in r.stderr
    assert not (d / "author_excluded.csv").exists()


@pytest.mark.parametrize("line,part,accepted", [
    ("sidewalk_b: query failed", "flags", False),               # the host script's own report
    ('ERROR:  relation "x" does not exist', "flags", False),    # psql
    ("psql: error: connection failed: FATAL:  no such role", "flags", False),
    ("Report ERRORS to support", "flags", True),                # a banner is read by a human, not a failure
    ('ERROR:  relation "x" does not exist', "exclusion_changes", True),  # the log never blocks the flags
])
def test_accept_reads_only_the_flags_part_of_stderr_for_failures(env, line, part, accepted):
    """The status trailer says s1 = 0 here, so the stderr scan is the only thing that can refuse."""
    tmp_path, e = env
    d, _ = _stream(tmp_path, e)
    err = {"flags": "", "exclusion_changes": ""}
    err[part] = line + "\n"
    (d / "pull.err").write_text(f"@@@ stderr flags\n{err['flags']}@@@ stderr exclusion_changes\n"
                                f"{err['exclusion_changes']}")
    r, _ = _driver(tmp_path, e, "--accept")
    assert (r.returncode == 0) is accepted, r.stderr
    assert (d / "author_excluded.csv").exists() is accepted
    if not accepted:
        assert "names a failure" in r.stderr


def _changed_copy(tmp_path):
    """The driver and host scripts copied elsewhere, with one host script changed after the pull."""
    an = tmp_path / "repo" / "analysis"
    an.mkdir(parents=True)
    for name in ("round2_flags_pull.sh", "pull_author_excluded.sh", "pull_exclusion_changes.sh",
                 "check_author_excluded.py"):
        shutil.copy(os.path.join(ROOT, "analysis", name), an / name)
    with open(an / "pull_exclusion_changes.sh", "a") as f:
        f.write("# changed after the pull\n")
    return str(an / "round2_flags_pull.sh")


def test_build_never_replaces_the_bundle_a_waiting_stream_came_from(env):
    tmp_path, e = env
    r, d = _driver(tmp_path, e, "--connect", STUB_T="sidewalk_a:5")  # pulled, rejected by the checker
    assert r.returncode != 0
    sent = _sha(d / "remote_bundle.sh")
    argv = ["bash", _changed_copy(tmp_path), "--dir", str(d)]
    r2 = subprocess.run(argv, env=e, capture_output=True, text=True)
    assert r2.returncode != 0 and "refusing to replace" in r2.stderr
    assert _sha(d / "remote_bundle.sh") == sent
    r3, _ = _driver(tmp_path, e)  # an identical build is fine
    assert r3.returncode == 0, r3.stderr


def test_accept_refuses_a_bundle_that_is_not_the_one_sent(env):
    tmp_path, e = env
    r, d = _driver(tmp_path, e, "--connect", PYTHON="false")  # stops before connecting: pandas preflight
    assert r.returncode != 0
    r, d = _driver(tmp_path, e, "--connect", STUB_T="sidewalk_a:5")
    assert r.returncode != 0 and "check 7" in r.stderr
    with open(d / "remote_bundle.sh", "a") as f:
        f.write("# edited\n")
    r2, _ = _driver(tmp_path, e, "--accept")
    assert r2.returncode != 0 and "is not the bundle that was sent" in r2.stderr
    assert not (d / "author_excluded.csv").exists()


def test_accept_mode_on_an_existing_stream_never_calls_ssh(env):
    tmp_path, e = env
    d, _ = _stream(tmp_path, e)
    r2, _ = _driver(tmp_path, e, "--accept")
    assert r2.returncode == 0, r2.stderr
    assert (d / "author_excluded.csv").exists() and (d / "exclusion_changes.csv").exists()
    assert "nothing ties" in r2.stderr  # placed by hand: no record of the bundle that produced it
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
    # SINCE is a Pacific day, like the days in the output; it must not depend on the session's TimeZone
    assert "a.timestamp >= (TIMESTAMP '$SINCE' AT TIME ZONE 'America/Los_Angeles')" in text


def test_no_account_name_in_the_scripts():
    """Usernames stay out of the repo: the scripts speak of the pulling role."""
    for name in ("pull_exclusion_changes.sh", "round2_flags_pull.sh", "check_author_excluded.py"):
        text = open(os.path.join(ROOT, "analysis", name)).read()
        assert not re.search(r"\b(jonf|jfroehli)\b", text), name


@pytest.mark.parametrize("since", BAD_SINCE)
def test_exclusion_script_rejects_a_bad_since_before_any_query(env, since):
    tmp_path, e = env
    r = subprocess.run(["bash", EXCL], env={**e, "SINCE": since}, capture_output=True, text=True)
    assert r.returncode == 2 and "not YYYY-MM-DD" in r.stderr
    assert not (tmp_path / "psql.log").exists()
