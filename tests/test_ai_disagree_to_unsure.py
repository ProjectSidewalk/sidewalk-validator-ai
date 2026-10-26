"""End-to-end test of analysis/ai_disagree_to_unsure.sh and analysis/ai_disagree_census.sh against a real Postgres.

The scripts run on the database host with psql, so the test needs one of:
  * a throwaway database reachable with a local psql, named in AI_DISAGREE_TEST_DB (the test drops and recreates
    schemas named sidewalk_* in it, so never point it at a real one); or
  * Docker, in which case a postgres:16 container is started and the scripts run inside it.
With neither, every test here is skipped.

The seed covers the cases the plan names: a lone AI Disagree, an AI vote that broke a human tie, humans ahead by one, a
label the humans also voted down, deleted labels (Explore and non-Explore), a tutorial label, an excluded validator,
an AI vote on a label's old type, labelers at the 50-validation and 0.6-accuracy thresholds, a manual low-quality
labeler, a duplicated user_stat row (#4776), a second city reusing the same label_ids, a city too old for the port,
and the schemas the loop must skip.
"""
import csv
import io
import os
import shutil
import subprocess
import time
import uuid
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "analysis" / "ai_disagree_to_unsure.sh"
CENSUS = REPO / "analysis" / "ai_disagree_census.sh"
AI = "51b0b927-3c8a-45b2-93de-bd878d1e5cf4"
DB = "sidewalk_ci"

TYPES_SQL = """
DROP TYPE IF EXISTS validation_option CASCADE;
DROP TYPE IF EXISTS label_type CASCADE;
DROP TYPE IF EXISTS ui_source CASCADE;
CREATE TYPE validation_option AS ENUM ('Agree', 'Disagree', 'Unsure');
CREATE TYPE label_type AS ENUM
  ('CurbRamp', 'NoCurbRamp', 'Obstacle', 'SurfaceProblem', 'Crosswalk', 'Signal', 'NoSidewalk', 'Occlusion', 'Other');
CREATE TYPE ui_source AS ENUM ('Explore', 'Validate', 'LabelMap', 'GalleryImage', 'SidewalkAI');
"""

# Column names and types as in SidewalkWebpage's evolutions (only the columns the scripts read, plus NOT NULLs).
CITY_TABLES = """
DROP SCHEMA IF EXISTS {s} CASCADE;
CREATE SCHEMA {s};
SET search_path = {s}, public;
CREATE TABLE play_evolutions (id INT PRIMARY KEY, state VARCHAR(255));
INSERT INTO play_evolutions SELECT g, 'applied' FROM generate_series(1, {evo}) AS g;
CREATE TABLE config (tutorial_street_edge_id INT NOT NULL);
INSERT INTO config VALUES (999);
CREATE TABLE audit_task (audit_task_id SERIAL PRIMARY KEY, user_id TEXT NOT NULL, street_edge_id INT NOT NULL);
CREATE TABLE label (
    label_id SERIAL PRIMARY KEY, audit_task_id INT NOT NULL REFERENCES audit_task, user_id TEXT NOT NULL,
    label_type label_type NOT NULL, deleted BOOLEAN NOT NULL DEFAULT FALSE, deleted_source ui_source,
    tutorial BOOLEAN NOT NULL DEFAULT FALSE, street_edge_id INT NOT NULL,
    agree_count INT NOT NULL DEFAULT 0, disagree_count INT NOT NULL DEFAULT 0, unsure_count INT NOT NULL DEFAULT 0,
    correct BOOLEAN);
CREATE TABLE label_validation (
    label_validation_id SERIAL PRIMARY KEY, label_id INT NOT NULL REFERENCES label, label_type label_type NOT NULL,
    validation_result validation_option NOT NULL, user_id TEXT NOT NULL,
    end_timestamp TIMESTAMPTZ NOT NULL DEFAULT '2026-10-01 12:00:00+00', UNIQUE (label_id, user_id, label_type));
CREATE TABLE user_stat (
    user_stat_id SERIAL PRIMARY KEY, user_id TEXT NOT NULL, meters_audited DOUBLE PRECISION NOT NULL DEFAULT 0,
    labels_per_meter DOUBLE PRECISION, high_quality BOOLEAN NOT NULL DEFAULT TRUE, high_quality_manual BOOLEAN,
    own_labels_validated INT NOT NULL DEFAULT 0, accuracy DOUBLE PRECISION, excluded BOOLEAN NOT NULL DEFAULT FALSE);
CREATE TABLE label_ai_assessment (
    label_ai_assessment_id SERIAL PRIMARY KEY, label_id INT NOT NULL REFERENCES label,
    label_validation_id INT REFERENCES label_validation, validation_result validation_option);
-- add_label(labeler, audit_task, type, stored counts and correct, flags) -> label_id
CREATE FUNCTION add_label(u TEXT, at INT, a INT, d INT, n INT, c BOOLEAN,
                          del BOOLEAN DEFAULT FALSE, src ui_source DEFAULT NULL, tut BOOLEAN DEFAULT FALSE)
RETURNS INT LANGUAGE sql AS $$
    INSERT INTO label (audit_task_id, user_id, label_type, deleted, deleted_source, tutorial, street_edge_id,
                       agree_count, disagree_count, unsure_count, correct)
    SELECT at, u, 'CurbRamp', del, src, tut, audit_task.street_edge_id, a, d, n, c
    FROM audit_task WHERE audit_task_id = at
    RETURNING label_id $$;
CREATE FUNCTION vote(l INT, u TEXT, r validation_option, t label_type DEFAULT 'CurbRamp') RETURNS VOID
LANGUAGE sql AS $$ INSERT INTO label_validation (label_id, label_type, validation_result, user_id) VALUES (l, t, r, u) $$;
"""

# City A: every case. Stored counts, `correct`, accuracy and high_quality are what the app holds for these votes.
CITY_A_SEED = """
INSERT INTO audit_task (user_id, street_edge_id) VALUES ('L1', 1), ('L2', 1), ('L3', 1), ('L1', 999), ('L4', 1);
INSERT INTO user_stat (user_id, meters_audited, labels_per_meter, high_quality, high_quality_manual,
                       own_labels_validated, accuracy, excluded) VALUES
    ('L1', 100, 0.5, TRUE, NULL, 6, 1.0 / 6, FALSE),
    ('L2', 100, 0.5, FALSE, NULL, 50, 0.58, FALSE),
    ('L2', 100, 0.5, FALSE, NULL, 50, 0.58, FALSE),   -- a duplicate row (#4776)
    ('L3', 100, 0.5, FALSE, NULL, 60, 0.6, FALSE),
    ('L4', 100, 0.5, FALSE, FALSE, 1, 0.0, FALSE),    -- manually marked low quality
    ('H1', 0, NULL, TRUE, NULL, 0, NULL, FALSE), ('H2', 0, NULL, TRUE, NULL, 0, NULL, FALSE),
    ('H3', 0, NULL, TRUE, NULL, 0, NULL, FALSE), ('X', 0, NULL, FALSE, NULL, 0, NULL, TRUE),
    ('{ai}', 0, NULL, TRUE, TRUE, 0, NULL, FALSE);
DO $$
DECLARE l INT;
BEGIN
    -- 1 lone AI Disagree
    l := add_label('L1', 1, 0, 1, 0, FALSE); PERFORM vote(l, '{ai}', 'Disagree');
    -- 2 the AI broke a human tie
    l := add_label('L1', 1, 1, 2, 0, FALSE);
    PERFORM vote(l, 'H1', 'Agree'); PERFORM vote(l, 'H2', 'Disagree'); PERFORM vote(l, '{ai}', 'Disagree');
    -- 3 humans ahead by one, the AI tied it
    l := add_label('L1', 1, 2, 2, 0, NULL);
    PERFORM vote(l, 'H1', 'Agree'); PERFORM vote(l, 'H2', 'Agree'); PERFORM vote(l, 'H3', 'Disagree');
    PERFORM vote(l, '{ai}', 'Disagree');
    -- 4 humans also voted it down
    l := add_label('L1', 1, 0, 2, 0, FALSE); PERFORM vote(l, 'H1', 'Disagree'); PERFORM vote(l, '{ai}', 'Disagree');
    -- 5 deleted in Explore (never counts toward accuracy)
    l := add_label('L1', 1, 0, 1, 0, FALSE, TRUE, 'Explore'); PERFORM vote(l, '{ai}', 'Disagree');
    -- 6 an excluded validator's Disagree, which never counts
    l := add_label('L1', 1, 0, 1, 0, FALSE); PERFORM vote(l, 'X', 'Disagree'); PERFORM vote(l, '{ai}', 'Disagree');
    -- 7 an AI Agree, untouched
    l := add_label('L1', 1, 1, 0, 0, TRUE); PERFORM vote(l, '{ai}', 'Agree');
    -- 8 an AI Disagree on the label's old type, which does not count
    l := add_label('L1', 1, 0, 0, 0, NULL); PERFORM vote(l, '{ai}', 'Disagree', 'NoCurbRamp');
    -- 9 a tutorial label on the tutorial street
    l := add_label('L1', 4, 0, 1, 0, FALSE, FALSE, NULL, TRUE); PERFORM vote(l, '{ai}', 'Disagree');
    -- 10 deleted from the LabelMap while incorrect: counts toward accuracy until it is no longer incorrect
    l := add_label('L1', 1, 0, 1, 0, FALSE, TRUE, 'LabelMap'); PERFORM vote(l, '{ai}', 'Disagree');
    INSERT INTO label_ai_assessment (label_id, label_validation_id, validation_result)
    SELECT label_id, label_validation_id, 'Disagree' FROM label_validation WHERE label_id = 1 AND user_id = '{ai}';

    -- L2: 29 right, 20 wrong by a human, 1 wrong by the AI alone: 50 judged at 0.58, so low quality.
    FOR i IN 1..29 LOOP l := add_label('L2', 2, 1, 0, 0, TRUE); PERFORM vote(l, 'H1', 'Agree'); END LOOP;
    FOR i IN 1..20 LOOP l := add_label('L2', 2, 0, 1, 0, FALSE); PERFORM vote(l, 'H1', 'Disagree'); END LOOP;
    l := add_label('L2', 2, 0, 1, 0, FALSE); PERFORM vote(l, '{ai}', 'Disagree');
    -- L3: 36 right, 23 wrong, 1 wrong by an AI tie-break: 60 judged at exactly 0.6, which is not > 0.6.
    FOR i IN 1..36 LOOP l := add_label('L3', 3, 1, 0, 0, TRUE); PERFORM vote(l, 'H1', 'Agree'); END LOOP;
    FOR i IN 1..23 LOOP l := add_label('L3', 3, 0, 1, 0, FALSE); PERFORM vote(l, 'H1', 'Disagree'); END LOOP;
    l := add_label('L3', 3, 1, 2, 0, FALSE);
    PERFORM vote(l, 'H1', 'Agree'); PERFORM vote(l, 'H2', 'Disagree'); PERFORM vote(l, '{ai}', 'Disagree');
    -- L4: manually low quality, one lone AI Disagree
    l := add_label('L4', 5, 0, 1, 0, FALSE); PERFORM vote(l, '{ai}', 'Disagree');
END $$;
"""

# City B reuses label_id 1 for an unrelated label: nothing may be keyed across cities on label_id.
CITY_B_SEED = """
INSERT INTO audit_task (user_id, street_edge_id) VALUES ('L1', 1);
INSERT INTO user_stat (user_id, meters_audited, labels_per_meter, high_quality, own_labels_validated, accuracy)
VALUES ('L1', 100, 0.5, TRUE, 1, 0.0);
SELECT vote(add_label('L1', 1, 0, 1, 0, FALSE), '{ai}', 'Disagree');
"""


class Pg:
    """Runs SQL and the scripts against the test database, locally or inside a Docker container."""

    def __init__(self, container=None, env=None):
        self.container = container
        self.env = env or {}

    def _cmd(self, argv, env):
        if self.container is None:
            return argv, {**os.environ, **env}
        flags = []
        for k, v in env.items():
            flags += ["-e", f"{k}={v}"]
        return ["docker", "exec", "-i", *flags, self.container, *argv], None

    def sql(self, query, db=DB):
        argv, env = self._cmd(["psql", "-X", "-q", "-At", "-v", "ON_ERROR_STOP=1", "-d", db], self.env)
        out = subprocess.run(argv, input=query, capture_output=True, text=True, env=env)
        if out.returncode != 0:
            raise RuntimeError(out.stderr)
        return out.stdout.strip()

    def run(self, script, *args, stdin=""):
        target = self.put(script)
        argv, env = self._cmd(["bash", target, *args], {**self.env, "DB_NAME": DB,
                                                        "DB_PORT": self.env.get("PGPORT", "5432")})
        return subprocess.run(argv, input=stdin, capture_output=True, text=True, env=env)

    def put(self, path):
        if self.container is None:
            return str(path)
        subprocess.run(["docker", "cp", str(path), f"{self.container}:/tmp/{path.name}"], check=True)
        return f"/tmp/{path.name}"

    def read(self, path):
        if self.container is None:
            return Path(path).read_text()
        return subprocess.run(["docker", "exec", self.container, "cat", path], capture_output=True, text=True,
                              check=True).stdout

    def tmpdir(self):
        if self.container is None:
            d = Path(os.environ.get("TMPDIR", "/tmp")) / f"ai-disagree-{uuid.uuid4().hex[:8]}"
            d.mkdir()
            return str(d)
        d = f"/tmp/ai-disagree-{uuid.uuid4().hex[:8]}"
        subprocess.run(["docker", "exec", self.container, "mkdir", "-p", d], check=True)
        return d


def _docker_ok():
    if shutil.which("docker") is None:
        return False
    return subprocess.run(["docker", "info"], capture_output=True).returncode == 0


@pytest.fixture(scope="module")
def pg():
    local_db = os.environ.get("AI_DISAGREE_TEST_DB")
    if local_db and shutil.which("psql"):
        global DB
        DB = local_db
        yield Pg()
        return
    if not _docker_ok():
        pytest.skip("needs Docker or a local psql with AI_DISAGREE_TEST_DB set")
    name = f"ai-disagree-test-{uuid.uuid4().hex[:8]}"
    started = subprocess.run(["docker", "run", "--rm", "-d", "--name", name, "-e", "POSTGRES_HOST_AUTH_METHOD=trust",
                              "postgres:16"], capture_output=True, text=True)
    if started.returncode != 0:
        pytest.skip(f"could not start postgres:16: {started.stderr.strip()}")
    try:
        env = {"PGUSER": "postgres"}
        db = Pg(container=name, env=env)
        for _ in range(60):
            ready = subprocess.run(["docker", "exec", name, "psql", "-U", "postgres", "-Atc", "SELECT 1"],
                                   capture_output=True, text=True)
            if ready.returncode == 0 and ready.stdout.strip() == "1":
                break
            time.sleep(1)
        else:
            pytest.skip("postgres:16 did not come up")
        time.sleep(1)  # the image restarts the server once after initdb
        for _ in range(30):
            if subprocess.run(["docker", "exec", name, "psql", "-U", "postgres", "-Atc", f"CREATE DATABASE {DB}"],
                              capture_output=True).returncode == 0:
                break
            time.sleep(1)
        yield db
    finally:
        subprocess.run(["docker", "stop", name], capture_output=True)


@pytest.fixture()
def seeded(pg):
    pg.sql(TYPES_SQL)
    for schema in ("sidewalk_login", "sidewalk_x_backup"):
        pg.sql(f"DROP SCHEMA IF EXISTS {schema} CASCADE; CREATE SCHEMA {schema};")
    pg.sql(CITY_TABLES.format(s="sidewalk_testa", evo=420) + CITY_A_SEED.format(ai=AI))
    pg.sql(CITY_TABLES.format(s="sidewalk_testb", evo=420) + CITY_B_SEED.format(ai=AI))
    pg.sql(CITY_TABLES.format(s="sidewalk_testold", evo=399) + CITY_B_SEED.format(ai=AI))
    return pg


def rows(stdout):
    return {(r["phase"], r["schema"]): r for r in csv.DictReader(io.StringIO(stdout))}


def ints(row, *cols):
    return tuple(int(row[c]) for c in cols)


def snapshot(pg, schema):
    return pg.sql(f"""SELECT md5(string_agg(t, '|' ORDER BY t)) FROM (
        SELECT (label_validation.*)::text AS t FROM {schema}.label_validation UNION ALL
        SELECT (label.*)::text FROM {schema}.label UNION ALL SELECT (user_stat.*)::text FROM {schema}.user_stat) AS x""")


EXPECTED_A = {
    "ai_disagree_rows": 12, "on_deleted_or_tutorial": 3, "labels_recounted": 11,
    "false_to_null": 9, "null_to_true": 1, "correct_unchanged": 1, "other_transition": 0,
    "becomes_visible_now": 3, "becomes_visible_after_quality_flip": 2, "stays_hidden_low_quality": 1,
    "becomes_hidden": 0, "labelers_touched": 4, "accuracy_changed": 4, "quality_low_to_high": 2,
    "quality_high_to_low": 0, "no_longer_judged": 1, "stat_rows_rewritten": 5, "quality_rows_changed": 3,
}


def test_dry_run_reports_and_writes_nothing(seeded):
    before = {s: snapshot(seeded, s) for s in ("sidewalk_testa", "sidewalk_testb")}
    out = seeded.run(SCRIPT)
    assert out.returncode == 0, out.stderr
    r = rows(out.stdout)
    a = r[("dry_run", "sidewalk_testa")]
    assert {k: int(a[k]) for k in EXPECTED_A} == EXPECTED_A
    b = r[("dry_run", "sidewalk_testb")]
    assert ints(b, "ai_disagree_rows", "labels_recounted", "false_to_null", "becomes_visible_now",
                "accuracy_changed", "quality_rows_changed") == (1, 1, 1, 1, 1, 0)
    assert int(r[("dry_run", "TOTAL")]["ai_disagree_rows"]) == 13
    # skipped: too old, the login schema, backups
    assert {k[1] for k in r} == {"sidewalk_testa", "sidewalk_testb", "TOTAL"}
    assert "skipping sidewalk_testold" in out.stderr
    assert {s: snapshot(seeded, s) for s in before} == before


def test_apply_requires_backup_dir_and_confirmation(seeded):
    before = snapshot(seeded, "sidewalk_testa")
    out = seeded.run(SCRIPT, "--apply")
    assert out.returncode == 1 and "--backup-dir" in out.stderr
    out = seeded.run(SCRIPT, "--apply", "--backup-dir", seeded.tmpdir())  # no --yes, no terminal
    assert out.returncode == 1 and "needs a terminal" in out.stderr
    assert snapshot(seeded, "sidewalk_testa") == before


def test_apply_converts_and_recomputes(seeded):
    backup = seeded.tmpdir()
    out = seeded.run(SCRIPT, "--cities", "testa,sidewalk_testb", "--apply", "--yes", "--backup-dir", backup)
    assert out.returncode == 0, out.stderr
    r = rows(out.stdout)
    assert {k: int(r[("before", "sidewalk_testa")][k]) for k in EXPECTED_A} == EXPECTED_A
    after = r[("after", "sidewalk_testa")]
    for col in ("ai_disagree_rows", "labels_recounted", "accuracy_changed", "quality_low_to_high",
                "stat_rows_rewritten", "quality_rows_changed"):
        assert int(after[col]) == 0, col
    assert "step 1: 12 AI vote row(s)" in out.stderr and "step 2: 11 label(s)" in out.stderr

    q = lambda s: seeded.sql(f"SET search_path = sidewalk_testa, public; {s}")
    # The AI says Disagree nowhere; its Agree and the assessment row are untouched.
    assert q(f"SELECT validation_result, COUNT(*) FROM label_validation WHERE user_id = '{AI}' "
             "GROUP BY 1 ORDER BY 1") == "Agree|1\nUnsure|12"
    assert q("SELECT validation_result FROM label_ai_assessment") == "Disagree"
    labels = q("SELECT label_id, agree_count, disagree_count, unsure_count, COALESCE(correct::text, 'null') "
               "FROM label WHERE label_id <= 10 ORDER BY 1").splitlines()
    assert labels == [
        "1|0|0|1|null", "2|1|1|1|null", "3|2|1|1|true", "4|0|1|1|false", "5|0|0|1|null",
        "6|0|0|1|null", "7|1|0|0|true", "8|0|0|0|null", "9|0|0|1|null", "10|0|0|1|null"]
    stats = q("SELECT user_id, own_labels_validated, round(accuracy::numeric, 4), high_quality FROM user_stat "
              "WHERE user_id LIKE 'L%' ORDER BY user_stat_id").splitlines()
    assert stats == ["L1|3|0.6667|t", "L2|49|0.5918|t", "L2|49|0.5918|t", "L3|59|0.6102|t", "L4|0||f"]

    # The backup names its city on every row and holds exactly the converted rows.
    run_dir = out.stderr.split("Backups and per-city reports: ")[1].split()[0]
    backup_a = list(csv.DictReader(io.StringIO(seeded.read(f"{run_dir}/sidewalk_testa_ai_disagree_votes.csv"))))
    assert len(backup_a) == 12 and {row["schema_name"] for row in backup_a} == {"sidewalk_testa"}
    backup_b = list(csv.DictReader(io.StringIO(seeded.read(f"{run_dir}/sidewalk_testb_ai_disagree_votes.csv"))))
    assert [(row["schema_name"], row["label_id"]) for row in backup_b] == [("sidewalk_testb", "1")]

    # City B's label 1 is its own label, recounted independently of city A's label 1.
    assert seeded.sql("SELECT unsure_count, correct IS NULL FROM sidewalk_testb.label WHERE label_id = 1") == "1|t"

    # A second run has nothing left to do, and the dry run agrees.
    again = seeded.run(SCRIPT, "--cities", "testa", "--apply", "--yes", "--backup-dir", backup)
    assert again.returncode == 0, again.stderr
    assert "nothing to convert in sidewalk_testa" in again.stderr
    dry = rows(seeded.run(SCRIPT, "--cities", "testa").stdout)[("dry_run", "sidewalk_testa")]
    assert int(dry["labels_recounted"]) == 0 and int(dry["stat_rows_rewritten"]) == 0


def test_apply_rolls_back_when_a_step_is_off(seeded):
    """A backup that does not match the database (here: a row the database does not have) aborts the city whole."""
    before = snapshot(seeded, "sidewalk_testa")
    backup = seeded.tmpdir()
    # Make step 1 change fewer rows than the backup holds: flip one AI vote to Unsure after the backup is taken by
    # running the apply with a PSQL wrapper that does so just before the write transaction.
    wrapper = Path(os.environ.get("TMPDIR", "/tmp")) / f"psql-wrap-{uuid.uuid4().hex[:6]}.sh"
    wrapper.write_text(
        "#!/bin/bash\n"
        "# Before the write connection (the only one with a lock_timeout), sneak one vote out of Disagree.\n"
        "if [[ \"$*\" == *lock_timeout* ]]; then\n"
        f"  PGOPTIONS= psql -X -q -d {DB} -c \"UPDATE sidewalk_testa.label_validation SET validation_result = 'Unsure'"
        f" WHERE label_validation_id = (SELECT min(label_validation_id) FROM sidewalk_testa.label_validation"
        f" WHERE user_id = '{AI}' AND validation_result = 'Disagree')\" >/dev/null\n"
        "fi\n"
        "exec psql \"$@\"\n")
    target = seeded.put(wrapper)
    wrapper.unlink()
    env_backup = dict(seeded.env)
    seeded.env = {**seeded.env, "PSQL": f"bash {target}"}
    try:
        out = seeded.run(SCRIPT, "--cities", "testa", "--apply", "--yes", "--backup-dir", backup)
    finally:
        seeded.env = env_backup
    assert out.returncode == 1
    assert "step 1 changed 11 rows, not the 12 in the backup" in out.stderr
    # Only the wrapper's own sneaked change is there; the script's transaction left nothing behind.
    assert seeded.sql(f"SELECT COUNT(*) FROM sidewalk_testa.label_validation WHERE user_id = '{AI}' "
                      "AND validation_result = 'Unsure'") == "1"
    seeded.sql(f"UPDATE sidewalk_testa.label_validation SET validation_result = 'Disagree' "
               f"WHERE user_id = '{AI}' AND validation_result = 'Unsure'")
    assert snapshot(seeded, "sidewalk_testa") == before


def test_census(seeded):
    out = seeded.run(CENSUS)
    assert out.returncode == 0, out.stderr
    r = {(row["schema"], row["label_type"]): row for row in csv.DictReader(io.StringIO(out.stdout))}
    a = r[("sidewalk_testa", "CurbRamp")]
    # Non-deleted, non-tutorial labels with an AI Disagree on their current type: 1,2,3,4,6 + L2 + L3 + L4.
    assert ints(a, "ai_disagree_votes", "hidden", "lone_ai", "ai_broke_tie", "humans_voted",
                "human_majority_disagree", "human_majority_agree", "human_tied") == (8, 7, 4, 2, 4, 1, 1, 2)
    # testold (evolution 399) is new enough for the census, which needs only 395.
    assert int(r[("ALL", "ALL")]["ai_disagree_votes"]) == 10
    assert r[("ALL", "ALL")]["human_majority_sides_with_ai"] == "0.500"
