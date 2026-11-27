"""The resolution study's klone files (docs/resolution-2026-11.md): they parse; the job is preemptible, requeues and
runs offline in the backbone study's env; run_one passes --input-res, refuses an empty field or a size outside the
study, keeps the records and the best weights (checking the saved processor crops the run's size) before deleting
checkpoints, and stops at its done guard; the launcher writes the planned rows with resolution_study's tags, refuses
518 without ALLOW_518, a smoke tag outside the smoke job and the 93rd GPU job, and resumes only started, unfinished,
not-in-flight rows. Everything runs against stubs (sbatch, squeue, hyakstorage, df, curl and a fake python)."""
import json
import os
import re
import shutil
import stat
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANALYSIS = os.path.join(REPO, "analysis")
SCRIPTS = ["resolution.slurm", "resolution_submit.sh"]
pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="no bash")


def read(name):
    with open(os.path.join(ANALYSIS, name)) as f:
        return f.read()


@pytest.mark.parametrize("name", SCRIPTS)
def test_bash_parses(name):
    r = subprocess.run(["bash", "-n", os.path.join(ANALYSIS, name)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_slurm_header_env_and_offline():
    text = read("resolution.slurm")
    for line in ("#SBATCH --partition=ckpt-all", "#SBATCH --requeue", "#SBATCH --account=makelab",
                 "#SBATCH --job-name=resolution-study", "#SBATCH --open-mode=append"):
        assert line in text
    assert "HF_HUB_OFFLINE=1" in text
    body = text[text.index("setup_env() {"):text.index("run_one() {")]
    assert '"transformers": "4.56.2"' in body and '"torch": "2.7.1"' in body and "exit 3" in body
    submit = read("resolution_submit.sh")
    names = re.findall(r'--job-name="([^"]+)"', submit)
    assert names and all(n.startswith("$SLUG-") for n in names) and "SLUG=resolution-study" in submit
    assert "JOB_CAP=92" in submit and "analysis/resolution_inputs.json" in submit


@pytest.mark.parametrize("name", SCRIPTS)
def test_every_required_variable_is_documented_in_the_header(name):
    text = read(name)
    header = "\n".join(ln for ln in text.splitlines() if ln.startswith("#"))
    for var in set(re.findall(r"\$\{([A-Za-z_]\w*):\?", text)) - {"SLURM_ARRAY_TASK_ID"}:
        assert re.search(rf"\b{var}\b", header), f"{name}: {var} is required but not documented"


# ---------------------------------------------------------------------------------------------------------------------
# run_one
# ---------------------------------------------------------------------------------------------------------------------

FAKE_PY = r"""#!/bin/bash
# A stand-in for python in run_one: logs each call, and writes what the real script would.
echo "$*" >> "$CALLS"
if [ "$1" = -u ] && [ "$2" = fine_tune_classifier.py ]; then
  out=$(echo "$*" | sed -E 's/.*--output-dir ([^ ]+).*/\1/')
  res=$(echo "$*" | sed -E 's/.*--input-res ([^ ]+).*/\1/')
  [ -z "${WRONG_RES:-}" ] || res="$WRONG_RES"
  mkdir -p "$out/checkpoint-10" "$out/checkpoint-20"
  for c in 10 20; do
    for f in model.safetensors config.json trainer_state.json; do echo "$f" > "$out/checkpoint-$c/$f"; done
    echo "{\"crop_size\": {\"height\": $res, \"width\": $res}}" > "$out/checkpoint-$c/preprocessor_config.json"
  done
  [ -z "${DROP:-}" ] || rm "$out/checkpoint-20/$DROP"
  echo '{}' > "$out/training_run.json"; echo '{}' > "$out/run_key.json"
elif [ "$1" = -c ]; then
  echo "$3/checkpoint-20"
elif [ "$1" = - ]; then
  shift; exec "$REAL_PY" - "$@"
elif [ "$1" = -u ] && [ "$2" = evaluate.py ]; then
  [ -z "${EVAL_FAIL:-}" ] || exit 1
  lt=$(echo "$*" | sed -E 's/.*--label_type ([^ ]+).*/\1/'); tag=$(echo "$*" | sed -E 's/.*--tag ([^ ]+).*/\1/')
  data=$(echo "$*" | sed -E 's/.*--data-dir ([^ ]+).*/\1/')
  for f in "metrics_$tag.json" "predictions_$tag.csv" "per_city_$tag.md" "slices_$tag.md" "report_$tag.txt"; do
    echo x > "$data/$lt/$f"
  done
fi
"""


@pytest.fixture
def job(tmp_path):
    py = tmp_path / "fakepy"
    py.write_text(FAKE_PY)
    py.chmod(py.stat().st_mode | stat.S_IEXEC)
    (tmp_path / "data" / "Obstacle").mkdir(parents=True)
    env = {**os.environ, "PY": str(py), "REAL_PY": sys.executable, "CALLS": str(tmp_path / "calls.log"),
           "DATA": str(tmp_path / "data"), "RUNS": str(tmp_path / "runs"), "KEEP": str(tmp_path / "keep"),
           "PATH": "/usr/bin:/bin"}

    def call(*args, **extra):
        cmd = f'source "{os.path.join(ANALYSIS, "resolution.slurm")}"; run_one ' + " ".join(f'"{a}"' for a in args)
        return subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, cwd=REPO, env={**env, **extra})
    return call, tmp_path


def calls(tmp_path):
    p = tmp_path / "calls.log"
    return p.read_text().splitlines() if p.exists() else []


@pytest.mark.parametrize("args", [("Obstacle", "448", "42", ""), ("Obstacle", "", "42", "res-r448"),
                                  ("Obstacle", "384", "42", "res-r384")])
def test_run_one_refuses_an_empty_field_or_another_size(job, args):
    call, tmp = job
    r = call(*args)
    assert r.returncode != 0 and "refusing" in r.stderr and calls(tmp) == []


def test_run_one_passes_the_size_keeps_records_and_best_and_stops_at_done(job):
    call, tmp = job
    r = call("Obstacle", "448", "43", "res-r448-s43")
    assert r.returncode == 0, r.stderr
    run, keep = tmp / "runs" / "Obstacle" / "res-r448-s43", tmp / "keep" / "Obstacle" / "runs" / "res-r448-s43"
    assert sorted(os.listdir(run / "best")) == ["config.json", "model.safetensors", "preprocessor_config.json",
                                                "trainer_state.json"]
    assert not list(run.glob("checkpoint-*"))
    assert {"metrics_res-r448-s43.json", "predictions_res-r448-s43.csv", "training_run.json", "run_key.json",
            "trainer_state.json", "preprocessor_config.json", "done"} <= set(os.listdir(keep))
    assert not list(keep.glob("*.safetensors"))
    c = [ln for ln in calls(tmp) if ln.startswith("-u")]
    assert [ln.split()[1] for ln in c] == ["fine_tune_classifier.py", "evaluate.py"]
    assert "--input-res 448 --seed 43 --epochs 25 --num-workers 7" in c[0]
    assert "--batch-size" not in c[0] and "--gradient-checkpointing" not in c[0]
    assert "--tag res-r448-s43 --per-city" in c[1]
    before = calls(tmp)
    r2 = call("Obstacle", "448", "43", "res-r448-s43")
    assert r2.returncode == 0 and "done already" in r2.stdout and calls(tmp) == before


def test_the_memory_fallbacks_reach_the_training_command(job):
    call, tmp = job
    assert call("Obstacle", "518", "42", "res-r518", BATCH="4", GC="1").returncode == 0
    c = calls(tmp)[0]
    assert "--input-res 518" in c and "--batch-size 4" in c and "--gradient-checkpointing" in c


def test_a_processor_at_the_wrong_size_is_never_scored(job):
    """PR #133 review S2: the check runs before evaluate.py, so no predictions of a wrong-size run reach $KEEP."""
    call, tmp = job
    r = call("Obstacle", "448", "42", "res-r448", WRONG_RES="224")
    assert r.returncode != 0 and "does not crop 448" in r.stderr
    assert not [ln for ln in calls(tmp) if "evaluate.py" in ln]
    assert not list((tmp / "keep" / "Obstacle" / "runs" / "res-r448").glob("predictions_*"))
    run = tmp / "runs" / "Obstacle" / "res-r448"
    assert len(list(run.glob("checkpoint-*"))) == 2
    assert not (tmp / "keep" / "Obstacle" / "runs" / "res-r448" / "done").exists()


def test_a_failed_copy_leaves_every_checkpoint(job):
    call, tmp = job
    r = call("Obstacle", "448", "42", "res-r448", DROP="config.json")
    assert r.returncode != 0
    assert len(list((tmp / "runs" / "Obstacle" / "res-r448").glob("checkpoint-*"))) == 2


def test_a_requeue_after_training_scores_without_retraining(job):
    """PR #133 review S6: preempted (here: evaluate.py failed) after training finished, the requeued element scores
    the run without re-running the Trainer, which would rewrite training_run.json from a zero-step attempt."""
    call, tmp = job
    assert call("Obstacle", "448", "42", "res-r448", EVAL_FAIL="1").returncode != 0
    run = tmp / "runs" / "Obstacle" / "res-r448"
    assert (run / "trained").exists() and not (run / "evaluated").exists()
    before = len([ln for ln in calls(tmp) if "fine_tune_classifier.py" in ln])
    r = call("Obstacle", "448", "42", "res-r448")
    assert r.returncode == 0, r.stderr
    assert len([ln for ln in calls(tmp) if "fine_tune_classifier.py" in ln]) == before == 1
    assert (tmp / "keep" / "Obstacle" / "runs" / "res-r448" / "preprocessor_config.json").exists()


def test_memory_check_needs_sdpa_and_a_peak_under_the_ceiling(tmp_path):
    def run(records, **env):
        for i, rec in enumerate(records):
            d = tmp_path / "keep" / "CurbRamp" / "runs" / f"r{i}"
            d.mkdir(parents=True, exist_ok=True)
            (d / "training_run.json").write_text(json.dumps(rec))
        cmd = f'source "{os.path.join(ANALYSIS, "resolution.slurm")}"; PY="{sys.executable}"; memory_check "{tmp_path / "keep"}"'
        return subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, env={**os.environ, **env})
    ok = {"input_res": 448, "attn_implementation": "sdpa", "peak_gpu_memory_bytes": 18 * 2**30}
    assert run([ok]).returncode == 0
    assert run([ok, {**ok, "attn_implementation": "eager"}]).returncode != 0
    shutil.rmtree(tmp_path / "keep")
    assert run([{**ok, "peak_gpu_memory_bytes": 40 * 2**30}]).returncode != 0
    shutil.rmtree(tmp_path / "keep")
    assert run([{**ok, "peak_gpu_memory_bytes": 40 * 2**30}], MEM_CEILING_GB="44").returncode == 0


def test_a_requeue_after_best_was_copied_finishes_the_run(job):
    call, tmp = job
    assert call("Obstacle", "224", "42", "res-r224").returncode == 0
    run, keep = tmp / "runs" / "Obstacle" / "res-r224", tmp / "keep" / "Obstacle" / "runs" / "res-r224"
    os.remove(run / "slimmed")
    os.remove(keep / "done")
    (run / "checkpoint-10").mkdir()
    r = call("Obstacle", "224", "42", "res-r224")
    assert r.returncode == 0, r.stderr
    assert (keep / "done").exists() and (run / "slimmed").exists() and not list(run.glob("checkpoint-*"))


# ---------------------------------------------------------------------------------------------------------------------
# the launcher
# ---------------------------------------------------------------------------------------------------------------------

TYPES = ["Obstacle", "SurfaceProblem", "NoCurbRamp", "CurbRamp", "Crosswalk"]
STUBS = {
    "sbatch": '#!/bin/bash\necho "$* STEP=$STEP TASKS=${TASKS:-} KEEP=$KEEP EPOCHS=${EPOCHS:-} BATCH=${BATCH:-}" '
              '>> "$SB_LOG"\necho 4242\n',
    "hyakstorage": '#!/bin/bash\necho "│ Total:  │ 500GB / 1024GB  │ 500000 / 1000000 files │"\n',
    "df": '#!/bin/bash\necho "Filesystem 1024-blocks Used Available Capacity Mounted"\n'
          'echo "scrubbed 9999999999 1 ${DF_AVAIL_KB:-9999999999} 1% /x"\n',
    "curl": '#!/bin/bash\nwhile [ $# -gt 0 ]; do [ "$1" = -o ] && echo \'{"cities": []}\' > "$2"; shift; done\n',
    "squeue": '#!/bin/bash\n[ -z "${SQ_FAIL:-}" ] || exit 1\n[ -z "${SQ_OUT:-}" ] || cat "$SQ_OUT"\n',
}


@pytest.fixture
def launcher(tmp_path):
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    for name, body in STUBS.items():
        p = bin_ / name
        p.write_text(body)
        p.chmod(0o755)
    out, keep = tmp_path / "out", tmp_path / "keep"
    for lt in TYPES:
        (out / "data" / lt).mkdir(parents=True)
        (out / "data" / lt / "split.csv").write_text("x\n")
    (out / "smoke").mkdir()
    (out / "smoke" / "resolution_inputs_smoke.json").write_text("{}")
    env = {**os.environ, "PATH": f"{bin_}:{os.environ['PATH']}", "OUT": str(out), "KEEP": str(keep),
           "R": str(tmp_path / "r3"), "PY": sys.executable, "SB_LOG": str(tmp_path / "sbatch.log")}

    def run(*args, **extra):
        return subprocess.run(["bash", os.path.join(ANALYSIS, "resolution_submit.sh"), *args], capture_output=True,
                              text=True, cwd=REPO, env={**env, **extra})
    return {"run": run, "out": out, "keep": keep, "log": tmp_path / "sbatch.log"}


def rows(path):
    return [ln.split("\t") for ln in open(path).read().splitlines() if ln and not ln.startswith("#")]


def test_main_writes_fifty_rows_with_the_studys_tags_and_one_throttled_array(launcher):
    from analysis import resolution_study as rs
    r = launcher["run"]("main")
    assert r.returncode == 0, r.stderr
    t = rows(sorted(launcher["out"].glob("tasks_main_*.tsv"))[0])
    assert len(t) == 50
    assert {(x[0], x[1], x[2]) for x in t} == {(lt, str(res), str(s)) for lt in TYPES for res in (224, 448)
                                               for s in rs.SEEDS}
    for lt, res, seed, tag in t:
        assert tag == rs.make_tag(int(res), int(seed))
    assert {x[2] for x in t[:10]} == {"42"}   # seed 42 of every row first, so a partial study is still paired
    sb = launcher["log"].read_text()
    assert "--job-name=resolution-study-main" in sb and "--array=1-50%8" in sb and "STEP=train" in sb
    assert "EPOCHS=25" in sb and "BATCH=8" in sb
    ledger = rows(launcher["out"] / "resolution_jobs.txt")
    assert ledger[0][1:4] == ["4242", "50", "main"] and len(ledger[0][6]) == 64
    assert (launcher["keep"] / "resolution_jobs.txt").exists()


def test_518_needs_jons_word(launcher):
    r = launcher["run"]("r518")
    assert r.returncode != 0 and "ALLOW_518" in r.stderr and not launcher["log"].exists()
    r = launcher["run"]("r518", ALLOW_518="1")
    assert r.returncode == 0, r.stderr
    t = rows(sorted(launcher["out"].glob("tasks_r518_*.tsv"))[0])
    assert len(t) == 25 and {x[1] for x in t} == {"518"}


def test_the_job_cap_refuses_the_93rd_element(launcher):
    out = launcher["out"]
    (out / "resolution_jobs.txt").write_text("t\t1\t43\tsmoke\tc\t-\t-\n")
    r = launcher["run"]("main")
    assert r.returncode != 0 and "cap" in r.stderr and not launcher["log"].exists()
    (out / "resolution_jobs.txt").write_text("t\t1\t42\tsmoke\tc\t-\t-\n")
    assert launcher["run"]("main").returncode == 0


def test_too_little_scrubbed_space_refuses(launcher):
    r = launcher["run"]("main", DF_AVAIL_KB="1048576")
    assert r.returncode != 0 and "scrubbed" in r.stderr and not launcher["log"].exists()


def test_done_runs_are_skipped(launcher):
    d = launcher["keep"] / "Obstacle" / "runs" / "res-r448"
    d.mkdir(parents=True)
    (d / "done").write_text("")
    r = launcher["run"]("main", TYPES="Obstacle", SEEDS="42")
    assert r.returncode == 0, r.stderr
    t = rows(sorted(launcher["out"].glob("tasks_main_*.tsv"))[0])
    assert [x[3] for x in t] == ["res-r224"] and "skipping Obstacle res-r448: done already" in r.stdout


def test_smoke_is_one_two_hour_job_over_three_sizes(launcher):
    r = launcher["run"]("smoke")
    assert r.returncode == 0, r.stderr
    t = rows(launcher["out"] / "smoke" / "tasks_smoke.tsv")
    assert [x[3] for x in t] == ["smoke-res-r224", "smoke-res-r448", "smoke-res-r518"]
    sb = launcher["log"].read_text()
    assert sb.count("\n") == 1 and "--time=02:00:00" in sb and "STEP=smoke" in sb and "EPOCHS=1" in sb
    assert (launcher["out"] / "smoke" / ".start").exists()


def test_bench_is_one_a40_job(launcher):
    r = launcher["run"]("bench")
    assert r.returncode == 0, r.stderr
    sb = launcher["log"].read_text()
    assert "--constraint=a40" in sb and "STEP=bench" in sb and "--job-name=resolution-study-bench" in sb
    assert rows(launcher["out"] / "resolution_jobs.txt")[0][3:4] == ["bench"]


def test_resume_takes_started_unfinished_rows_not_in_flight(launcher, tmp_path):
    out, keep = launcher["out"], launcher["keep"]
    assert launcher["run"]("main", TYPES="Obstacle").returncode == 0
    main = sorted(out.glob("tasks_main_*.tsv"))[0]
    launcher["log"].unlink()
    for tag in ("res-r448", "res-r448-s43", "res-r224"):   # started
        (out / "runs" / "Obstacle" / tag).mkdir(parents=True)
    (keep / "Obstacle" / "runs" / "res-r224").mkdir(parents=True)
    (keep / "Obstacle" / "runs" / "res-r224" / "done").write_text("")   # finished
    idx = [x[3] for x in rows(main)].index("res-r448-s43") + 1
    q = tmp_path / "squeue.txt"
    q.write_text(f"4242_{idx} resolution-study-main\n555_1 someone-else\n")
    r = launcher["run"]("resume", SQ_OUT=str(q))
    assert r.returncode == 0, r.stderr
    t = rows(sorted(out.glob("tasks_resume_*.tsv"))[0])
    assert [x[3] for x in t] == ["res-r448"]
    assert "--job-name=resolution-study-resume" in launcher["log"].read_text()
    assert not list(out.glob(".squeue*"))


def test_an_unreadable_squeue_refuses(launcher):
    r = launcher["run"]("main", SQ_FAIL="1")
    assert r.returncode != 0 and "squeue" in r.stderr and not launcher["log"].exists()


def test_a_smoke_tag_never_reaches_a_main_array():
    text = read("resolution_submit.sh")
    body = text[text.index("submit_array() {"):text.index("taskfile() {")]
    assert "grep -q '^smoke-'" in body and body.index("grep -q '^smoke-'") < body.index('"$SBATCH"')
