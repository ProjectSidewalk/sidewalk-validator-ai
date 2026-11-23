"""The context-arms job files (issue #14): they parse, run preemptible with requeue and offline, document every
required variable, refuse tags that would overwrite October's files, copy the run record before deleting checkpoints,
and the launcher's stage presets expand to exactly the planned jobs (DRY_RUN=1, nothing submitted)."""
import json
import os
import re
import shutil
import stat
import subprocess

import pytest

from analysis import confident_learning as cl

ROOT = os.path.join(os.path.dirname(__file__), "..", "analysis")
SCRIPTS = ["context_arms.slurm", "context_submit.sh"]
needs_bash = pytest.mark.skipif(shutil.which("bash") is None, reason="no bash")


def read(name):
    with open(os.path.join(ROOT, name)) as f:
        return f.read()


@needs_bash
@pytest.mark.parametrize("name", SCRIPTS)
def test_bash_parses(name):
    r = subprocess.run(["bash", "-n", os.path.join(ROOT, name)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_slurm_is_preemptible_requeues_and_is_offline():
    text = read("context_arms.slurm")
    assert "#SBATCH --partition=ckpt-all" in text and "#SBATCH --requeue" in text
    assert "#SBATCH --account=makelab" in text and "HF_HUB_OFFLINE=1" in text
    assert "#SBATCH --constraint=a40|l40|l40s|a100|h200" in text
    assert "already evaluated" in text and 'rm -rf "$RUN"/checkpoint-*' in text and "TIMEOUT is NOT requeued" in text
    assert os.access(os.path.join(ROOT, "context_submit.sh"), os.X_OK)


@pytest.mark.parametrize("name", SCRIPTS)
def test_every_required_variable_is_documented_in_the_header(name):
    text = read(name)
    header = "\n".join(ln for ln in text.splitlines() if ln.startswith("#"))
    required = set(re.findall(r"\$\{(\w+):\?", text))
    assert required, name
    for var in required:
        assert re.search(rf"\b{var}\b", header), f"{name}: {var} is required but not documented in the header"


def test_train_step_pins_the_recipe_and_scores_each_kind_of_run():
    train = read("context_arms.slurm").split("  train)")[1]
    assert "--target hard --best-metric macro_f1" in train and "--num-workers 7" in train
    assert '--context-arm "$ARM" --context-view-res "$RES"' in train
    # the baseline rerun is scored by evaluate.py from its checkpoint; an arm by context_model.py predict first
    none, arm = train.split('if [ "$ARM" = none ]; then\n      ')[1].split("    else\n", 1)
    assert 'evaluate.py --label_type "$LT" --data-dir "$DATA" --checkpoint "$BEST" --tag "$TAG" --per-city' in none
    assert arm.index("context_model.py predict") < arm.index("--skip-inference --tag \"$TAG\" --per-city")


def test_run_record_is_copied_before_any_checkpoint_is_deleted():
    train = read("context_arms.slurm").split("  train)")[1]
    record = train.index('cp "$RUN/training_run.json" "$RUN/run_key.json" "$REC/"')
    delete = train.index('prune_checkpoints "$KEEP_CKPT" "$(basename "$BEST")"')
    assert train.index("evaluate.py") < record < delete
    assert '"$BEST/trainer_state.json" "$REC/trainer_state.json"' in train and '"$REC/config.json"' in train
    # every delete is in prune_checkpoints, which runs only after the record (and in the short-circuit)
    assert train.count('rm -rf "$RUN"/checkpoint-*') == 1 and train.count('prune_checkpoints "$') == 2
    # the finished marker sits beside the record (not under RUNS) and goes in before any delete, so a preemption
    # between the delete and the KEEP copy, or a purged RUNS, never retrains a finished run
    marker = train.index('touch "$REC/.evaluated"')
    assert record < train.index("keep_ckpt=%s") < marker < delete and 'touch "$RUN/.evaluated"' not in train
    assert "node=" in train and "hostname -s" in train
    short = train.split("already evaluated")[1].split("exit 0")[0]
    assert "keep_copy" in short   # the KEEP copy is redone (idempotently) when a finished run is resubmitted


def test_train_holds_an_exclusive_lock_on_its_run_directory():
    train = read("context_arms.slurm").split("  train)")[1]
    assert 'exec 8>"$RUN/.train.lock"' in train and "flock -n 8" in train
    assert 'exec 7>"$DATA/$LT/.train-$TAG.lock"' in train and "flock -n 7" in train
    assert train.index("flock -n 7") < train.index("flock -n 8") < train.index("already evaluated") \
        < train.index("fine_tune_classifier.py")


def test_backfill_holds_an_exclusive_lock_and_verifies():
    bf = read("context_arms.slurm").split("  backfill)")[1].split("  train)")[0]
    assert 'exec 9>"$DATA/$LT/context/.backfill.lock"' in bf and "flock -n 9" in bf
    assert bf.index("flock -n 9") < bf.index("context_crops.py backfill") and "--verify 50" in bf
    assert '["rawlabels"]' in bf   # the rawLabels snapshot comes from the shadow's record


def run_slurm(tmp_path, **env):
    e = {"PATH": os.environ["PATH"], "HOME": str(tmp_path), "STEP": "train", "LT": "Toy", "DATA": str(tmp_path),
         "RUNS": str(tmp_path / "runs"), "ARM": "none", **env}
    return subprocess.run(["bash", os.path.join(ROOT, "context_arms.slurm")], env=e, capture_output=True, text=True)


@needs_bash
@pytest.mark.parametrize("tag", ["", "baseline", "baseline-s43", "none"])
def test_train_refuses_tags_that_would_overwrite_octobers_files(tmp_path, tag):
    r = run_slurm(tmp_path, TAG=tag)
    assert r.returncode == 2 and "refusing TAG" in r.stderr, r.stderr
    assert "module" not in r.stderr   # refused before any module or GPU work


@needs_bash
def test_train_refuses_unknown_arm_and_keep_ckpt(tmp_path):
    assert run_slurm(tmp_path, ARM="side_info_hq", TAG="x").returncode == 2
    r = run_slurm(tmp_path, TAG="baseline-rerun", KEEP_CKPT="some")
    assert r.returncode == 2 and "KEEP_CKPT" in r.stderr


# ------------------------------------------------------------------------------------------- the job, stubbed
# The train step run for real with stub commands on PATH: python (the trainer, the best-checkpoint lookup and
# evaluate.py write what the real ones would), module, conda, nvidia-smi, and flock (fcntl.flock on the inherited
# descriptor, which is what flock(1) does, so a lock the test holds is seen by the job).

STUB_PY = r"""
import json, os, sys
args = sys.argv[1:]
open(os.environ["STUB_LOG"], "a").write(" ".join(args) + "\n")
def opt(name):
    return args[args.index(name) + 1]
if args[:1] == ["-c"]:
    print(os.path.join(args[2], "checkpoint-20"))
elif "fine_tune_classifier.py" in args:
    out = opt("--output-dir")
    for c in ("checkpoint-10", "checkpoint-20"):
        os.makedirs(os.path.join(out, c), exist_ok=True)
        for n in ("trainer_state.json", "config.json", "model.safetensors", "optimizer.pt"):
            open(os.path.join(out, c, n), "w").write("{}")
    for n in ("training_run.json", "run_key.json"):
        json.dump({"seed": int(opt("--seed"))}, open(os.path.join(out, n), "w"))
elif "evaluate.py" in args:
    d = os.path.join(opt("--data-dir"), opt("--label_type"))
    open(os.path.join(d, "report_" + opt("--tag") + ".txt"), "w").write("report\n")
"""

STUB_FLOCK = r"""
import fcntl, sys
try:
    fcntl.flock(int(sys.argv[-1]), fcntl.LOCK_EX | fcntl.LOCK_NB)
except OSError:
    sys.exit(1)
"""


def stub_bin(tmp_path):
    import sys
    b = tmp_path / "stubbin"
    b.mkdir(exist_ok=True)
    for name, body in (("python", STUB_PY), ("flock", STUB_FLOCK)):
        (b / name).write_text(f"#!{sys.executable}\n{body}")
    for name in ("module", "conda"):
        (b / name).write_text("#!/bin/bash\nexit 0\n")
    (b / "nvidia-smi").write_text("#!/bin/bash\necho 'Stub GPU, 48 GiB'\n")
    for f in b.iterdir():
        f.chmod(f.stat().st_mode | stat.S_IEXEC)
    return b


def run_job(tmp_path, runs="runs", **env):
    """The train step of context_arms.slurm with stub commands; DATA is tmp_path/data (Toy/ exists)."""
    data = tmp_path / "data"
    (data / "Toy").mkdir(parents=True, exist_ok=True)
    e = {"PATH": f"{stub_bin(tmp_path)}:{os.environ['PATH']}", "HOME": str(tmp_path), "STEP": "train", "LT": "Toy",
         "DATA": str(data), "RUNS": str(tmp_path / runs), "ARM": "none", "TAG": "baseline-rerun",
         "STUB_LOG": str(tmp_path / "stub.log"), "USER": "tester", **env}
    return subprocess.run(["bash", os.path.join(ROOT, "context_arms.slurm")], env=e, capture_output=True, text=True,
                          cwd=str(tmp_path))


def trained(tmp_path):
    log = tmp_path / "stub.log"
    return log.exists() and "fine_tune_classifier.py" in log.read_text()


@needs_bash
def test_job_runs_end_to_end_and_records_its_checkpoint_policy(tmp_path):
    r = run_job(tmp_path, KEEP_CKPT="best", KEEP=str(tmp_path / "keep"))
    assert r.returncode == 0, r.stderr
    rec = tmp_path / "data" / "Toy" / "runs" / "baseline-rerun"
    job = (rec / "job.txt").read_text()
    assert "best=checkpoint-20\n" in job and "keep_ckpt=best\n" in job and (rec / ".evaluated").exists()
    run = tmp_path / "runs" / "Toy" / "baseline-rerun"
    assert sorted(p.name for p in run.glob("checkpoint-*")) == ["checkpoint-20"]
    assert (run / "checkpoint-20" / "model.safetensors").exists() and not (run / "checkpoint-20" / "optimizer.pt").exists()
    assert (tmp_path / "keep" / "Toy" / "runs" / "baseline-rerun" / "job.txt").exists()
    assert not list(rec.glob("*.lock")) and not (tmp_path / "keep" / "Toy" / "runs" / "baseline-rerun" / ".train.lock").exists()


@needs_bash
@pytest.mark.parametrize("policy, left", [("0", []), ("best", ["checkpoint-20"]), ("1", ["checkpoint-10", "checkpoint-20"])])
def test_requeue_after_the_marker_finishes_the_delete_with_the_recorded_policy(tmp_path, policy, left):
    """A job preempted between `touch .evaluated` and the delete requeues into the short-circuit, which finishes the
    delete with the KEEP_CKPT and best checkpoint the finished job recorded, not the resubmission's (default 0)."""
    assert run_job(tmp_path, KEEP_CKPT="1").returncode == 0      # finish, keeping every checkpoint
    rec = tmp_path / "data" / "Toy" / "runs" / "baseline-rerun"
    job = rec / "job.txt"
    job.write_text(job.read_text().replace("keep_ckpt=1", f"keep_ckpt={policy}"))   # as if preempted before the delete
    (tmp_path / "stub.log").unlink()
    r = run_job(tmp_path)                                         # the requeue: KEEP_CKPT unset (0)
    assert r.returncode == 0 and "already evaluated" in r.stdout, r.stderr
    assert not trained(tmp_path)
    run = tmp_path / "runs" / "Toy" / "baseline-rerun"
    assert sorted(p.name for p in run.glob("checkpoint-*")) == left
    again = run_job(tmp_path)                                     # idempotent
    assert again.returncode == 0 and sorted(p.name for p in run.glob("checkpoint-*")) == left


@needs_bash
def test_short_circuit_leaves_checkpoints_of_a_record_without_a_policy(tmp_path):
    """The pilot's code (9dace0c) wrote no keep_ckpt= and deleted before its marker: nothing to finish."""
    assert run_job(tmp_path, KEEP_CKPT="1").returncode == 0
    job = tmp_path / "data" / "Toy" / "runs" / "baseline-rerun" / "job.txt"
    job.write_text("".join(ln + "\n" for ln in job.read_text().splitlines() if not ln.startswith("keep_ckpt=")))
    assert run_job(tmp_path).returncode == 0
    assert len(list((tmp_path / "runs" / "Toy" / "baseline-rerun").glob("checkpoint-*"))) == 2


@needs_bash
def test_the_output_lock_stops_a_second_job_under_another_runs(tmp_path):
    """Two jobs with the same type and tag but different RUNS write the same predictions and record: the lock beside
    the outputs stops the second; the run-directory lock still stops one under the same RUNS and another DATA."""
    import fcntl
    data = tmp_path / "data" / "Toy"
    data.mkdir(parents=True)
    with open(data / ".train-baseline-rerun.lock", "w") as held:    # a first job, still training
        fcntl.flock(held.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        r = run_job(tmp_path, runs="other-runs")
        assert r.returncode == 1 and ".train-baseline-rerun.lock" in r.stderr and not trained(tmp_path)
        assert run_job(tmp_path, TAG="side_info", ARM="side_info").returncode == 0   # another tag is not blocked
    (tmp_path / "stub.log").unlink()
    run = tmp_path / "runs" / "Toy" / "baseline-rerun"
    run.mkdir(parents=True)
    with open(run / ".train.lock", "w") as held:
        fcntl.flock(held.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        r = run_job(tmp_path)
        assert r.returncode == 1 and "/.train.lock" in r.stderr and not trained(tmp_path)
    assert run_job(tmp_path).returncode == 0 and trained(tmp_path)   # both free again


# ---------------------------------------------------------------------------------------------------- launcher

def launcher(tmp_path, *types, preflight="0", env_python=None, jobs=None, **env):
    data = tmp_path / "data"
    for lt in types or ("NoCurbRamp",):
        (data / lt).mkdir(parents=True, exist_ok=True)
        (data / lt / "split.csv").write_text("label_uid\n")
    if not (data / "shadow_run.json").exists():
        (data / "shadow_run.json").write_text(json.dumps({"rawlabels": "/r/rawlabels"}))
    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir(exist_ok=True)
    for name, n in (jobs or {}).items():
        (jobs_dir / name).write_text("".join(f"{i}\tsmoke\tX\tt\tcmd\n" for i in range(n)))
    e = {"PATH": os.environ["PATH"], "HOME": str(tmp_path), "DATA": str(data), "RUNS": str(tmp_path / "runs"),
         "JOBS_DIR": str(jobs_dir), "DRY_RUN": "1", "PREFLIGHT": preflight, "ENV": str(tmp_path / "env"), **env}
    if env_python is not None:
        (tmp_path / "env" / "bin").mkdir(parents=True, exist_ok=True)
        py = tmp_path / "env" / "bin" / "python"
        py.write_text(env_python)
        py.chmod(py.stat().st_mode | stat.S_IEXEC)
    r = subprocess.run(["bash", os.path.join(ROOT, "context_submit.sh"), *(types or ("NoCurbRamp",))], env=e,
                       capture_output=True, text=True)
    lines = [ln[len("DRY-RUN: "):] for ln in r.stderr.splitlines() if ln.startswith("DRY-RUN: ")]
    return r, lines


def field(line, name):
    m = re.search(rf"(?:^| ){name}=(\S*)", line)
    return m.group(1) if m else None


@needs_bash
def test_smoke_preset_every_code_path_once_with_one_backfill(tmp_path):
    r, lines = launcher(tmp_path, STAGE="smoke", PANO_STORE="/abs/store")
    assert r.returncode == 0, r.stderr
    assert len(lines) == 7
    bf = [ln for ln in lines if field(ln, "STEP") == "backfill"]
    assert len(bf) == 1 and field(bf[0], "RAWLABELS") == "/r/rawlabels" and field(bf[0], "PANO_STORE") == "/abs/store"
    train = [ln for ln in lines if field(ln, "STEP") == "train"]
    tags = {field(ln, "TAG"): ln for ln in train}
    assert set(tags) == {"baseline-rerun", "two_scale", "view_heatmap", "view_heatmap-r448", "side_info",
                         "side_info_nohq"}
    assert field(tags["baseline-rerun"], "ARM") == "none" and field(tags["view_heatmap-r448"], "RES") == "448"
    for tag, ln in tags.items():
        assert field(ln, "EPOCHS") == "1"
        assert f"--job-name=context-arms-launch-smoke-NoCurbRamp-{tag} " in ln
        waits = "--dependency=afterok:dry-NoCurbRamp-backfill" in ln
        assert waits == (tag in ("two_scale", "view_heatmap", "view_heatmap-r448")), tag
    assert "--job-name=context-arms-launch-smoke-bf-NoCurbRamp " in bf[0]
    assert not list((tmp_path / "jobs").glob("context_jobs_*.txt"))   # a dry run logs nothing


@needs_bash
def test_pilot_preset_seven_jobs_per_type_no_backfill_no_dependency(tmp_path):
    types = ("NoCurbRamp", "Obstacle", "SurfaceProblem")
    r, lines = launcher(tmp_path, *types, STAGE="pilot", KEEP="/keep")
    assert r.returncode == 0, r.stderr
    assert len(lines) == 21 and all(field(ln, "STEP") == "train" for ln in lines)
    assert not any("--dependency" in ln for ln in lines) and all(field(ln, "KEEP") == "/keep" for ln in lines)
    tags = sorted(field(ln, "TAG") for ln in lines if field(ln, "LT") == "Obstacle")
    assert tags == sorted(["baseline-rerun", "baseline-rerun-s43", "baseline-rerun-s44", "baseline-rerun-s45",
                           "baseline-rerun-s46", "side_info", "side_info_nohq"])
    seeds = {field(ln, "TAG"): field(ln, "SEED") for ln in lines}
    assert seeds["baseline-rerun-s45"] == "45" and seeds["side_info_nohq"] == "42"
    assert all(field(ln, "EPOCHS") is None for ln in lines)   # the job's default, 25


@needs_bash
def test_stage_a_one_backfill_per_type_and_every_image_arm_waits_for_it(tmp_path):
    r, lines = launcher(tmp_path, "NoCurbRamp", "Obstacle", STAGE="A", PANO_STORE="/abs/store")
    assert r.returncode == 0, r.stderr
    assert len(lines) == 16
    for lt in ("NoCurbRamp", "Obstacle"):
        mine = [ln for ln in lines if field(ln, "LT") == lt]
        assert sum(field(ln, "STEP") == "backfill" for ln in mine) == 1
        train = [ln for ln in mine if field(ln, "STEP") == "train"]
        assert len(train) == 7 and all(f"--dependency=afterok:dry-{lt}-backfill" in ln for ln in train)
    r5, lines5 = launcher(tmp_path, "NoCurbRamp", STAGE="A", A5="1", PANO_STORE="/abs/store")
    assert len(lines5) == 12 and "two_scale-s46" in " ".join(lines5)


@needs_bash
def test_stage_b_needs_a_winner_and_backfill_attach(tmp_path):
    r, _ = launcher(tmp_path, "CurbRamp", STAGE="B", PANO_STORE="/abs/store")
    assert r.returncode != 0 and "WINNER" in r.stderr
    r, lines = launcher(tmp_path, "CurbRamp", STAGE="B", WINNER="view_heatmap", BACKFILL_JOB="123")
    assert r.returncode == 0, r.stderr
    # the rule's five baseline seeds on the guard types too, or a guard-type REGRESSION could never be read
    assert len(lines) == 8 and not any(field(ln, "STEP") == "backfill" for ln in lines)
    vh = [ln for ln in lines if field(ln, "ARM") == "view_heatmap"]
    assert len(vh) == 3 and all("--dependency=afterok:123" in ln for ln in vh)
    r, _ = launcher(tmp_path, "CurbRamp", "Crosswalk", STAGE="B", WINNER="view_heatmap", BACKFILL_JOB="123")
    assert r.returncode != 0 and "one label type" in r.stderr


@needs_bash
def test_runs_spec_tags_round_trip_through_seed_run_re(tmp_path):
    r, lines = launcher(tmp_path, STAGE="A", RUNS_SPEC="view_heatmap@448:43 side_info_nohq:44", SKIP_BACKFILL="1",
                        TIME="20:00:00")
    assert r.returncode == 0, r.stderr
    tags = [field(ln, "TAG") for ln in lines]
    assert tags == ["view_heatmap-r448-s43", "side_info_nohq-s44"]
    m = cl.SEED_RUN_RE.match("metrics_view_heatmap-r448-s43.json")
    assert (m["variant"], int(m["seed"])) == ("view_heatmap-r448", 43)
    assert all("--time=20:00:00" in ln for ln in lines) and not any("--dependency" in ln for ln in lines)
    for bad in ("two_scale@448:42", "baseline:42", "side_info:4x"):
        r, _ = launcher(tmp_path, STAGE="A", RUNS_SPEC=bad, SKIP_BACKFILL="1")
        assert r.returncode == 2, bad


@needs_bash
def test_job_caps_per_invocation_and_per_session(tmp_path):
    r, lines = launcher(tmp_path, "NoCurbRamp", "Obstacle", STAGE="pilot", MAX_JOBS="10")
    assert r.returncode == 2 and "MAX_JOBS" in r.stderr and not lines
    fake = tmp_path / "bin"
    fake.mkdir()
    (fake / "sbatch").write_text("#!/bin/bash\necho 1\n")
    (fake / "sbatch").chmod(0o755)
    r, lines = launcher(tmp_path, "NoCurbRamp", "Obstacle", "SurfaceProblem", STAGE="pilot", DRY_RUN="0",
                        jobs={"context_jobs_smoke.txt": 7, "context_jobs_rerun.txt": 13},
                        PATH=f"{fake}:{os.environ['PATH']}")
    assert r.returncode == 2 and "session cap" in r.stderr and "20 logged + 21 now" in r.stderr
    assert not (tmp_path / "jobs" / "context_jobs_pilot.txt").exists()
    # a dry run is never refused by the session cap (it submits nothing); it says what the real one needs
    r, lines = launcher(tmp_path, "NoCurbRamp", "Obstacle", "SurfaceProblem", STAGE="pilot",
                        jobs={"context_jobs_smoke.txt": 7, "context_jobs_rerun.txt": 13})
    assert r.returncode == 0 and len(lines) == 21 and "ALLOW_OVER_SESSION_CAP=1" in r.stdout
    r, lines = launcher(tmp_path, "NoCurbRamp", "Obstacle", "SurfaceProblem", STAGE="pilot",
                        jobs={"context_jobs_smoke.txt": 7, "context_jobs_rerun.txt": 13}, ALLOW_OVER_SESSION_CAP="1")
    assert r.returncode == 0 and len(lines) == 21


@needs_bash
def test_launcher_refuses_the_round3_directory_and_missing_pano_store(tmp_path):
    (tmp_path / "data").mkdir()
    e = {"PATH": os.environ["PATH"], "HOME": str(tmp_path), "STAGE": "pilot", "DATA": str(tmp_path / "data"),
         "RUNS": str(tmp_path / "runs"), "DRY_RUN": "1", "PREFLIGHT": "0", "JOBS_DIR": str(tmp_path)}
    r = subprocess.run(["bash", os.path.join(ROOT, "context_submit.sh"), "Toy"], env=e, capture_output=True, text=True)
    assert r.returncode == 2 and "no shadow_run.json" in r.stderr
    r, _ = launcher(tmp_path, STAGE="smoke")
    assert r.returncode == 2 and "PANO_STORE" in r.stderr
    r, _ = launcher(tmp_path, STAGE="pilot", RUNS="relative/runs")
    assert r.returncode == 2 and "absolute" in r.stderr


@needs_bash
def test_launcher_runs_preflight_with_the_env_python_and_stops_on_failure(tmp_path):
    record = tmp_path / "preflight_args"
    ok = f'#!/bin/bash\necho "$@" > {record}\nexit 0\n'
    r, lines = launcher(tmp_path, "NoCurbRamp", STAGE="smoke", PANO_STORE="/abs/store", preflight="1", env_python=ok,
                        KEEP="/keep")
    assert r.returncode == 0, r.stderr
    args = record.read_text()
    assert args.startswith("analysis/context_arms.py preflight ") and "--backfill --pano-store /abs/store" in args
    assert "--arms none,side_info,side_info_nohq,two_scale,view_heatmap" in args and "--keep /keep" in args
    assert "--pair-october" not in args   # the smoke shadow is a subset without October's files
    r, lines = launcher(tmp_path, "NoCurbRamp", STAGE="pilot", preflight="1", env_python=ok)
    assert "--pair-october" in record.read_text() and "--backfill" not in record.read_text()
    r, lines = launcher(tmp_path, "NoCurbRamp", STAGE="pilot", preflight="1", env_python="#!/bin/bash\nexit 1\n")
    assert r.returncode == 2 and "preflight failed; nothing submitted" in r.stderr and not lines


@needs_bash
def test_real_submission_logs_one_line_per_job_and_stops_at_the_first_failure(tmp_path):
    fake = tmp_path / "bin"
    fake.mkdir()
    counter = tmp_path / "n"
    counter.write_text("0")
    sb = fake / "sbatch"
    sb.write_text(f'#!/bin/bash\nn=$(( $(cat {counter}) + 1 )); echo $n > {counter}\n'
                  f'[ $n -ge 3 ] && {{ echo "socket timeout" >&2; exit 1; }}\necho $((1000 + n))\n')
    sb.chmod(sb.stat().st_mode | stat.S_IEXEC)
    r, _ = launcher(tmp_path, "NoCurbRamp", STAGE="pilot", DRY_RUN="0", PATH=f"{fake}:{os.environ['PATH']}")
    assert r.returncode != 0 and "Check squeue before resubmitting" in r.stderr
    log = (tmp_path / "jobs" / "context_jobs_pilot.txt").read_text().splitlines()
    assert [ln.split("\t")[0] for ln in log] == ["1001", "1002"]   # the failed third job is not logged, none retried
    assert counter.read_text().strip() == "3"
    assert log[0].split("\t")[1:4] == ["pilot", "NoCurbRamp", "baseline-rerun"] and "sbatch" in log[0]


@needs_bash
def test_launcher_refuses_repeated_tags_and_types(tmp_path):
    r, lines = launcher(tmp_path, STAGE="A", RUNS_SPEC="side_info:42,42 side_info:42", SKIP_BACKFILL="1")
    assert r.returncode == 2 and "listed twice: side_info" in r.stderr and not lines
    r, lines = launcher(tmp_path, "NoCurbRamp", "NoCurbRamp", STAGE="pilot")
    assert r.returncode == 2 and "passed twice: NoCurbRamp" in r.stderr and not lines


@needs_bash
def test_launcher_refuses_a_type_and_tag_already_in_squeue(tmp_path):
    fake = tmp_path / "bin"
    fake.mkdir()
    (fake / "squeue").write_text("#!/bin/bash\necho context-arms-launch-pilot-Obstacle-side_info_nohq\n")
    (fake / "squeue").chmod(0o755)
    path = f"{fake}:{os.environ['PATH']}"
    # the same type and tag under another stage name still shares the run directory
    r, lines = launcher(tmp_path, "Obstacle", STAGE="A", RUNS_SPEC="side_info_nohq:42", SKIP_BACKFILL="1", PATH=path)
    assert r.returncode == 2 and "already queued or running" in r.stderr and not lines
    r, lines = launcher(tmp_path, "NoCurbRamp", STAGE="A", RUNS_SPEC="side_info_nohq:42", SKIP_BACKFILL="1", PATH=path)
    assert r.returncode == 0 and len(lines) == 1


def preset_seeds(lines):
    """{variant: sorted seeds} of one type's training lines."""
    out = {}
    for ln in lines:
        if field(ln, "STEP") != "train":
            continue
        tag = field(ln, "TAG")
        variant = cl.SEED_RUN_RE.match(f"metrics_{tag}.json")["variant"] if "-s" in tag else tag
        out.setdefault(variant, []).append(int(field(ln, "SEED")))
    return {v: sorted(s) for v, s in out.items()}


@needs_bash
def test_every_stage_preset_trains_exactly_the_rules_seeds(tmp_path):
    """Each stage's preset satisfies the decision's completeness check (S2): the baseline at seeds_baseline wherever
    it is trained, each confirmatory arm at seeds_confirmatory; and the launcher hands both lists to the preflight.
    The rule reads five confirmatory seeds (J5, 2026-11-20), so stages A and B take A5=1."""
    from analysis import context_arms as ca
    rule = ca.DECISION_RULE
    _, pilot = launcher(tmp_path, STAGE="pilot")
    assert preset_seeds(pilot)["baseline-rerun"] == rule["seeds_baseline"]
    _, a = launcher(tmp_path, STAGE="A", A5="1", PANO_STORE="/abs/store")
    sa = preset_seeds(a)
    assert sa["two_scale"] == sa["view_heatmap"] == rule["seeds_confirmatory"]
    for winner in ca.CONFIRMATORY:
        _, b = launcher(tmp_path, "CurbRamp", STAGE="B", WINNER=winner, A5="1", PANO_STORE="/abs/store")
        sb = preset_seeds(b)
        assert sb["baseline-rerun"] == rule["seeds_baseline"] and sb[winner] == rule["seeds_confirmatory"]
    record = tmp_path / "preflight_args"
    ok = f'#!/bin/bash\necho "$@" > {record}\nexit 0\n'
    launcher(tmp_path, "CurbRamp", STAGE="B", WINNER="two_scale", A5="1", PANO_STORE="/abs/store", preflight="1",
             env_python=ok)
    args = record.read_text()
    assert "--confirmatory-seeds 42,43,44,45,46 " in args and "--baseline-seeds 42,43,44,45,46" in args
    r, _ = launcher(tmp_path, STAGE="A", A5="1", PANO_STORE="/abs/store", preflight="1", env_python=ok)
    for text in (record.read_text(), r.stdout):   # each seed once, though both arms list them
        assert re.search(r"--confirmatory-seeds 42,43,44,45,46(\s|$)", text), text
    # without A5=1 the preset trains three seeds the rule does not read: the launcher passes them, and the preflight
    # refuses them (test_preflight_refuses_a_stage_seed_list_the_rule_does_not_read)
    launcher(tmp_path, STAGE="A", PANO_STORE="/abs/store", preflight="1", env_python=ok)
    assert re.search(r"--confirmatory-seeds 42,43,44(\s|$)", record.read_text())
    # a RUNS_SPEC rerun of single jobs is not compared
    launcher(tmp_path, STAGE="A", RUNS_SPEC="two_scale:43", SKIP_BACKFILL="1", preflight="1", env_python=ok)
    assert "--confirmatory-seeds" not in record.read_text()
