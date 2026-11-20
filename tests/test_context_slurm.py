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
    assert train.index("evaluate.py") < record < train.index('rm -rf "$RUN"/checkpoint-*')
    assert record < train.index('rm -f "$BEST/optimizer.pt"')
    assert '"$BEST/trainer_state.json" "$REC/trainer_state.json"' in train and '"$REC/config.json"' in train
    assert train.index("touch \"$RUN/.evaluated\"") > train.index('cp -r "$REC" "$KEEP/$LT/runs/"')


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
    assert len(lines) == 6 and not any(field(ln, "STEP") == "backfill" for ln in lines)
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
    r, lines = launcher(tmp_path, "NoCurbRamp", "Obstacle", "SurfaceProblem", STAGE="pilot",
                        jobs={"context_jobs_smoke.txt": 7, "context_jobs_rerun.txt": 13})
    assert r.returncode == 2 and "session cap" in r.stderr and "20 logged + 21 now" in r.stderr and not lines
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
