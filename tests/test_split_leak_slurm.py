"""The split-leak job files (issue #3): they parse, run on the preemptible partition with requeue, train on the
arm's own split into the run directory, never write under the round-3 data directory, delete the weights, and the
launcher throttles the GPU array and prints everything under DRY_RUN."""
import os
import shutil
import subprocess

import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
GPU, CPU, SUBMIT = "analysis/split_leak.slurm", "analysis/split_leak_cpu.slurm", "analysis/split_leak_submit.sh"


def read(name):
    with open(os.path.join(ROOT, name)) as f:
        return f.read()


@pytest.mark.skipif(shutil.which("bash") is None, reason="no bash")
@pytest.mark.parametrize("name", [GPU, CPU, SUBMIT])
def test_bash_parses(name):
    r = subprocess.run(["bash", "-n", os.path.join(ROOT, name)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


@pytest.mark.parametrize("name", [GPU, CPU])
def test_preemptible_with_requeue(name):
    text = read(name)
    assert "#SBATCH --partition=ckpt-all" in text and "#SBATCH --requeue" in text
    assert "#SBATCH --account=makelab" in text


def test_gpu_job_resources_and_training_command():
    text = read(GPU)
    for line in ("#SBATCH --gpus=1", "#SBATCH --cpus-per-task=8", "#SBATCH --mem=64G", "#SBATCH --time=24:00:00"):
        assert line in text, line
    train = text.split("python -u fine_tune_classifier.py")[1].split("||")[0]
    assert '--split "$SPLIT"' in train and '--output-dir "$RUN"' in train and "--target hard" in train
    assert "--epochs" in train and '--seed "$SEED"' in train
    assert "HF_HUB_OFFLINE=1" in text


def test_gpu_job_writes_only_into_its_run_directory_and_deletes_the_weights():
    text = read(GPU)
    assert 'RUN="$RUNS/$LT/$ARM-s$SEED"' in text
    assert "split_leak.py predict" in text and '--out "$RUN/predictions.csv"' in text
    assert 'rm -rf "$RUN"/checkpoint-*' in text
    assert "evaluate.py" not in text                   # evaluate.py writes predictions_<tag>.csv under $DATA/$LT
    for line in text.splitlines():
        if line.lstrip().startswith("#"):
            continue
        for op in (">", "cp ", "mv ", "mkdir", "rm "):
            if op in line and "$DATA" in line.split(op, 1)[1]:
                pytest.fail(f"writes under $DATA: {line}")
    assert "already scored" in text                      # a requeue after scoring does nothing


def test_launcher_throttles_and_kills_invalid_dependents():
    text = read(SUBMIT)
    assert '%${MAX_PARALLEL:-4}' in text and "--kill-on-invalid-dep=yes" in text
    assert "split_leak_jobs.txt" in text and 'DRY_RUN="${DRY_RUN:-0}"' in text
    assert os.access(os.path.join(ROOT, SUBMIT), os.X_OK)


def _dry(tmp_path, **env):
    full = {**os.environ, "DRY_RUN": "1", "DATA": "/d", "RUNS": str(tmp_path / "runs"), "RAWLABELS": "/raw", **env}
    return subprocess.run(["bash", SUBMIT], cwd=ROOT, env=full, capture_output=True, text=True)


def test_dry_run_stage0_is_one_cpu_job(tmp_path):
    r = _dry(tmp_path, STAGE="0")
    assert r.returncode == 0, r.stderr
    sb = [line for line in r.stdout.splitlines() if line.startswith("sbatch")]
    assert len(sb) == 1 and "STEP=stage0" in sb[0] and CPU in sb[0]
    assert not (tmp_path / "runs").exists()               # a dry run creates nothing


def test_dry_run_stage1_refuses_without_stage0(tmp_path):
    r = _dry(tmp_path, STAGE="1")
    assert r.returncode == 2 and "Stage 0" in r.stderr


def _stage0_outputs(tmp_path, seeds=(42, 43, 44)):
    runs = tmp_path / "runs"
    for t in ("CurbRamp", "NoCurbRamp", "Obstacle", "SurfaceProblem", "Crosswalk"):
        d = runs / t / "splits"
        d.mkdir(parents=True)
        for s in seeds:
            (d / f"label_level_s{s}.csv").write_text("x\n")
            (d / f"exposure_s{s}.csv").write_text("x\n")
    (runs / "report").mkdir()
    (runs / "report" / "design.json").write_text("{}")
    (runs / "report" / "bound.json").write_text("{}")


def test_dry_run_stage1_and_stage2_arrays(tmp_path):
    _stage0_outputs(tmp_path)
    r = _dry(tmp_path, STAGE="1")
    assert r.returncode == 0, r.stderr
    sb = [line for line in r.stdout.splitlines() if line.startswith("sbatch")]
    assert len(sb) == 2
    assert "--array=0-4%4" in sb[0] and GPU in sb[0]
    assert "--dependency=afterany:dry1" in sb[1] and "STEP=report" in sb[1] and "SEEDS=42," in sb[1]
    assert "G_SOURCE=stored" in sb[1] and "G_SEEDS=42:43:44" in sb[1]
    tasks = [line for line in r.stdout.splitlines() if line.startswith("task ")]
    assert tasks == [f"task {i}: {t} L 42" for i, t in enumerate(
        ("CurbRamp", "NoCurbRamp", "Obstacle", "SurfaceProblem", "Crosswalk"))]
    r = _dry(tmp_path, STAGE="2", MAX_PARALLEL="2", FRESH_G="1")
    assert r.returncode == 2 and "G-s42/predictions.csv" in r.stderr      # no fresh grouped seed 42 yet
    for t in ("CurbRamp", "NoCurbRamp", "Obstacle", "SurfaceProblem", "Crosswalk"):
        (tmp_path / "runs" / t / "G-s42").mkdir()
        (tmp_path / "runs" / t / "G-s42" / "predictions.csv").write_text("x\n")
    r = _dry(tmp_path, STAGE="2", MAX_PARALLEL="2", FRESH_G="1")
    assert r.returncode == 0, r.stderr
    sb = [line for line in r.stdout.splitlines() if line.startswith("sbatch")]
    assert "--array=0-19%2" in sb[0]                       # 5 types x (L, G) x seeds 43, 44
    assert "SEEDS=42:43:44" in sb[1] and "G_SEEDS=42:43:44" in sb[1] and "G_SOURCE=fresh" in sb[1]
    tasks = [line for line in r.stdout.splitlines() if line.startswith("task ")]
    assert tasks[0] == "task 0: CurbRamp L 43" and tasks[1] == "task 1: CurbRamp G 43"   # interleaved


def test_dry_run_stage1_refuses_without_design_or_bound(tmp_path):
    """The per-split files alone are not Stage 0: design.json and bound.json gate the GPU stages too."""
    _stage0_outputs(tmp_path)
    for name in ("design.json", "bound.json"):
        f = tmp_path / "runs" / "report" / name
        f.unlink()
        r = _dry(tmp_path, STAGE="1")
        assert r.returncode == 2 and name in r.stderr, name
        f.write_text("{}")
    assert _dry(tmp_path, STAGE="1").returncode == 0


def _gpu_task(tmp_path, run_files, checkpoints):
    """Run split_leak.slurm for one L task with stub python/conda/module/nvidia-smi on PATH; python logs its
    arguments and fails on predict and training (a stand-in for a best checkpoint that is already gone)."""
    import time
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    log = tmp_path / "python.log"
    (bin_ / "python").write_text(f'#!/bin/bash\necho "$*" >> {log}\n'
                                 'case "$*" in *predict*|*fine_tune*) exit 1;; esac\nexit 0\n')
    for name in ("conda", "module", "nvidia-smi"):
        (bin_ / name).write_text("#!/bin/bash\nexit 0\n")
    for f in bin_.iterdir():
        f.chmod(0o755)
    runs = tmp_path / "runs"
    (runs / "CurbRamp" / "splits").mkdir(parents=True)
    (runs / "CurbRamp" / "splits" / "label_level_s42.csv").write_text("x\n")
    run = runs / "CurbRamp" / "L-s42"
    run.mkdir()
    for name in run_files:                     # in order, each newer than the last
        (run / name).write_text("x\n")
        os.utime(run / name, (time.time() + run_files.index(name), time.time() + run_files.index(name)))
    for c in checkpoints:
        (run / c).mkdir()
        (run / c / "trainer_state.json").write_text("{}")
    (tmp_path / "tasks.txt").write_text("CurbRamp L 42\n")
    env = {**os.environ, "PATH": f"{bin_}:{os.environ['PATH']}", "DATA": str(tmp_path / "data"),
           "RUNS": str(runs), "TASKS": str(tmp_path / "tasks.txt"), "SLURM_ARRAY_TASK_ID": "0",
           "ENV": str(tmp_path / "env")}
    r = subprocess.run(["bash", os.path.join(ROOT, GPU)], cwd=ROOT, env=env, capture_output=True, text=True)
    return r, run, (log.read_text() if log.exists() else "")


@pytest.mark.skipif(shutil.which("bash") is None, reason="no bash")
def test_a_requeue_during_weight_deletion_only_finishes_the_deletion(tmp_path):
    """Review round 1, S1: predictions.csv newer than training_run.json with a checkpoint left means preempted
    inside the deletion. The task must delete the rest and succeed, never train or score again."""
    r, run, log = _gpu_task(tmp_path, ["training_run.json", "trainer_state.best.json", "predictions.csv"],
                            ["checkpoint-1500"])
    assert r.returncode == 0, r.stdout + r.stderr
    assert not list(run.glob("checkpoint-*")) and (run / "predictions.csv").exists()
    assert "predict" not in log and "fine_tune" not in log
    assert "finishing the deletion" in r.stdout


@pytest.mark.skipif(shutil.which("bash") is None, reason="no bash")
def test_a_requeue_after_training_still_scores(tmp_path):
    """Trained, not scored (no predictions.csv): the task scores rather than trains (here predict fails)."""
    r, run, log = _gpu_task(tmp_path, ["training_run.json"], ["checkpoint-1000"])
    assert r.returncode == 1 and "predict failed" in r.stderr
    assert "predict" in log and "fine_tune" not in log
    assert list(run.glob("checkpoint-*"))                  # nothing deleted before scoring succeeds
