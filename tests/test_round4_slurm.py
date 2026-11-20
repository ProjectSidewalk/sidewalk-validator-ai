"""The round-4 job files and launcher (issue #74): they parse, run preemptible with requeue, keep the klone rules, and
the launcher submits exactly the pilot it promises (an sbatch stub records every call's argv and environment)."""
import os
import shutil
import subprocess

import pytest

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
AN = os.path.join(ROOT, "analysis")
SLURM = ["round4_cpu.slurm", "round4_score.slurm", "confident_learning.slurm"]
SCRIPTS = SLURM + ["round4_submit.sh"]
needs_bash = pytest.mark.skipif(shutil.which("bash") is None, reason="no bash")


def read(name):
    with open(os.path.join(AN, name)) as f:
        return f.read()


@needs_bash
@pytest.mark.parametrize("name", SCRIPTS)
def test_bash_parses(name):
    r = subprocess.run(["bash", "-n", os.path.join(AN, name)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_launcher_is_executable():
    assert os.access(os.path.join(AN, "round4_submit.sh"), os.X_OK)


@pytest.mark.parametrize("name", SLURM)
def test_job_files_are_preemptible(name):
    text = read(name)
    assert "#SBATCH --partition=ckpt-all" in text and "#SBATCH --requeue" in text
    assert "#SBATCH --account=makelab" in text


@pytest.mark.parametrize("name", ["round4_score.slurm", "confident_learning.slurm"])
def test_gpu_job_files_pin_newer_gpus_and_stay_offline(name):
    text = read(name)
    assert "#SBATCH --constraint=a40|l40|l40s|a100|h200" in text
    assert "HF_HUB_OFFLINE=1" in text and "HF_HUB_DISABLE_XET=1" in text


def test_training_line_hooks():
    text = read("confident_learning.slurm")
    train = text.split("python -u fine_tune_classifier.py", 1)[1].split("||", 1)[0]
    assert "--num-workers" in train and "$EXTRA_TRAIN_ARGS" in train
    assert text.index('if [ "$KEEP_BEST" = 1 ]; then') < text.index('rm -rf "$RUN"/checkpoint-*; echo "removed')
    assert 'mv "$RUN/best.tmp" "$RUN/best"' in text
    assert '[ -f "$RUN/best/model.safetensors" ]' in text          # the early-exit guard requires best/


@pytest.mark.parametrize("name", ["round4_cpu.slurm", "round4_score.slurm"])
def test_job_files_refuse_tmp_and_makelab(name):
    assert 'case "$R4" in /tmp*|/gscratch/makelab*)' in read(name)


def test_score_job_resolves_round3_next_to_trainer_state():
    assert "round4.py round3-ckpt" in read("round4_score.slurm")
    assert "round4.py round3-ckpt" in read("round4_submit.sh")      # also in the login-node preflight


def test_launcher_text():
    text = read("round4_submit.sh")
    assert "--kill-on-invalid-dep=yes" in text and "dd if=/dev/zero" in text
    assert 'WORK="${WORK:-$SCRUB/round4-disagree-2026-11}"' in text and 'SCRUB="${SCRUB:-/gscratch/scrubbed/$USER}"' in text
    assert 'KEEP="${KEEP:-/gscratch/makelab/$USER/round4-disagree-2026-11-keep}"' in text
    assert "HF_HUB_DISABLE_XET=1 HF_HOME=$HF_HOME huggingface-cli download facebook/dinov2-large" in text
    assert '/tmp /gscratch/makelab' in text
    assert "env -u CONDA_PREFIX" in text and "( preflight ) || exit 2" in text   # Lmod refuses an inherited conda
    for line in text.splitlines():
        if "--job-name=" in line:
            assert '--job-name=round4-disagree-' in line or '--job-name="round4-disagree-' in line, line


STUB = """#!/bin/bash
n=$(( $(cat "$STUB_DIR/count" 2>/dev/null || echo 1000) + 1 )); echo $n > "$STUB_DIR/count"
{ printf 'ARGV'; for a in "$@"; do printf '\\t%s' "$a"; done; printf '\\n'
  printf 'ENV\\tCONDA_PREFIX=%s\\tLT=%s\\tVARIANT=%s\\tTAG=%s\\tSEED=%s\\tSTEP=%s\\tMODE=%s\\tKEEP_BEST=%s\\tKEEP_CKPT=%s\\tDATA=%s\\n' \
    "${CONDA_PREFIX:-}" "${LT:-}" "${VARIANT:-}" "${TAG:-}" "${SEED:-}" "${STEP:-}" "${MODE:-}" "${KEEP_BEST:-}" "${KEEP_CKPT:-}" "${DATA:-}"
} >> "$STUB_DIR/calls"
echo $n
"""


@pytest.fixture
def stub(tmp_path):
    bin_dir, cwd = tmp_path / "bin", tmp_path / "cwd"
    bin_dir.mkdir()
    cwd.mkdir()
    (bin_dir / "sbatch").write_text(STUB)
    (bin_dir / "sbatch").chmod(0o755)
    os.symlink(AN, cwd / "analysis")

    def run(stage, **env):
        e = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "STUB_DIR": str(tmp_path), "STAGE": stage,
             "PREFLIGHT": "0", "WORK": str(tmp_path / "work"), "ROUND4_FORBIDDEN_PREFIXES": "/nonexistent-prefix",
             "USER": "tester", "CONDA_PREFIX": "/inherited/env", **env}
        return subprocess.run(["bash", os.path.join(AN, "round4_submit.sh")], cwd=cwd, env=e, capture_output=True,
                              text=True)

    def calls():
        p = tmp_path / "calls"
        if not p.exists():
            return []
        lines = p.read_text().splitlines()
        out = []
        for a, b in zip(lines[0::2], lines[1::2]):
            argv = a.split("\t")[1:]
            env = dict(x.split("=", 1) for x in b.split("\t")[1:])
            out.append({"argv": argv, "env": env, "file": argv[-1]})
        return out

    return run, calls, tmp_path


def opt(call, name):
    for a in call["argv"]:
        if a.startswith(f"--{name}="):
            return a.split("=", 1)[1]
    return None


@needs_bash
def test_pilot_submits_29_gpu_and_2_cpu_jobs(stub):
    run, calls, tmp = stub
    r = run("pilot")
    assert r.returncode == 0, r.stderr
    assert "preflight skipped" in r.stdout
    c = calls()
    gpu = [x for x in c if x["file"].endswith(("confident_learning.slurm", "round4_score.slurm"))]
    cpu = [x for x in c if x["file"].endswith("round4_cpu.slurm")]
    assert len(gpu) == 29 and len(cpu) == 2
    prep_id = "1001"
    assert cpu[0]["env"]["STEP"] == "prep"
    trains = [x for x in c if x["file"].endswith("confident_learning.slurm")]
    assert len(trains) == 27
    ids = [str(1002 + i) for i in range(27)]
    for i, t in enumerate(trains):
        assert t["argv"][0:2] == ["--parsable", "--kill-on-invalid-dep=yes"]
        assert opt(t, "job-name") == f"round4-disagree-{t['env']['LT']}-{t['env']['TAG']}"   # overrides sva-cl
        dep = opt(t, "dependency").split(",")
        assert dep[0] == f"afterok:{prep_id}"
        assert dep[1:] == ([] if i < 16 else [f"afterany:{ids[i - 16]}"])    # only the throttle besides prep
        assert t["env"]["KEEP_BEST"] == "1" and t["env"]["KEEP_CKPT"] == "0"
        assert t["env"]["CONDA_PREFIX"] == ""                                   # not inherited by the job
    assert [t["env"]["SEED"] for t in trains[:9]] == ["42"] * 9                 # seed 42 first
    assert {t["env"]["TAG"] for t in trains} == {"baseline", "baseline-s43", "baseline-s44", "cl-drop",
                                                 "cl-drop-s43", "cl-drop-s44"}
    assert not any(t["env"]["LT"] == "SurfaceProblem" and t["env"]["VARIANT"] == "cl-drop" for t in trains)
    jobs = (tmp / "work" / "round4_jobs.txt").read_text().splitlines()
    assert sum(1 for ln in jobs if ln.split("\t")[1] == "gpu") == 29
    assert all("sha256=" in ln for ln in jobs)
    # a second pilot would pass the cap of 40
    r2 = run("pilot")
    assert r2.returncode != 0 and "MAX_GPU_JOBS" in r2.stderr
    assert len(calls()) == len(c)


@needs_bash
def test_smoke_submits_one_cpu_and_two_gpu_jobs(stub):
    run, calls, _ = stub
    r = run("smoke")
    assert r.returncode == 0, r.stderr
    c = calls()
    assert [x["file"].rsplit("/", 1)[-1] for x in c] == ["round4_cpu.slurm", "confident_learning.slurm",
                                                         "round4_score.slurm"]
    assert c[0]["env"]["STEP"] == "prep"                                         # A4: prep through the job file
    assert opt(c[1], "dependency") == "afterok:1001" and opt(c[2], "dependency") == "afterok:1002"
    assert opt(c[1], "job-name") == "round4-disagree-smoke-train"
    assert c[2]["env"]["MODE"] == "round3,runs"


@needs_bash
def test_stage2_refuses_without_confirmation(stub):
    run, calls, _ = stub
    r = run("stage2")
    assert r.returncode != 0 and "CONFIRM_STAGE2=1" in r.stderr and not calls()


@needs_bash
def test_launcher_refuses_work_under_tmp_by_default(stub):
    run, calls, _ = stub
    r = run("pilot", WORK="/tmp/round4-test-work", ROUND4_FORBIDDEN_PREFIXES="")   # empty: the default list
    assert r.returncode != 0 and "under /tmp" in r.stderr and not calls()
