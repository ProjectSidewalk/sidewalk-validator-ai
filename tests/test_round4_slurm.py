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
    assert "-u CONDA_PREFIX" in text and "( preflight ) || exit 2" in text       # Lmod refuses an inherited conda
    for line in text.splitlines():
        if "--job-name=" in line:
            assert '--job-name=round4-disagree-' in line or '--job-name="round4-disagree-' in line, line


STUB_VARS = ["CONDA_PREFIX", "LT", "VARIANT", "TAG", "SEED", "STEP", "MODE", "KEEP_BEST", "KEEP_CKPT", "DATA", "TAGS",
             "TYPES", "CL_TYPES", "CROPS", "EXPECT", "GOLD_EXPECT", "DEST", "R4", "RUNS"]
STUB = """#!/bin/bash
n=$(( $(cat "$STUB_DIR/count" 2>/dev/null || echo 1000) + 1 )); echo $n > "$STUB_DIR/count"
{ printf 'ARGV'; for a in "$@"; do printf '\\t%s' "$a"; done; printf '\\n'
  printf 'ENV'; for v in """ + " ".join(STUB_VARS) + """; do printf '\\t%s=%s' "$v" "${!v:-}"; done; printf '\\n'
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
    # stage 2 cuts the crop list on the login node (`timeout 600 python analysis/adjudicator.py crops`): a no-op here
    (bin_dir / "timeout").write_text("#!/bin/bash\necho \"timeout stub: $*\"\n")
    (bin_dir / "timeout").chmod(0o755)
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

    return run, calls, tmp_path, bin_dir


def opt(call, name):
    for a in call["argv"]:
        if a.startswith(f"--{name}="):
            return a.split("=", 1)[1]
    return None


@needs_bash
def test_pilot_submits_29_gpu_and_2_cpu_jobs(stub):
    run, calls, tmp, _ = stub
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
    run, calls, _, _ = stub
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
    run, calls, _, _ = stub
    r = run("stage2")
    assert r.returncode != 0 and "CONFIRM_STAGE2=1" in r.stderr and not calls()


@needs_bash
def test_launcher_refuses_work_under_tmp_by_default(stub):
    run, calls, _, _ = stub
    r = run("pilot", WORK="/tmp/round4-test-work", ROUND4_FORBIDDEN_PREFIXES="")   # empty: the default list
    assert r.returncode != 0 and "under /tmp" in r.stderr and not calls()


# ---------------------------------------------------------------------------------------------------------------------
# Review round 1 (S2, S5, S9): what each submitted job is given, stage 2, a replayed log line, the CPU job file
# ---------------------------------------------------------------------------------------------------------------------

PILOT_TAGS = "baseline,baseline-s43,baseline-s44,cl-drop,cl-drop-s43,cl-drop-s44"
CL4 = "Crosswalk,CurbRamp,NoCurbRamp,Obstacle"


@needs_bash
def test_pilot_wires_data_tags_expect_and_dependencies(stub):
    run, calls, tmp, _ = stub
    assert run("pilot").returncode == 0
    c = calls()
    r4 = str(tmp / "work" / "data")
    trains = [x for x in c if x["file"].endswith("confident_learning.slurm")]
    assert {t["env"]["DATA"] for t in trains} == {r4}                 # the gold-free mirror, never round 3's DATA
    ids = [str(1002 + i) for i in range(27)]
    r3, sr = [x for x in c if x["file"].endswith("round4_score.slurm")]
    col = [x for x in c if x["file"].endswith("round4_cpu.slurm")][1]
    assert r3["env"]["MODE"] == "round3" and opt(r3, "dependency") == "afterok:1001"
    assert sr["env"]["MODE"] == "runs" and sr["env"]["TAGS"] == PILOT_TAGS and sr["env"]["CROPS"] == "r3crops"
    assert opt(sr, "dependency") == "afterany:" + ":".join(ids)
    assert col["env"]["STEP"] == "collect" and opt(col, "dependency") == "afterany:" + ":".join(ids + ["1029", "1030"])
    assert col["env"]["EXPECT"] == f"baseline:42,43,44+cl-drop:42,43,44:{CL4}"


@needs_bash
def test_a_logged_line_replays_without_the_inherited_conda(stub):
    run, calls, tmp, bin_dir = stub
    assert run("smoke").returncode == 0
    first = calls()[1]                                                 # the smoke training
    line = (tmp / "work" / "round4_jobs.txt").read_text().splitlines()[1]
    cmd = line.split("\t", 4)[4]
    assert cmd.startswith("env -u CONDA_PREFIX")
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "STUB_DIR": str(tmp),
           "CONDA_PREFIX": "/inherited/env"}
    r = subprocess.run(["bash", "-c", cmd], env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    replay = calls()[-1]
    assert replay["env"]["CONDA_PREFIX"] == "" and replay["argv"] == first["argv"] and replay["env"] == first["env"]


@needs_bash
def test_stage2_submits_scoring_of_every_arm_and_a_collect(stub):
    run, calls, tmp, _ = stub
    r = run("stage2", CONFIRM_STAGE2="1", MAX_GPU_JOBS="100")
    assert r.returncode == 0, r.stderr
    c = calls()
    gpu_files = ("confident_learning.slurm", "round4_score.slurm", "adjudicator.slurm")
    gpu = [x for x in c if x["file"].endswith(gpu_files)]
    assert len(gpu) == 19
    jobs = (tmp / "work" / "round4_jobs.txt").read_text().splitlines()
    assert sum(1 for ln in jobs if ln.split("\t")[1] == "gpu") == 19
    by_name = {opt(x, "job-name"): x for x in c}
    pull = by_name["round4-disagree-pano-pull"]
    assert "--requeue" in pull["argv"] and pull["env"]["DEST"] == "/gscratch/scrubbed/tester/pano-stage/round4-disagree"
    su = by_name["round4-disagree-score-uniform"]
    assert su["env"]["CROPS"] == "uniform" and su["env"]["TAGS"] == PILOT_TAGS and su["env"]["MODE"] == "round3,runs"
    # 32 checkpoints at up to 10.9 min each is 349 min, so the uniform scoring needs 6 h; the 12 rb checkpoints
    # fit the job file's 4 h default
    assert opt(su, "time") == "6:00:00"
    rb = [x for x in c if x["file"].endswith("confident_learning.slurm")]
    assert len(rb) == 12 and {x["env"]["DATA"] for x in rb} == {str(tmp / "work" / "data")}
    assert len({opt(x, "job-name") for x in rb}) == 12
    srb = by_name["round4-disagree-score-rb"]
    assert opt(srb, "time") is None
    assert srb["env"]["CROPS"] == "uniform" and srb["env"]["TYPES"] == "CurbRamp,Crosswalk"
    assert set(srb["env"]["TAGS"].split(",")) == {"symmetric-rb", "matched-rb", "symmetric-rb-s43", "matched-rb-s43",
                                                  "symmetric-rb-s44", "matched-rb-s44"}
    dep = opt(srb, "dependency")
    man_id = c.index(by_name["round4-disagree-gold-manifest"]) + 1001
    assert dep.startswith(f"afterok:{man_id},afterany:") and len(dep.split("afterany:")[1].split(":")) == 12
    col = by_name["round4-disagree-collect-stage2"]
    assert "symmetric-rb:42,43,44:CurbRamp,Crosswalk" in col["env"]["EXPECT"]
    assert col["env"]["EXPECT"].startswith(f"baseline:42,43,44+cl-drop:42,43,44:{CL4}")
    assert "uniform:" in col["env"]["GOLD_EXPECT"] and "matched-rb" in col["env"]["GOLD_EXPECT"]
    assert opt(col, "dependency").count(":") == 2 + 12
    # the default cap of 40 cannot take stage 2 after the pilot's 34 GPU jobs (Jon's call to raise it)
    log = tmp / "work" / "round4_jobs.txt"
    log.write_text("\n".join(f"{i}\tgpu\tt\tsha256=x\tcmd" for i in range(34)) + "\n")
    r2 = run("stage2", CONFIRM_STAGE2="1")
    assert r2.returncode != 0 and "needs 19 GPU jobs; 34 already" in r2.stderr


@needs_bash
def test_cpu_job_file_runs_collect(tmp_path):
    """round4_cpu.slurm STEP=collect end to end, with `module` / `conda` stubbed and a tiny run layout. The work
    directory sits under the repository, not under /tmp, which the job file refuses (as on klone)."""
    import tempfile
    import sys
    work = tempfile.mkdtemp(prefix=".pytest-round4-cpu-", dir=ROOT)
    try:
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        for name, body in (("module", "exit 0"), ("conda", "exit 0"),
                           ("python", f'[ "$1" = -c ] && exit 0\nexec "{sys.executable}" "$@"')):
            (bin_dir / name).write_text(f"#!/bin/bash\n{body}\n")
            (bin_dir / name).chmod(0o755)
        r4d, runs = os.path.join(work, "r4"), os.path.join(work, "runs")
        os.makedirs(os.path.join(r4d, "Obstacle"))
        with open(os.path.join(r4d, "Obstacle", "metrics_baseline.json"), "w") as f:
            f.write("{}")
        with open(os.path.join(r4d, "Obstacle", "predictions_baseline.csv"), "w") as f:
            f.write("label_uid\nx\n")
        best = os.path.join(runs, "Obstacle", "baseline", "best")
        os.makedirs(best)
        with open(os.path.join(best, "model.safetensors"), "wb") as f:
            f.write(b"w")
        env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "STEP": "collect", "DATA": work,
               "R4": r4d, "GOLD_UIDS": os.path.join(work, "none.csv"), "RUNS": runs,
               "KEEP": os.path.join(work, "keep"), "EXPECT": "baseline:42", "TYPES": "Obstacle"}
        r = subprocess.run(["bash", os.path.join(AN, "round4_cpu.slurm")], cwd=ROOT, env=env, capture_output=True,
                           text=True)
        assert r.returncode == 0, r.stdout + r.stderr
        assert "done collect" in r.stdout
        assert os.path.exists(os.path.join(r4d, "round4_manifest.json"))
        assert os.path.exists(os.path.join(work, "keep", "runs", "Obstacle", "baseline")) is False   # no run files
        assert os.path.exists(os.path.join(r4d, "Obstacle", "metrics.json"))                       # untagged copy
    finally:
        shutil.rmtree(work, ignore_errors=True)
