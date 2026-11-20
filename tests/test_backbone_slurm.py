"""The backbone comparison's klone files (issue #119): they parse; the job is preemptible, requeues and runs offline;
run_one refuses an empty tag, keeps the records and the best weights before deleting checkpoints, and stops at its
done guard; the launcher writes the expected task rows, skips unavailable arms, refuses a smoke tag outside the smoke
job and the 41st GPU job. Everything runs against stubs (sbatch, hyakstorage, df, curl and a fake python)."""
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
SCRIPTS = ["backbone.slurm", "backbone_submit.sh", "backbone_env.sh", "backbone_prefetch.sh"]
pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="no bash")


def read(name):
    with open(os.path.join(ANALYSIS, name)) as f:
        return f.read()


@pytest.mark.parametrize("name", SCRIPTS)
def test_bash_parses(name):
    r = subprocess.run(["bash", "-n", os.path.join(ANALYSIS, name)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_slurm_header_and_offline():
    text = read("backbone.slurm")
    for line in ("#SBATCH --partition=ckpt-all", "#SBATCH --requeue", "#SBATCH --account=makelab",
                 "#SBATCH --constraint=a40|l40|l40s|a100|h200", "#SBATCH --job-name=backbone-comparison"):
        assert line in text
    assert "HF_HUB_OFFLINE=1" in text and "SVA_BACKBONE_DIR" in text
    assert "HF_HUB_DISABLE_XET=1" in read("backbone_prefetch.sh")
    submit = read("backbone_submit.sh")
    names = re.findall(r'--job-name="([^"]+)"', submit)
    assert names and all(n.startswith("$SLUG-") for n in names) and "SLUG=backbone-comparison" in submit
    assert "--time=02:00:00" in submit.split("  smoke)")[1].split(";;")[0]   # A5


@pytest.mark.parametrize("name", ["backbone.slurm", "backbone_submit.sh"])
def test_every_required_variable_is_documented_in_the_header(name):
    text = read(name)
    header = "\n".join(ln for ln in text.splitlines() if ln.startswith("#"))
    for var in set(re.findall(r"\$\{([A-Za-z_]\w*):\?", text)) - {"SLURM_ARRAY_TASK_ID", "SEEDS", "TYPES", "ARMS"}:
        assert re.search(rf"\b{var}\b", header), f"{name}: {var} is required but not documented"


def test_env_overlay_keeps_the_base_env():
    text = read("backbone_env.sh")
    assert "--system-site-packages" in text and "pip list --format=freeze" in text
    assert "--no-deps" in text and "import torchvision" in text
    assert '"torch": ("2.7.1", torch, base)' in text and '"pandas": ("2.3.1", pandas, base)' in text
    assert '"transformers": ("4.56.2", transformers, venv)' in text
    assert re.search(r'\$BASE_ENV/bin/pip"? install', text) is None   # never installs into the base env
    req = open(os.path.join(REPO, "requirements-backbone.txt")).read()
    assert "transformers==4.56.2" in req and "timm==" in req and "tokenizers==0.22.2" in req
    # a venv that scrubbed's purge has partly deleted must fail the idempotence check (review nit 12)
    assert '"tokenizers": ("0.22.2", tokenizers, venv)' in text and '"timm": ("1.0.31", timm, venv)' in text


# ---------------------------------------------------------------------------------------------------------------------
# run_one
# ---------------------------------------------------------------------------------------------------------------------

FAKE_PY = r"""#!/bin/bash
# A stand-in for python in run_one: logs each call, and writes what the real script would.
echo "$* RANDOM_INIT=${SVA_BACKBONE_RANDOM_INIT:-0}" >> "$CALLS"
if [ "$1" = -u ] && [ "$2" = fine_tune_classifier.py ]; then
  out=$(echo "$*" | sed -E 's/.*--output-dir ([^ ]+).*/\1/')
  mkdir -p "$out/checkpoint-10" "$out/checkpoint-20"
  for f in model.safetensors config.json preprocessor_config.json trainer_state.json; do
    echo "$f" > "$out/checkpoint-20/$f"; echo "$f" > "$out/checkpoint-10/$f"
  done
  [ -z "${DROP:-}" ] || rm "$out/checkpoint-20/$DROP"
  echo '{}' > "$out/training_run.json"; echo '{}' > "$out/run_key.json"
elif [ "$1" = -c ]; then
  echo "$3/checkpoint-20"
elif [ "$1" = -u ] && [ "$2" = evaluate.py ]; then
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
    (tmp_path / "data" / "CurbRamp").mkdir(parents=True)
    env = {**os.environ, "PY": str(py), "CALLS": str(tmp_path / "calls.log"), "DATA": str(tmp_path / "data"),
           "RUNS": str(tmp_path / "runs"), "KEEP": str(tmp_path / "keep"), "PATH": "/usr/bin:/bin"}

    def call(*args, **extra):
        cmd = f'source "{os.path.join(ANALYSIS, "backbone.slurm")}"; run_one ' + " ".join(f'"{a}"' for a in args)
        return subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, cwd=REPO, env={**env, **extra})
    return call, tmp_path


def calls(tmp_path):
    p = tmp_path / "calls.log"
    return p.read_text().splitlines() if p.exists() else []


def test_run_one_refuses_an_empty_tag(job):
    call, tmp = job
    r = call("CurbRamp", "dinov2-large", "42", "", "1e-5")
    assert r.returncode != 0 and "refusing" in r.stderr
    assert calls(tmp) == []   # nothing ran, evaluate.py least of all


def test_run_one_keeps_records_and_best_then_deletes_checkpoints_and_stops_at_done(job):
    call, tmp = job
    r = call("CurbRamp", "convnextv2-base", "43", "bb-convnextv2-base-s43", "1e-5")
    assert r.returncode == 0, r.stderr
    run, keep = tmp / "runs" / "CurbRamp" / "bb-convnextv2-base-s43", tmp / "keep" / "CurbRamp" / "runs" / "bb-convnextv2-base-s43"
    assert sorted(os.listdir(run / "best")) == ["config.json", "model.safetensors", "preprocessor_config.json",
                                                "trainer_state.json"]
    assert not list(run.glob("checkpoint-*"))
    assert {"metrics_bb-convnextv2-base-s43.json", "predictions_bb-convnextv2-base-s43.csv", "training_run.json",
            "run_key.json", "trainer_state.json", "done"} <= set(os.listdir(keep))
    assert not list(keep.glob("*.safetensors"))
    c = calls(tmp)
    assert [ln.split()[1] for ln in c if ln.startswith("-u")] == ["fine_tune_classifier.py", "evaluate.py"]
    assert "--backbone convnextv2-base --seed 43 --epochs 25 --learning-rate 1e-5" in c[0]
    assert "--checkpoint" in c[-1] and "--tag bb-convnextv2-base-s43 --per-city" in c[-1]
    assert all("RANDOM_INIT=0" in ln for ln in c)
    r2 = call("CurbRamp", "convnextv2-base", "43", "bb-convnextv2-base-s43", "1e-5")
    assert r2.returncode == 0 and "done already" in r2.stdout and calls(tmp) == c


def test_a_failed_copy_leaves_every_checkpoint(job):
    call, tmp = job
    r = call("CurbRamp", "dinov2-large", "42", "bb-dinov2-large", "1e-5", DROP="preprocessor_config.json")
    assert r.returncode != 0
    run = tmp / "runs" / "CurbRamp" / "bb-dinov2-large"
    assert len(list(run.glob("checkpoint-*"))) == 2 and not (tmp / "keep" / "CurbRamp" / "runs" / "bb-dinov2-large" / "done").exists()
    text = read("backbone.slurm")
    body = text[text.index("run_one() {"):text.index("task_line() {")]
    # The first rm only runs once best_copied exists (the requeue path); the main one follows the checked copies.
    assert body.index('[ -f "$RUN/best_copied" ]') < body.index('rm -rf "$RUN"/checkpoint-*')
    assert body.index('cp "$DATA/$LT/$f" "$K/"') < body.rindex('rm -rf "$RUN"/checkpoint-*')
    assert body.index('cp "$BEST/$f" "$RUN/best/$f"') < body.rindex('touch "$RUN/best_copied"') \
        < body.rindex('rm -rf "$RUN"/checkpoint-*')


def test_the_task_row_learning_rate_reaches_the_training_command(job):
    call, tmp = job
    assert call("CurbRamp", "convnextv2-base", "42", "bb-convnextv2-base-lr3e5", "3e-5").returncode == 0
    c = calls(tmp)
    assert "--backbone convnextv2-base --seed 42 --epochs 25 --learning-rate 3e-5 --num-workers 7" in c[0]
    assert "--output-dir " + str(tmp / "runs" / "CurbRamp" / "bb-convnextv2-base-lr3e5") in c[0]


def test_random_init_only_for_smoke_rand_tags(job):
    call, tmp = job
    assert call("CurbRamp", "dinov3-vitl16", "42", "smoke-rand-dinov3-vitl16", "1e-5").returncode == 0
    c = calls(tmp)
    assert "RANDOM_INIT=1" in c[0] and c[0].startswith("-u fine_tune_classifier.py")
    assert all("RANDOM_INIT=0" in ln for ln in c[1:])   # evaluate.py loads the trained weights


def test_task_line_skips_comments_and_selects_by_index(tmp_path):
    tasks = tmp_path / "t.tsv"
    tasks.write_text("# header\nCurbRamp\ta\t42\tbb-a\t1e-5\nSurfaceProblem\tb\t43\tbb-b-s43\t1e-5\n")
    cmd = f'source "{os.path.join(ANALYSIS, "backbone.slurm")}"; TASKS="{tasks}"; task_line 2'
    r = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True)
    assert r.stdout.strip().split("\t") == ["SurfaceProblem", "b", "43", "bb-b-s43", "1e-5"]


# ---------------------------------------------------------------------------------------------------------------------
# the launcher
# ---------------------------------------------------------------------------------------------------------------------

STUBS = {
    "sbatch": '#!/bin/bash\necho "$* STEP=$STEP TASKS=$TASKS KEEP=$KEEP EPOCHS=${EPOCHS:-}" >> "$SB_LOG"\necho 4242\n',
    "hyakstorage": '#!/bin/bash\necho "│ Total:  │ 500GB / 1024GB  │ 500000 / 1000000 files │"\n',
    "df": '#!/bin/bash\necho "Filesystem 1024-blocks Used Available Capacity Mounted"\n'
          'echo "scrubbed 9999999999 1 ${DF_AVAIL_KB:-9999999999} 1% /x"\n',
    "curl": '#!/bin/bash\nwhile [ $# -gt 0 ]; do [ "$1" = -o ] && echo \'{"cities": []}\' > "$2"; shift; done\n',
    "squeue": '#!/bin/bash\n[ -z "${SQ_FAIL:-}" ] || exit 1\n[ -z "${SQ_OUT:-}" ] || cat "$SQ_OUT"\n',
}

FOUR = ["dinov2-large", "dinov2-reg-large", "convnextv2-base", "rampnet-convnextv2-base"]


@pytest.fixture
def launcher(tmp_path):
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    for name, body in STUBS.items():
        p = bin_ / name
        p.write_text(body)
        p.chmod(0o755)
    out, keep = tmp_path / "out", tmp_path / "keep"
    for lt in ("CurbRamp", "SurfaceProblem"):
        (out / "data" / lt).mkdir(parents=True)
        (out / "data" / lt / "split.csv").write_text("x\n")
    (out / "smoke").mkdir()
    (out / "smoke" / "backbone_inputs_smoke.json").write_text("{}")
    (out / "backbones" / "rampnet-convnextv2-base").mkdir(parents=True)
    (out / "backbones" / "rampnet-convnextv2-base" / "conversion.json").write_text('{"verified_against_timm": true}')

    def manifest(unavailable=()):
        rows = {k: {"available": k not in unavailable} for k in FOUR}
        (out / "backbone_manifest.json").write_text(json.dumps({"backbones": rows}))
    manifest()
    env = {**os.environ, "PATH": f"{bin_}:{os.environ['PATH']}", "OUT": str(out), "KEEP": str(keep),
           "R": str(tmp_path / "r3"), "PY": sys.executable, "SB_LOG": str(tmp_path / "sbatch.log")}

    def run(*args, **extra):
        return subprocess.run(["bash", os.path.join(ANALYSIS, "backbone_submit.sh"), *args], capture_output=True,
                              text=True, cwd=REPO, env={**env, **extra})
    return {"run": run, "out": out, "keep": keep, "manifest": manifest, "log": tmp_path / "sbatch.log"}


def rows(path):
    return [ln.split("\t") for ln in open(path).read().splitlines() if ln and not ln.startswith("#")]


def test_pilot_writes_thirty_rows_with_the_planned_tags_and_one_throttled_array(launcher):
    r = launcher["run"]("pilot")
    assert r.returncode == 0, r.stderr
    task = sorted(launcher["out"].glob("tasks_pilot_*.tsv"))[0]
    t = rows(task)
    assert len(t) == 30
    main = [x for x in t if x[4] == "1e-5"]
    lr = [x for x in t if x[4] == "3e-5"]
    assert len(main) == 24 and len(lr) == 6
    assert {(x[1], x[2]) for x in main} == {(a, s) for a in FOUR for s in ("42", "43", "44")}
    assert {x[1] for x in lr} == {"dinov2-large", "convnextv2-base", "rampnet-convnextv2-base"}
    from analysis.backbone_comparison import make_tag
    for lt, arm, seed, tag, rate in t:
        assert tag == make_tag("bb-", arm, int(seed), float(rate))
    assert t[0][3] == "bb-dinov2-large"   # seed 42 rows first, so a partial pilot is still paired
    sb = launcher["log"].read_text()
    assert "--job-name=backbone-comparison-pilot" in sb and "--array=1-30%8" in sb and "STEP=train" in sb
    ledger = rows(launcher["out"] / "backbone_jobs.txt")
    assert ledger[0][1:4] == ["4242", "30", "pilot"] and len(ledger[0][6]) == 64
    assert (launcher["keep"] / "backbone_jobs.txt").exists() and (launcher["keep"] / task.name).exists()


def test_an_unavailable_arm_is_skipped(launcher):
    launcher["manifest"](unavailable=("dinov2-reg-large",))
    r = launcher["run"]("pilot")
    assert r.returncode == 0 and "skipping dinov2-reg-large" in r.stdout
    t = rows(sorted(launcher["out"].glob("tasks_pilot_*.tsv"))[0])
    assert len(t) == 24 and "dinov2-reg-large" not in {x[1] for x in t}


def test_an_unverified_conversion_skips_rampnet(launcher):
    (launcher["out"] / "backbones" / "rampnet-convnextv2-base" / "conversion.json").write_text('{"verified_against_timm": false}')
    r = launcher["run"]("pilot")
    assert r.returncode == 0 and "skipping rampnet-convnextv2-base" in r.stdout


def test_the_job_cap_refuses_the_56th_element(launcher):
    out = launcher["out"]
    (out / "backbone_jobs.txt").write_text("t\t1\t26\tsmoke\tc\tf\ts\n")
    r = launcher["run"]("pilot")
    assert r.returncode != 0 and "cap" in r.stderr and not launcher["log"].exists()
    (out / "backbone_jobs.txt").write_text("t\t1\t25\tsmoke\tc\tf\ts\n")
    assert launcher["run"]("pilot").returncode == 0


def test_too_little_scrubbed_space_refuses(launcher):
    r = launcher["run"]("pilot", DF_AVAIL_KB="1048576")   # 1 GB
    assert r.returncode != 0 and "scrubbed" in r.stderr and not launcher["log"].exists()


def test_a_smoke_tag_is_refused_outside_the_smoke_job(launcher, tmp_path):
    bad = tmp_path / "bad.tsv"
    bad.write_text("CurbRamp\tdinov2-large\t42\tsmoke-dinov2-large\t1e-5\n")
    r = launcher["run"]("resubmit", str(bad))
    assert r.returncode != 0 and "smoke" in r.stderr and not launcher["log"].exists()
    good = tmp_path / "good.tsv"
    good.write_text("CurbRamp\tdinov2-large\t43\tbb-dinov2-large-s43\t1e-5\n")
    assert launcher["run"]("resubmit", str(good)).returncode == 0
    assert "--array=1-1%8" in launcher["log"].read_text()


def test_smoke_is_one_two_hour_job_over_five_rows(launcher):
    r = launcher["run"]("smoke")
    assert r.returncode == 0, r.stderr
    t = rows(launcher["out"] / "smoke" / "tasks_smoke.tsv")
    assert [x[3] for x in t] == ["smoke-dinov2-large", "smoke-dinov2-reg-large", "smoke-convnextv2-base",
                                 "smoke-rampnet-convnextv2-base", "smoke-rand-dinov3-vitl16"]
    sb = launcher["log"].read_text()
    assert sb.count("\n") == 1 and "--time=02:00:00" in sb and "--job-name=backbone-comparison-smoke" in sb
    assert "STEP=smoke" in sb and "EPOCHS=1" in sb and str(launcher["out"] / "smoke" / "keep") in sb
    assert (launcher["out"] / "smoke" / ".start").exists()


def test_a_requeue_after_best_was_copied_finishes_the_run(job):
    """PR #121 review S6: preempted between deleting checkpoint-* and touching slimmed, the requeued element must
    finish from best/ instead of failing on the deleted best checkpoint."""
    call, tmp = job
    assert call("CurbRamp", "dinov2-large", "42", "bb-dinov2-large", "1e-5").returncode == 0
    run, keep = tmp / "runs" / "CurbRamp" / "bb-dinov2-large", tmp / "keep" / "CurbRamp" / "runs" / "bb-dinov2-large"
    os.remove(run / "slimmed")
    os.remove(keep / "done")
    (run / "checkpoint-10").mkdir()          # a half-finished rm -rf
    r = call("CurbRamp", "dinov2-large", "42", "bb-dinov2-large", "1e-5")
    assert r.returncode == 0, r.stderr
    assert (keep / "done").exists() and (run / "slimmed").exists() and not list(run.glob("checkpoint-*"))
    assert len(os.listdir(run / "best")) == 4


def test_a_pilot_run_without_the_marker_finishes_from_a_full_best(job):
    """Final review R2: a run slimmed by the pilot's commit (no best_copied marker) and preempted after its best
    checkpoint was deleted finishes from best/."""
    call, tmp = job
    assert call("CurbRamp", "dinov2-large", "42", "bb-dinov2-large", "1e-5").returncode == 0
    run, keep = tmp / "runs" / "CurbRamp" / "bb-dinov2-large", tmp / "keep" / "CurbRamp" / "runs" / "bb-dinov2-large"
    for m in (run / "slimmed", run / "best_copied", keep / "done"):
        os.remove(m)
    r = call("CurbRamp", "dinov2-large", "42", "bb-dinov2-large", "1e-5")
    assert r.returncode == 0, r.stderr
    assert (keep / "done").exists() and (run / "slimmed").exists()


def test_the_job_checks_its_environment_and_appends_its_log():
    text = read("backbone.slurm")
    assert "#SBATCH --open-mode=append" in text
    body = text[text.index("setup_env() {"):text.index("run_one() {")]
    assert '"transformers": "4.56.2"' in body and "exit 3" in body


def test_dinov3_stage_submits_eighteen_rows_with_explicit_epochs_and_no_conda_state(launcher):
    out = launcher["out"]
    rows_ = {k: {"available": True} for k in FOUR + ["dinov3-vitl16", "dinov3-vitl16-256", "dinov3-vitb16"]}
    (out / "backbone_manifest.json").write_text(json.dumps({"backbones": rows_}))
    (out / "pip_freeze.txt").write_text("x==1\n")
    r = launcher["run"]("dinov3", CONDA_PREFIX="/somewhere", EPOCHS="1")
    assert r.returncode == 0, r.stderr
    f = sorted(out.glob("tasks_dinov3_*.tsv"))[-1]
    got = rows(f)
    assert len(got) == 18 and {g[1] for g in got} == {"dinov3-vitl16", "dinov3-vitl16-256", "dinov3-vitb16"}
    assert {g[3] for g in got} >= {"bb-dinov3-vitl16", "bb-dinov3-vitb16-s43", "bb-dinov3-vitl16-256-s44"}
    log = launcher["log"].read_text()
    assert "--array=1-18%8" in log and "EPOCHS=25" in log
    assert (launcher["keep"] / "pip_freeze.txt").exists() and (launcher["keep"] / "backbone_manifest.json").exists()


def test_dinov3_stage_refuses_other_arms_and_skips_done_runs(launcher):
    out, keep = launcher["out"], launcher["keep"]
    rows_ = {k: {"available": True} for k in FOUR + ["dinov3-vitl16", "dinov3-vitl16-256", "dinov3-vitb16"]}
    (out / "backbone_manifest.json").write_text(json.dumps({"backbones": rows_}))
    r = launcher["run"]("dinov3", ARMS="dinov2-large")
    assert r.returncode != 0 and "DINOv3 arms only" in r.stderr
    (keep / "CurbRamp" / "runs" / "bb-dinov3-vitl16").mkdir(parents=True)
    (keep / "CurbRamp" / "runs" / "bb-dinov3-vitl16" / "done").write_text("")
    r = launcher["run"]("dinov3", ARMS="dinov3-vitl16", SEEDS="42")
    assert r.returncode == 0, r.stderr
    got = rows(sorted(out.glob("tasks_dinov3_*.tsv"))[-1])
    assert [g[3] for g in got] == ["bb-dinov3-vitl16"] and got[0][0] == "SurfaceProblem"


# ---------------------------------------------------------------------------------------------------------------------
# fix round 2: in-flight rows, resubmit, refused stages
# ---------------------------------------------------------------------------------------------------------------------

def pilot_task_file(launcher, name="tasks_pilot_old.tsv"):
    """A copy of what the pilot stage writes, registered in the ledger as job 777."""
    assert launcher["run"]("pilot").returncode == 0
    first = sorted(launcher["out"].glob("tasks_pilot_*.tsv"))[0]
    old = launcher["out"] / name
    old.write_text(first.read_text())
    first.unlink()
    (launcher["out"] / "backbone_jobs.txt").write_text(f"t\t777\t1\tpilot\tc\t{old}\ts\n")
    launcher["log"].unlink()
    return rows(old)


def test_queued_or_running_rows_are_never_submitted_twice(launcher, tmp_path):
    """PR #121 final review R4: squeue's elements of this stream map back to task rows through the ledger; those rows
    are skipped (here 2 running, 5 and 6 pending in an unexpanded range), other users' jobs are ignored."""
    old = pilot_task_file(launcher)
    q = tmp_path / "squeue.txt"
    q.write_text("777_2 backbone-comparison-pilot\n777_[5-6%8] backbone-comparison-pilot\n555_1 someone-else\n")
    r = launcher["run"]("pilot", SQ_OUT=str(q))
    assert r.returncode == 0, r.stderr
    got = rows(sorted(launcher["out"].glob("tasks_pilot_2*.tsv"))[-1])
    skipped = {(old[i][0], old[i][3]) for i in (1, 4, 5)}
    assert len(got) == 27 and not skipped & {(g[0], g[3]) for g in got}
    assert r.stdout.count("queued or running (squeue)") == 3
    assert not list(launcher["out"].glob(".squeue*"))   # the squeue snapshot and its mapping are removed on exit


def test_the_keep_copy_maps_a_purged_task_file(launcher, tmp_path):
    old = pilot_task_file(launcher)
    keep = launcher["keep"]
    keep.mkdir(exist_ok=True)
    (keep / "tasks_pilot_old.tsv").write_text((launcher["out"] / "tasks_pilot_old.tsv").read_text())
    (launcher["out"] / "tasks_pilot_old.tsv").unlink()
    q = tmp_path / "squeue.txt"
    q.write_text("777_1 backbone-comparison-pilot\n")
    r = launcher["run"]("pilot", SQ_OUT=str(q))
    assert r.returncode == 0, r.stderr
    assert f"skipping {old[0][0]} {old[0][3]}: queued or running" in r.stdout


def test_an_unreadable_squeue_or_an_unknown_job_refuses(launcher, tmp_path):
    r = launcher["run"]("pilot", SQ_FAIL="1")
    assert r.returncode != 0 and "squeue" in r.stderr and not launcher["log"].exists()
    q = tmp_path / "squeue.txt"
    q.write_text("888_1 backbone-comparison-pilot\n")
    r = launcher["run"]("pilot", SQ_OUT=str(q))
    assert r.returncode != 0 and "not in the ledger" in r.stderr and not launcher["log"].exists()


def test_resubmit_skips_done_and_in_flight_rows_and_refuses_unplanned_tags(launcher, tmp_path):
    old = pilot_task_file(launcher)
    keep = launcher["keep"]
    (keep / old[0][0] / "runs" / old[0][3]).mkdir(parents=True)
    (keep / old[0][0] / "runs" / old[0][3] / "done").write_text("")
    q = tmp_path / "squeue.txt"
    q.write_text("777_2 backbone-comparison-pilot\n")
    src = tmp_path / "again.tsv"
    src.write_text("# failed elements\n" + "\n".join("\t".join(x) for x in old[:3]) + "\n")
    r = launcher["run"]("resubmit", str(src), SQ_OUT=str(q))
    assert r.returncode == 0, r.stderr
    got = rows(sorted(launcher["out"].glob("tasks_resubmit_*.tsv"))[-1])
    assert got == [old[2]] and "--array=1-1%8" in launcher["log"].read_text()
    assert "done already" in r.stdout and "queued or running" in r.stdout
    src.write_text("\n".join("\t".join(x) for x in old[:2]) + "\n")
    r = launcher["run"]("resubmit", str(src), SQ_OUT=str(q))
    assert r.returncode != 0 and "no runnable rows" in r.stderr
    src.write_text("CurbRamp\trampnet-convnextv2-base\t42\tbb-rampnet-convnextv2-base-r2\t1e-5\n")
    r = launcher["run"]("resubmit", str(src))
    assert r.returncode != 0 and "not the planned tag" in r.stderr


@pytest.mark.parametrize("stage", ["seeds", "rest"])
def test_seeds_and_rest_are_refused_until_the_report_reads_them(launcher, stage):
    r = launcher["run"](stage, ARMS="dinov2-reg-large", TYPES="Obstacle", SEEDS="43")
    assert r.returncode != 0 and "needs the follow-up PR" in r.stderr and not launcher["log"].exists()


def test_onnx_is_refused_before_spending_a_cap_job(launcher, tmp_path):
    run, out = launcher["run"], launcher["out"]
    r = run("onnx", LT="CurbRamp", TAG="bb-dinov2-reg-large")
    assert r.returncode != 0 and "opset 17" in r.stderr
    nope, yes = tmp_path / "py_no_onnx", tmp_path / "py_onnx"
    nope.write_text("#!/bin/bash\nexit 1\n")
    yes.write_text("#!/bin/bash\nexit 0\n")
    for p in (nope, yes):
        p.chmod(0o755)
    r = run("onnx", LT="CurbRamp", TAG="bb-convnextv2-base", PY=str(nope))
    assert r.returncode != 0 and "onnx and onnxruntime do not import" in r.stderr
    r = run("onnx", LT="CurbRamp", TAG="bb-convnextv2-base", PY=str(yes))
    assert r.returncode != 0 and "best/model.safetensors" in r.stderr
    assert not launcher["log"].exists()
    best = out / "runs" / "CurbRamp" / "bb-convnextv2-base" / "best"
    best.mkdir(parents=True)
    (best / "model.safetensors").write_text("w")
    r = run("onnx", LT="CurbRamp", TAG="bb-convnextv2-base", PY=str(yes))
    assert r.returncode == 0, r.stderr
    assert "STEP=onnx" in launcher["log"].read_text()
