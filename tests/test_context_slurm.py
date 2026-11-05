"""The context-arms job files (issue #14): they parse, run preemptible with requeue, document every required
variable, and the launcher cancels dependents of a failed job."""
import os
import re
import shutil
import subprocess

import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..", "analysis")
SCRIPTS = ["context_arms.slurm", "context_submit.sh"]


def read(name):
    with open(os.path.join(ROOT, name)) as f:
        return f.read()


@pytest.mark.skipif(shutil.which("bash") is None, reason="no bash")
@pytest.mark.parametrize("name", SCRIPTS)
def test_bash_parses(name):
    r = subprocess.run(["bash", "-n", os.path.join(ROOT, name)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_slurm_is_preemptible_requeues_and_is_offline():
    text = read("context_arms.slurm")
    assert "#SBATCH --partition=ckpt-all" in text and "#SBATCH --requeue" in text
    assert "#SBATCH --account=makelab" in text and "HF_HUB_OFFLINE=1" in text
    assert "already evaluated" in text and 'rm -rf "$RUN"/checkpoint-*' in text


@pytest.mark.parametrize("name", SCRIPTS)
def test_every_required_variable_is_documented_in_the_header(name):
    text = read(name)
    header = "\n".join(ln for ln in text.splitlines() if ln.startswith("#"))
    required = set(re.findall(r"\$\{(\w+):\?", text))
    assert required, name
    for var in required:
        assert re.search(rf"\b{var}\b", header), f"{name}: {var} is required but not documented in the header"


def test_train_step_scores_with_context_model_then_evaluate():
    text = read("context_arms.slurm")
    train = text.split("  train)")[1]
    assert "--context-arm \"$ARM\"" in train and "--context-view-res \"$RES\"" in train
    assert train.index("context_model.py predict") < train.index("evaluate.py")
    assert "--skip-inference --tag \"$TAG\" --per-city" in train
    assert "context_crops.py backfill" in text and "--verify 50" in text


def test_launcher_kills_invalid_dependents_and_logs_job_ids():
    text = read("context_submit.sh")
    assert "--kill-on-invalid-dep=yes" in text and "context_jobs.txt" in text
    assert "STEP=backfill" in text and "STEP=train" in text and 'afterok:$bf' in text
    assert os.access(os.path.join(ROOT, "context_submit.sh"), os.X_OK)
