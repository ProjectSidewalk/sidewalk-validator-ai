"""The cross-region job files: they parse, run on the preemptible partition with requeue, and the launcher cancels
dependents of a failed job."""
import os
import shutil
import subprocess

import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..", "analysis")
SLURM = ["region_transfer.slurm", "region_transfer_cpu.slurm"]
SCRIPTS = SLURM + ["region_submit.sh"]


def read(name):
    with open(os.path.join(ROOT, name)) as f:
        return f.read()


@pytest.mark.skipif(shutil.which("bash") is None, reason="no bash")
@pytest.mark.parametrize("name", SCRIPTS)
def test_bash_parses(name):
    r = subprocess.run(["bash", "-n", os.path.join(ROOT, name)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


@pytest.mark.parametrize("name", SLURM)
def test_slurm_is_preemptible_and_requeues(name):
    text = read(name)
    assert "#SBATCH --partition=ckpt-all" in text and "#SBATCH --requeue" in text
    assert "#SBATCH --account=makelab" in text


def test_gpu_job_is_idempotent_and_offline():
    text = read("region_transfer.slurm")
    assert "HF_HUB_OFFLINE=1" in text and "already evaluated" in text
    assert 'rm -rf "$RUN"/checkpoint-*' in text and "KEEP_CKPT=1" in text   # global keeps its weights for ft
    for arm in ("global)", "ft)", "cond)", "condall)"):
        assert arm in text


def test_launcher_kills_invalid_dependents_and_logs_job_ids():
    text = read("region_submit.sh")
    assert "--kill-on-invalid-dep=yes" in text and "region_jobs.txt" in text
    assert "region_transfer.py variants" in text and "region_transfer.py list" in text
    assert os.access(os.path.join(ROOT, "region_submit.sh"), os.X_OK)


@pytest.mark.parametrize("name", ["region_submit.sh", "region_transfer_cpu.slurm"])
def test_every_variants_call_passes_rawlabels(name):
    # split.csv has no coordinates (data_generation.MANIFEST_COLUMNS + hard + split), so the 3 m merge of the pools
    # needs the rawLabels cache; without it variants raises before a single job is submitted.
    text = read(name)
    calls = text.split("region_transfer.py variants")[1:]
    assert calls
    for call in calls:
        assert '--rawlabels "$RAWLABELS"' in call.split("--out")[0], name
    assert 'RAWLABELS="${RAWLABELS:-$DATA/rawlabels}"' in text


def test_split_link_is_made_once_by_the_launcher_not_per_gpu_job():
    assert "ln -sfn" not in read("region_transfer.slurm")
    assert "ln -sfn" not in read("region_submit.sh")
    assert '|| ln -s "$DATA/$LT/split.csv" "$DATA/$LT/region/eval/$LT/split.csv"' in read("region_submit.sh")
    assert '|| ln -s "$DATA/$LT/split.csv" "$DATA/$LT/region/eval/$LT/split.csv"' in read("region_transfer_cpu.slurm")


def test_force_unk_ablation_is_scored_for_condall_only():
    text = read("region_transfer.slurm")
    assert text.count("\" --force-unk") == 1 and 'if [ "$ARM" = condall ]; then' in text
