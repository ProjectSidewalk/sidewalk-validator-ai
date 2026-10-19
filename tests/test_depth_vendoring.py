"""Depth Anything V2 is a pinned git submodule, not a vendored tree, and no Python bytecode is tracked.

Reads only git metadata, so it needs neither the submodule's files nor torch.
"""
import configparser
import pathlib
import subprocess

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
DEPTH_PATH = "Depth-Anything-V2"   # data_generation.py puts Depth-Anything-V2/metric_depth on sys.path


def _git(*args):
    try:
        return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        pytest.skip(f"not a git checkout: {e}")


def test_gitmodules_pins_depth_anything_v2():
    cfg = configparser.ConfigParser()
    cfg.read(ROOT / ".gitmodules")
    sub = cfg[f'submodule "{DEPTH_PATH}"']
    assert sub["path"] == DEPTH_PATH
    assert sub["url"].rstrip("/").removesuffix(".git") == "https://github.com/DepthAnything/Depth-Anything-V2"
    # Mode 160000 is a gitlink: the path is pinned to one upstream commit, not a copy of its files.
    entry = _git("ls-files", "--stage", "--", DEPTH_PATH).split()
    assert entry and entry[0] == "160000", f"{DEPTH_PATH} is not a submodule gitlink: {entry}"


def test_no_bytecode_tracked():
    tracked = [p for p in _git("ls-files").splitlines() if p.endswith(".pyc") or "__pycache__" in p.split("/")]
    assert not tracked, f"tracked bytecode: {tracked}"
