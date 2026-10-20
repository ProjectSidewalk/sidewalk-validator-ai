"""Depth Anything V2 is a pinned git submodule, not a vendored tree, and no Python bytecode is tracked.

Reads only git metadata, so it needs neither the submodule's files nor torch.
"""
import configparser
import pathlib
import subprocess

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
DEPTH_PATH = "Depth-Anything-V2"   # data_generation.py puts Depth-Anything-V2/metric_depth on sys.path


# The commit sidewalk-ai-api pins, so training and serving run the same depth code. Moving it is a deliberate change:
# update this, and check sidewalk-ai-api's pin.
PINNED_SHA = "a561b849ebae10a6f5ef49e26c83cbbcd36c71bf"


def _git(*args):
    """Run git in the repo root; skip unless ROOT is itself a git checkout (not, say, a tarball inside another repo)."""
    def run(*a):
        return subprocess.run(["git", *a], cwd=ROOT, check=True, capture_output=True, text=True).stdout
    try:
        top = run("rev-parse", "--show-toplevel").strip()
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        pytest.skip(f"not a git checkout: {e}")
    if pathlib.Path(top).resolve() != ROOT:
        pytest.skip(f"{ROOT} is not the top of a git checkout (top is {top})")
    return run(*args)


def test_gitmodules_pins_depth_anything_v2():
    cfg = configparser.ConfigParser()
    cfg.read(ROOT / ".gitmodules")
    sub = cfg[f'submodule "{DEPTH_PATH}"']
    assert sub["path"] == DEPTH_PATH
    assert sub["url"].rstrip("/").removesuffix(".git") == "https://github.com/DepthAnything/Depth-Anything-V2"
    # Mode 160000 is a gitlink: the path is pinned to one upstream commit, not a copy of its files.
    entry = _git("ls-files", "--stage", "--", DEPTH_PATH).split()
    assert entry and entry[0] == "160000", f"{DEPTH_PATH} is not a submodule gitlink: {entry}"
    assert entry[1] == PINNED_SHA, f"{DEPTH_PATH} is pinned to {entry[1]}, expected {PINNED_SHA}"


def test_no_bytecode_tracked():
    tracked = [p for p in _git("ls-files").splitlines() if p.endswith(".pyc") or "__pycache__" in p.split("/")]
    assert not tracked, f"tracked bytecode: {tracked}"
