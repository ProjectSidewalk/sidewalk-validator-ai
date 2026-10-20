# Repo size: purging the old vendored Depth-Anything-V2 tree from history (optional)

`Depth-Anything-V2/` used to be a vendored copy of upstream (86 files, 64 MB on disk, including 12
committed `.pyc` files). It is now a git submodule pinned to upstream
[`a561b84`](https://github.com/DepthAnything/Depth-Anything-V2/commit/a561b849ebae10a6f5ef49e26c83cbbcd36c71bf)
([#11](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/11)). Removing the tree from `HEAD` does
**not** make clones smaller: every old commit still references those blobs, so a full clone still downloads them.

## What history holds (measured 2026-10-18)

The pack is 45 MB. Of the 48.3 MB of blobs on disk in history, 47.4 MB are under `Depth-Anything-V2/`. The largest:

| Size (raw) | Path |
|---|---|
| 13.7 MB | `Depth-Anything-V2/metric_depth/dataset/splits/hypersim/train.txt` (compresses to 0.4 MB) |
| 12.8 MB | `Depth-Anything-V2/assets/teaser.png` |
| 9.7 MB | `Depth-Anything-V2/assets/examples_video/basketball.mp4` |
| 9.2 MB | `Depth-Anything-V2/metric_depth/assets/compare_zoedepth.png` |
| 5.3 MB | `Depth-Anything-V2/assets/examples_video/ferris_wheel.mp4` |
| 3.6 MB | `Depth-Anything-V2/metric_depth/dataset/splits/vkitti2/train.txt` |
| 1.7 MB | `Depth-Anything-V2/metric_depth/dataset/splits/hypersim/val.txt` |
| 1.2 MB | `Depth-Anything-V2/assets/DA-2K.png` |

Reproduce with:

```bash
git rev-list --objects --all \
  | git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(objectsize:disk) %(rest)' \
  | grep '^blob' | sort -k3 -n | tail
```

## No rewrite: a blobless clone

Anyone who just wants a small clone can skip the old blobs without changing the repo:

```bash
git clone --filter=blob:none --recurse-submodules https://github.com/ProjectSidewalk/sidewalk-validator-ai
```

Git fetches a blob only when a checkout or command needs it, so the vendored tree's blobs are never downloaded
unless someone checks out an old commit. This needs nothing from anyone else and is the recommended default.

## Rewrite: purge it from history (opt-in, not done)

This shrinks the pack for everyone (tested on a scratch clone: 45.4 MB to 254 KB, with the submodule gitlink
and `.gitmodules` intact, because the glob matches files *under* `Depth-Anything-V2/` but not the gitlink at
the path itself). **It rewrites every commit SHA.** Only run it after deciding to accept that:

- every commit, tag and branch gets a new SHA; links to old commit SHAs (in issues, PRs, `generation_run.json`
  records, Hugging Face model cards) stop resolving to this history;
- every existing clone, including klone's, must be deleted and re-cloned (pulling into an old clone re-merges
  the old history and undoes the purge);
- every commit from before the submodule loses `Depth-Anything-V2/` entirely (the glob also removes the
  vendored files from those commits), so the commits the published models were generated from no longer run
  as checked out: reproducing them needs upstream
  [`e5a2732`](https://github.com/DepthAnything/Depth-Anything-V2/commit/e5a2732d3ea2cddc081d7bfd708fc0bf09f812f1)
  (what the vendored copy was) cloned into `Depth-Anything-V2/` by hand;
- the push below rewrites the open PRs' branches along with `master`, so those PRs follow automatically (merged
  PRs keep showing the old SHAs); unpushed local work is what needs carrying over (rebase it onto the new
  history after re-cloning);
- nothing should be running from a checkout (data generation, training) while this happens.

[`git filter-repo`](https://github.com/newren/git-filter-repo) (`pip install git-filter-repo`, or
`uvx git-filter-repo`) works on a fresh mirror clone:

```bash
git clone --mirror https://github.com/ProjectSidewalk/sidewalk-validator-ai sva-purge.git
cd sva-purge.git
git filter-repo --path-glob 'Depth-Anything-V2/*' --invert-paths
git count-objects -vH            # size-pack should now be well under 1 MB
# filter-repo removes the origin remote as a safety measure; add it back to push
git remote add origin https://github.com/ProjectSidewalk/sidewalk-validator-ai
git push --force origin 'refs/heads/*:refs/heads/*' 'refs/tags/*:refs/tags/*'
```

Push branches and tags explicitly rather than with `--mirror`: a mirror clone also holds GitHub's read-only
`refs/pull/*`, and `git push --mirror` tries to push those too, which GitHub rejects, so the push exits
non-zero.

GitHub keeps the old objects reachable through PR refs (`refs/pull/*`) and its caches, so the server-side size
drops only after GitHub garbage-collects, which may need a request to GitHub Support. Clones made afterwards are
small either way.
