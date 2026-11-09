# Golden fixtures for `job_crop.py` (copied, do not edit)

Verbatim copy of [ProjectSidewalk/SidewalkWebpage](https://github.com/ProjectSidewalk/SidewalkWebpage)
`test/resources/crops/` at commit `5d89dc179` (`mechanics.json`, `sizing-v2.json`, `synthetic-pano.png`,
`expected/*.png`; the upstream `README.md` and `generate_fixtures.py` are not copied). They are read by
`tests/test_job_crop.py`, which checks the port against them the way `test/service/CropGeometrySpec.scala` checks the
backend.

Upstream they were written by the reference implementation itself: `generate_fixtures.py` in that directory, run
against ProjectSidewalk/sidewalk-panorama-tools `CropRunner.py` at `c1d0751ae3f9715a823a55cd9ddaf48503b030f0`
(see the upstream README for the environment and command).

**Regenerate upstream, never edit here.** When `CropSizingRule.Version` changes, SidewalkWebpage regenerates these
files; copy the new set over this directory, update the commit above, and bump `job_crop.RULE_VERSION`.

SidewalkWebpage's license, which covers these files:

```
MIT License

Copyright (c) 2016-2022 Makeability Lab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
