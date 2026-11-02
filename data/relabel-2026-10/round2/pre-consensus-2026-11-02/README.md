# Ground-truth re-label round 2: votes before the consensus meeting

Snapshot taken 2026-11-02, after both raters finished all 300 items and before the consensus meeting. In the
meeting the raters change some of their votes in Project Sidewalk, which overwrites them in the production database,
so this folder is the only record of the independent votes. The agreement numbers in
`docs/ai-disagree-relabel-2026-10.md` are computed from these files.

| File | Contents |
|---|---|
| `votes_jon.csv`, `votes_mikey.csv` | Each rater's 300 verdicts (`correct` / `incorrect` / `cant_tell`), keyed by packet id |
| `comments_jon.csv`, `comments_mikey.csv` | Each rater's Gallery comments, keyed by `(schema, label_id)` |
| `unmasked.csv` | The 11 items rated with the AI vote visible (author excluded, so hidden from Gallery); left out of agreement |
| `mismatches.csv` | The 17 shared items where the two verdicts differ: the consensus meeting's agenda |
| `SHA256SUMS` | sha256 of each CSV; each matches the pin in the round's `manifest.json` |

The packet key that maps packet ids to labels is committed after the meeting, because it also carries the AI vote and
the crowd votes, which stay hidden from the raters until then.

Check the files with `shasum -a 256 -c SHA256SUMS`.
