"""analysis/retrain_report.py's split reader (issue #114). The report itself is covered through the scoreboard and
job-crop tests that import it."""
import pandas as pd

from analysis import retrain_report as rr


def test_load_type_keeps_all_digit_pano_ids_as_strings(tmp_path):
    # Mapillary pano_ids are all digits; pano_id is the cluster-bootstrap group, so it must stay a string.
    ids = ["1792391791667719", "2001774006895864"]
    (tmp_path / "CurbRamp").mkdir()
    pd.DataFrame({"label_uid": ["m:1", "m:2"], "true": ["correct", "incorrect"], "p_correct": [0.8, 0.3],
                  "p_incorrect": [0.2, 0.7], "split": "test"}).to_csv(tmp_path / "CurbRamp" / "predictions.csv",
                                                                      index=False)
    pd.DataFrame({"label_uid": ["m:1", "m:2"], "city": "m", "pano_id": ids, "label_class": ["correct", "incorrect"],
                  "class_source": "votes", "author": "human", "hard": 0, "split": "test", "severity": "",
                  "tags": ""}).to_csv(tmp_path / "CurbRamp" / "split.csv", index=False)
    df, split = rr.load_type(str(tmp_path), "CurbRamp")
    for frame in (df, split):
        assert not pd.api.types.is_numeric_dtype(frame.pano_id)
        assert frame.pano_id.tolist() == ids
