"""Calibration of the validator's confidence into the accuracy map served by sidewalk-ai-api.

See ``_isotonic_map`` (shared fitting and the API's lookup), ``calibrate_from_production`` (maps for the deployed
models from production outcomes) and ``calibrate`` (maps for a retrained model from its predictions.csv).
"""
