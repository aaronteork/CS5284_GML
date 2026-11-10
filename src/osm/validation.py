"""Check that paired prediction diagnostics compare the same target population."""

import numpy as np


def validate_prediction_pairs(control, variant):
    if not control or len(control) != len(variant):
        raise ValueError("Control and variant need equal, nonzero numbers of paired seeds")
    reference = control[0]
    for prediction in [*control, *variant]:
        if list(map(tuple, prediction["months"])) != list(map(tuple, reference["months"])):
            raise ValueError("Prediction months differ or are ordered differently")
        for key in ("road_index", "y"):
            if not np.array_equal(np.asarray(prediction[key]), np.asarray(reference[key])):
                raise ValueError(f"Prediction {key} differs; align roads and targets before comparison")
        if np.shape(prediction["pred"]) != np.shape(reference["y"]):
            raise ValueError("Prediction shape differs from target shape")
