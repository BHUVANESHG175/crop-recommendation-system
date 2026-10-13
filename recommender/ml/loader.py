
import math
import pickle
import warnings
from functools import lru_cache
from pathlib import Path

import pandas as pd
from django.conf import settings
from sklearn.exceptions import InconsistentVersionWarning


MODEL_PATH = (
    Path(settings.BASE_DIR)
    / "recommender"
    / "ml"
    / "Crop_recommendation_RF.pkl"
)

EXPECTED_FEATURES = [
    "N",
    "P",
    "K",
    "temperature",
    "humidity",
    "ph",
    "rainfall",
]


@lru_cache(maxsize=1)
def load_bundle():
    """Load and validate the saved crop recommendation model."""

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            "The crop recommendation model file was not found."
        )

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", InconsistentVersionWarning)

            with MODEL_PATH.open("rb") as model_file:
                bundle = pickle.load(model_file)

    except (
        pickle.UnpicklingError,
        EOFError,
        AttributeError,
        ImportError,
        ModuleNotFoundError,
        ValueError,
    ) as error:
        raise ValueError(
            "The crop recommendation model file could not be loaded."
        ) from error

    if not isinstance(bundle, dict):
        raise ValueError("The model file has an invalid format.")

    required_keys = {"models", "features_cols", "classes"}
    if not required_keys.issubset(bundle):
        raise ValueError("The model file is missing required information.")

    model = bundle["models"]
    feature_order = bundle["features_cols"]
    classes = bundle["classes"]

    if not hasattr(model, "predict"):
        raise ValueError("The saved model does not support prediction.")

    if not isinstance(feature_order, (list, tuple)):
        raise ValueError("The model feature list has an invalid format.")

    if list(feature_order) != EXPECTED_FEATURES:
        raise ValueError(
            "The model feature order does not match the application."
        )

    if not isinstance(classes, (list, tuple)) or len(classes) == 0:
        # NumPy arrays are also valid class lists.
        if not hasattr(classes, "size") or classes.size == 0:
            raise ValueError("The model class list is empty or invalid.")

    if getattr(model, "n_features_in_", len(feature_order)) != len(feature_order):
        raise ValueError(
            "The model expects an unexpected number of features."
        )

    return bundle


def predict_one(features_dict):
    """Return one predicted crop label."""

    if not isinstance(features_dict, dict):
        raise ValueError(
            "Prediction inputs must be provided as a dictionary."
        )

    bundle = load_bundle()
    model = bundle["models"]
    feature_order = bundle["features_cols"]
    classes = bundle["classes"]

    values = {}

    for feature in feature_order:
        if feature not in features_dict:
            raise ValueError(f"Missing required feature: {feature}")

        try:
            value = float(features_dict[feature])
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"The value for {feature} must be numeric."
            ) from error

        if not math.isfinite(value):
            raise ValueError(
                f"The value for {feature} must be a finite number."
            )

        values[feature] = value

    # DataFrame column names and order match the model's training features.
    input_data = pd.DataFrame(
        [[values[feature] for feature in feature_order]],
        columns=feature_order,
    )

    predictions = model.predict(input_data)

    if len(predictions) != 1:
        raise ValueError(
            "The model returned an unexpected number of predictions."
        )

    predicted_label = predictions[0]

    if hasattr(predicted_label, "item"):
        predicted_label = predicted_label.item()

    if predicted_label not in classes:
        raise ValueError("The model returned an unknown crop label.")

    return str(predicted_label)
