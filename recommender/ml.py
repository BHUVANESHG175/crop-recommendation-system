
import logging
import math
from pathlib import Path

import joblib
import pandas as pd
from django.conf import settings


logger = logging.getLogger(__name__)

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


def load_bundle():
    """Load and validate the saved crop prediction model."""

    if not MODEL_PATH.is_file():
        logger.error("Crop model file not found: %s", MODEL_PATH)
        return None

    try:
        bundle = joblib.load(MODEL_PATH)

        if not isinstance(bundle, dict):
            raise ValueError("The model file must contain a dictionary.")

        required_keys = {"models", "features_cols", "classes"}

        if not required_keys.issubset(bundle):
            raise ValueError("The model file is missing required keys.")

        model = bundle["models"]
        feature_order = bundle["features_cols"]
        classes = bundle["classes"]

        if not callable(getattr(model, "predict", None)):
            raise ValueError("The saved model cannot make predictions.")

        if list(feature_order) != EXPECTED_FEATURES:
            raise ValueError("The model feature order is incorrect.")

        if len(classes) == 0:
            raise ValueError("The model has no crop classes.")

        if getattr(model, "n_features_in_", None) != len(EXPECTED_FEATURES):
            raise ValueError("The model expects an incorrect feature count.")

        logger.info("Crop prediction model loaded successfully.")
        return bundle

    except Exception:
        logger.exception("Unable to load or validate the crop model.")
        return None


def predict_one(bundle, input_data):
    """Return one crop prediction using the saved model bundle."""

    if bundle is None:
        raise RuntimeError(
            "The crop prediction model is unavailable. "
            "Please contact the administrator."
        )

    if not isinstance(input_data, dict):
        raise ValueError("Prediction inputs must be a dictionary.")

    model = bundle.get("models")
    feature_order = bundle.get("features_cols")
    classes = bundle.get("classes")

    if (
        not callable(getattr(model, "predict", None))
        or list(feature_order or []) != EXPECTED_FEATURES
        or classes is None
        or len(classes) == 0
    ):
        raise RuntimeError("The crop model bundle is invalid.")

    values = []

    for name in EXPECTED_FEATURES:
        if name not in input_data:
            raise ValueError(f"Missing required input: {name}")

        try:
            value = float(input_data[name])
        except (TypeError, ValueError):
            raise ValueError(
                f"The value for {name} must be numeric."
            ) from None

        if not math.isfinite(value):
            raise ValueError(f"{name} must be a finite number.")

        values.append(value)

    # Preserve the feature names and order used during training.
    input_frame = pd.DataFrame(
        [values],
        columns=EXPECTED_FEATURES,
    )

    try:
        predictions = model.predict(input_frame)
    except Exception as error:
        logger.exception("Crop prediction failed.")
        raise RuntimeError(
            "The crop prediction failed. Please try again later."
        ) from error

    if len(predictions) != 1:
        raise RuntimeError("The model returned an unexpected result.")

    crop = str(predictions[0]).strip()
    allowed_classes = {str(label) for label in classes}

    if not crop or crop not in allowed_classes:
        raise RuntimeError("The model returned an invalid crop label.")

    return crop
