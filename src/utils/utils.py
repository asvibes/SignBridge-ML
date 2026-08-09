"""
utils.py
--------
Shared helper functions used across the SignBridge-ML pipeline:
config loading, landmark normalization, logging setup, and path helpers.

Keeping these in one place makes it easy to plug in new feature
representations later (e.g. sequences of landmarks for dynamic signs)
without touching every script that uses them.
"""

import os
import logging

import yaml
import numpy as np


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load the project YAML config into a dict."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def setup_logger(name: str = "signbridge") -> logging.Logger:
    """Return a configured logger. Reused by every script for consistent output."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s %(name)s: %(message)s", "%H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def ensure_dir(path: str) -> None:
    """Create a directory (and parents) if it doesn't already exist."""
    os.makedirs(path, exist_ok=True)


def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """
    Normalize a (21, 3) array of MediaPipe hand landmarks so the model
    is robust to hand position/scale changes within the frame.

    Steps:
    1. Translate so the wrist (landmark 0) is the origin.
    2. Scale so the max distance from the wrist is 1.0.

    Returns a flattened (63,) feature vector: [x0,y0,z0,x1,y1,z1,...].

    NOTE: For dynamic signs (future work), this same per-frame
    normalization can be applied before stacking frames into a
    sequence for a sequence model (e.g. LSTM/GRU/Transformer).
    """
    landmarks = landmarks.copy().astype(np.float32)
    wrist = landmarks[0].copy()
    landmarks -= wrist

    max_dist = np.max(np.linalg.norm(landmarks, axis=1))
    if max_dist > 1e-6:
        landmarks /= max_dist

    return landmarks.flatten()


def landmarks_to_feature_vector(landmarks_list) -> np.ndarray:
    """
    Convert MediaPipe's landmark objects (each with .x, .y, .z) into a
    normalized (63,) numpy feature vector. Convenience wrapper around
    normalize_landmarks() for callers holding raw MediaPipe output.
    """
    coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks_list])
    return normalize_landmarks(coords)


# ---------------------------------------------------------------------------
# Model loading / label decoding
# ---------------------------------------------------------------------------
# Added for evaluate.py, predict.py and app.py. Kept here (rather than
# duplicated in each script) so every consumer of a trained model loads it
# the same way and gets the same error messages.
#
# NOTE on "label encoder": train_model.py fits scikit-learn classifiers
# directly on the string labels from the "label" column of the processed
# CSVs (e.g. "hello", "thank_you"). scikit-learn stores the sorted unique
# labels it saw during fit() on `model.classes_`, so the model already
# predicts human-readable strings -- there's no separate LabelEncoder
# artifact on disk. `decode_label()` below just turns the raw label
# ("thank_you") into a display-friendly one ("Thank You").


def load_model(model_path: str):
    """
    Load a trained scikit-learn model saved via joblib.dump()
    (see train_model.py, which writes models/final/model.joblib).

    Raises:
        FileNotFoundError: if model_path does not exist.
        RuntimeError: if the file exists but can't be unpickled
                      (e.g. corrupted file / incompatible joblib version).
    """
    import joblib

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No trained model found at '{model_path}'. "
            "Run src/models/train_model.py first to produce models/final/model.joblib."
        )

    try:
        model = joblib.load(model_path)
    except Exception as e:  # corrupted / incompatible pickle, wrong file type, etc.
        raise RuntimeError(f"Could not load model from '{model_path}': {e}") from e

    return model


def get_label_classes(model) -> np.ndarray:
    """
    Return the array of class labels the model was trained on
    (model.classes_), in the same order used internally by
    predict_proba(). Centralized here so callers don't need to know
    which attribute scikit-learn happens to expose it as.
    """
    if not hasattr(model, "classes_"):
        raise RuntimeError(
            "Loaded model has no 'classes_' attribute -- it doesn't look like "
            "a fitted scikit-learn classifier."
        )
    return model.classes_


def decode_label(raw_label: str) -> str:
    """
    Turn a raw vocabulary label (as stored in config.yaml / the training
    CSVs, e.g. 'thank_you') into a human-readable display label
    (e.g. 'Thank You').
    """
    return str(raw_label).replace("_", " ").strip().title()
