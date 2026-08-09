"""
app.py
-------
SignBridge-ML main application entry point.

Currently supports one mode -- real-time prediction -- which:
    1. Loads configs/config.yaml.
    2. Loads the trained model from models/final/model.joblib.
    3. Opens the webcam and runs the prediction loop from
       src/inference/predict.py until the user quits.
    4. Shuts down the camera and windows cleanly on exit.

Future modes (train / evaluate / collect) can be wired in later by
dispatching on a --mode CLI flag; the structure below leaves room for
that without changing how prediction mode works today.

Usage:
    python app.py
"""

import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.utils.utils import load_config, setup_logger, load_model
from src.inference.predict import run_prediction_loop, attach_file_logging

logger = setup_logger("signbridge.app")


def load_app_model(config: dict):
    """
    Load the trained model once at startup, from the path configured
    in config.yaml (paths.models_final/model.joblib). Centralizing this
    in app.py means predict.py doesn't have to reload the model itself
    when app.py is the caller.
    """
    final_dir = config["paths"]["models_final"]
    model_path = os.path.join(final_dir, "model.joblib")
    logger.info(f"Loading trained model from '{model_path}'...")
    model = load_model(model_path)
    logger.info("Model loaded successfully.")
    return model


def run(config_path: str = "configs/config.yaml"):
    logger.info("Starting SignBridge-ML...")

    # --- 1. Load configuration ----------------------------------------
    try:
        config = load_config(config_path)
    except FileNotFoundError:
        logger.error(
            f"Configuration file not found at '{config_path}'. "
            "Make sure configs/config.yaml exists before running app.py."
        )
        sys.exit(1)
    except Exception as e:
        logger.error(f"Could not parse configuration file '{config_path}': {e}")
        sys.exit(1)

    # File logging for the whole app session (predict.py adds its own
    # handler too, but this captures startup/model-loading messages as
    # well, in case the camera or model fails before predict.py starts).
    logs_dir = config.get("paths", {}).get("outputs_logs", "outputs/logs")
    try:
        attach_file_logging(logger, logs_dir)
    except Exception as e:
        logger.warning(f"Could not set up file logging ({e}); continuing with console logs only.")

    # --- 2/3. Load the trained model -----------------------------------
    try:
        model = load_app_model(config)
    except FileNotFoundError as e:
        logger.error(str(e))
        logger.error("Train a model first: python src/models/train_model.py")
        sys.exit(1)
    except RuntimeError as e:
        # Covers a corrupted / unreadable model file.
        logger.error(str(e))
        sys.exit(1)

    # --- 4. Initialize camera + start the prediction loop --------------
    # (Camera setup happens inside run_prediction_loop, since it owns
    # the camera's full lifecycle -- open, read frames, release.)
    try:
        run_prediction_loop(config, model=model)
    except RuntimeError as e:
        # Covers "could not open webcam".
        logger.error(str(e))
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Interrupted by user. Shutting down.")
    except Exception as e:
        logger.error(f"Unexpected error during prediction: {e}")
        sys.exit(1)

    logger.info("SignBridge-ML exited cleanly.")


def main():
    parser = argparse.ArgumentParser(description="SignBridge-ML application entry point.")
    parser.add_argument("--config", default="configs/config.yaml",
                         help="Path to config.yaml (default: configs/config.yaml)")
    # Reserved for future modes; only 'predict' is implemented today.
    parser.add_argument("--mode", default="predict", choices=["predict"],
                         help="Application mode (only 'predict' is available for now).")
    args = parser.parse_args()

    if args.mode == "predict":
        run(args.config)


if __name__ == "__main__":
    main()
