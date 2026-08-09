"""
predict.py
-----------
Real-time sign prediction from the webcam.

Pipeline per frame (this is the "prediction pipeline" referenced
throughout this file's comments):

    1. Grab a frame from the webcam.
    2. Run MediaPipe hand detection (mediapipe_detector.HandDetector)
       to get 21 (x, y, z) landmarks for the first detected hand.
    3. Normalize those landmarks with utils.landmarks_to_feature_vector()
       -- this re-centers on the wrist and scales to unit size, so the
       model is robust to where in the frame the hand is and how close
       it is to the camera (see utils.normalize_landmarks docstring).
    4. Feed the resulting (63,) feature vector to the trained classifier
       to get a predicted sign + a confidence score.
    5. Draw the landmarks, predicted sign, confidence and FPS on the
       frame and show it.

The loop keeps running (re-predicting every frame) until the user
presses 'q' or closes the window.

Continuous learning: press 'c' at any point where a hand is detected
to flag the current prediction as wrong. You'll be prompted in the
terminal for the correct label, and the *raw* (un-normalized) landmarks
for that exact frame are saved to dataset/raw/<label>/ in the same
format collect_data.py produces -- so these correction samples get
picked up automatically the next time you run retrain.py.

Usage:
    python src/inference/predict.py
Can also be imported and driven by app.py via run_prediction_loop().
"""

import argparse
import os
import sys
import time
import logging
from collections import deque, Counter
from datetime import datetime

import cv2
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.inference.mediapipe_detector import HandDetector
from src.data.collect_data import landmarks_to_array
from src.utils.utils import (
    load_config,
    ensure_dir,
    setup_logger,
    landmarks_to_feature_vector,
    load_model,
    decode_label,
)

logger = setup_logger(__name__)


def attach_file_logging(logger: logging.Logger, logs_dir: str) -> str:
    """
    Add a file handler on top of the existing console logger so
    predict.py's run also gets written to outputs/logs/predict_<ts>.log.
    Returns the log file path.
    """
    ensure_dir(logs_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(logs_dir, f"predict_{timestamp}.log")

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(
        logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s", "%H:%M:%S")
    )
    logger.addHandler(file_handler)
    return log_path


def predict_sign(model, feature_vector: np.ndarray, confidence_threshold: float):
    """
    Run the classifier on a single (63,) feature vector.

    "Confidence calculation": for models that support predict_proba
    (random_forest and mlp always do; svm does when config.yaml sets
    model.svm.probability: true), confidence is simply the highest
    class probability the model assigns to this sample. For a model
    without predict_proba, we fall back to reporting the prediction
    with confidence=None (displayed as "N/A") rather than crashing.

    Returns:
        display_label: human-readable sign name, or "Uncertain" if the
                        top confidence is below confidence_threshold.
        raw_label:     the raw class label as predicted by the model.
        confidence:    float in [0, 1], or None if unavailable.
    """
    X = feature_vector.reshape(1, -1)

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        best_idx = int(np.argmax(proba))
        raw_label = model.classes_[best_idx]
        confidence = float(proba[best_idx])
    else:
        raw_label = model.predict(X)[0]
        confidence = None

    if confidence is not None and confidence < confidence_threshold:
        display_label = "Uncertain"
    else:
        display_label = decode_label(raw_label)

    return display_label, raw_label, confidence


def draw_overlay(frame, display_label, confidence, fps, hand_found: bool):
    """Draw the prediction, confidence and FPS text onto the frame."""
    if hand_found:
        conf_text = f"{confidence * 100:.1f}%" if confidence is not None else "N/A"
        cv2.putText(frame, f"Sign: {display_label}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Confidence: {conf_text}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "No hand detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.putText(frame, f"FPS: {fps:.1f}", (10, frame.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, "q = quit | c = correct", (frame.shape[1] - 220, frame.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return frame


def save_correction(landmarks, config: dict) -> None:
    """
    Save the current frame's RAW (21, 3) landmarks as a new training
    sample under dataset/raw/<label>/, in the exact same format
    collect_data.py and convert_images_to_landmarks.py produce.

    This blocks briefly on terminal input() while the correct label is
    typed -- the webcam loop simply resumes on the next frame after.
    """
    label = input("\nCorrect label for this frame (blank to cancel): ").strip().lower()
    if not label:
        logger.info("Correction cancelled.")
        return

    if label not in config["vocabulary"]:
        logger.warning(
            f"'{label}' is not currently in configs/config.yaml vocabulary. "
            "The sample will still be saved, but add it to 'vocabulary' before "
            "running retrain.py or it won't be included in training."
        )

    raw_dir = config["paths"]["dataset_raw"]
    out_dir = os.path.join(raw_dir, label)
    ensure_dir(out_dir)

    existing = len([f for f in os.listdir(out_dir) if f.endswith(".npy")])
    out_path = os.path.join(out_dir, f"sample_{existing:04d}.npy")

    arr = landmarks_to_array(landmarks)
    np.save(out_path, arr)
    logger.info(f"Saved correction sample to '{out_path}' ({existing + 1} total for '{label}').")


def run_prediction_loop(config: dict, model=None):
    """
    Open the webcam and continuously predict signs until the user quits.

    Accepts an already-loaded `model` (so app.py can load it once and
    hand it in) or loads it itself from models/final/model.joblib if
    not provided.
    """
    logs_dir = config["paths"].get("outputs_logs", "outputs/logs")
    log_path = attach_file_logging(logger, logs_dir)
    logger.info(f"Logging this run to '{log_path}'")

    # --- Model loading -----------------------------------------------
    if model is None:
        final_dir = config["paths"]["models_final"]
        model_path = os.path.join(final_dir, "model.joblib")
        logger.info(f"Loading model from '{model_path}'...")
        model = load_model(model_path)  # raises FileNotFoundError / RuntimeError
    logger.info("Model loaded successfully.")

    confidence_threshold = config.get("inference", {}).get("confidence_threshold", 0.6)
    smoothing_window = config.get("inference", {}).get("smoothing_window", 5)
    recent_predictions = deque(maxlen=smoothing_window)

    # --- Detector + camera setup --------------------------------------
    detector = HandDetector(
        max_num_hands=config["mediapipe"]["max_num_hands"],
        min_detection_confidence=config["mediapipe"]["min_detection_confidence"],
        min_tracking_confidence=config["mediapipe"]["min_tracking_confidence"],
        static_image_mode=config["mediapipe"]["static_image_mode"],
    )

    cap = cv2.VideoCapture(config["camera"]["device_index"])
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config["camera"]["frame_width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config["camera"]["frame_height"])

    if not cap.isOpened():
        detector.close()
        raise RuntimeError(
            "Could not open webcam. Check that a camera is connected and that "
            "camera.device_index in configs/config.yaml points to the right device."
        )

    logger.info("Webcam opened. Press 'q' to quit, 'c' to correct a wrong prediction.")
    prev_time = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                logger.error("Failed to read frame from webcam. Stopping.")
                break

            frame = cv2.flip(frame, 1)  # mirror for a natural selfie-view

            try:
                landmarks, results = detector.detect(frame)
            except Exception as e:
                # A single bad frame shouldn't crash the whole session --
                # log it and just skip prediction for this frame.
                logger.error(f"Hand detection failed on this frame: {e}")
                landmarks, results = None, None

            display = detector.draw_landmarks(frame.copy(), results) if results else frame.copy()

            display_label, confidence, hand_found = "No hand", None, False

            if landmarks is not None:
                hand_found = True
                try:
                    # Step 3: normalize landmarks (see utils.normalize_landmarks)
                    feature_vector = landmarks_to_feature_vector(landmarks)
                    # Step 4: predict + compute confidence
                    display_label, raw_label, confidence = predict_sign(
                        model, feature_vector, confidence_threshold
                    )
                    recent_predictions.append(display_label)

                    # Simple majority-vote smoothing over the last N frames
                    # so the on-screen label doesn't flicker between two
                    # similar signs frame-to-frame.
                    if len(recent_predictions) == smoothing_window:
                        display_label = Counter(recent_predictions).most_common(1)[0][0]

                except Exception as e:
                    logger.error(f"Prediction failed on this frame: {e}")
                    display_label = "Error"

            # FPS (optional, purely for on-screen display)
            now = time.time()
            fps = 1.0 / (now - prev_time) if now > prev_time else 0.0
            prev_time = now

            display = draw_overlay(display, display_label, confidence, fps, hand_found)
            cv2.imshow("SignBridge - Real-Time Prediction", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                logger.info("Quit key pressed. Stopping.")
                break
            elif key == ord("c"):
                if landmarks is not None:
                    # input() blocks the loop briefly -- that's fine, it's
                    # an intentional pause for a human correction workflow.
                    save_correction(landmarks, config)
                    prev_time = time.time()  # don't let the FPS reading spike after the pause
                else:
                    logger.warning("No hand detected right now -- nothing to correct.")

    except KeyboardInterrupt:
        logger.info("Interrupted by user (Ctrl+C). Stopping.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.close()
        logger.info("Camera released and windows closed. Clean shutdown complete.")


def main():
    parser = argparse.ArgumentParser(description="Real-time SignBridge-ML prediction.")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    try:
        config = load_config(args.config)
    except FileNotFoundError:
        logger.error(f"Config file not found at '{args.config}'.")
        return
    except Exception as e:
        logger.error(f"Could not parse config file '{args.config}': {e}")
        return

    try:
        run_prediction_loop(config)
    except FileNotFoundError as e:
        logger.error(str(e))  # missing model file
    except RuntimeError as e:
        logger.error(str(e))  # camera / corrupted model
    except Exception as e:
        logger.error(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()