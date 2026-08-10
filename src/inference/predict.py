"""
predict.py
-----------
Real-time sign prediction from the webcam, fusing TWO models:

    1. Static alphabet model  (models/final/model.joblib, Random Forest)
       -- recognizes A-Y from a single frame's hand landmarks.
    2. J/Z sequence model     (models/final/sequence_model.pt, GRU)
       -- recognizes J/Z from a sliding window of the last
          `sequence_length` frames (motion signs).

Pipeline per frame:

    1. Grab a frame from the webcam.
    2. Run MediaPipe hand detection ONCE (mediapipe_detector.HandDetector)
       to get 21 (x, y, z) landmarks for the first detected hand.
    3. Normalize those landmarks with utils.landmarks_to_feature_vector()
       (re-centers on the wrist, scales to unit size).
    4. Feed the (63,) feature vector to the Random Forest -> static
       prediction + confidence, EXACTLY as before. This never changes.
    5. Also push the same feature vector onto a sliding window buffer
       for the GRU. Once the buffer is full, run the GRU too.
    6. Fusion / gating: the GRU's guess is *only* accepted as the
       frame's final label if BOTH of the following hold, otherwise
       the RF's (smoothed) label is shown, same as the pre-GRU app:
           a) GRU confidence >= sequence_project.inference.confidence_threshold
           b) There is real motion in the buffered window (measured as
              mean frame-to-frame landmark movement) >= motion_threshold
       This matters because the GRU only has two classes ('j','z') --
       its softmax ALWAYS picks one of them, even when you're just
       holding a static letter still. Without the motion gate, the GRU
       would silently override every static prediction as soon as its
       buffer filled up, which is exactly what broke A-Y previously.
    7. Draw landmarks, predicted sign, confidence, source (static vs
       motion) and FPS on the frame and show it.

The loop keeps running until the user presses 'q' or closes the window.

Continuous learning: press 'c' at any point where a hand is detected
to flag the current STATIC prediction as wrong (this only affects the
A-Y / Random Forest model, same as before). You'll be prompted in the
terminal for the correct label, and the *raw* (un-normalized) landmarks
for that exact frame are saved to dataset/raw/<label>/ in the same
format collect_data.py produces.

Usage:
    python src/inference/predict.py
Can also be imported and driven by app.py via run_prediction_loop().
"""

import argparse
import json
import os
import sys
import time
import logging
from collections import deque, Counter
from datetime import datetime

import cv2
import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.inference.mediapipe_detector import HandDetector
from src.data.collect_data import landmarks_to_array
from src.models.train_sequence_model import SignGRU
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


# ---------------------------------------------------------------------------
# Static (Random Forest, A-Y) prediction -- UNCHANGED from the working
# pre-integration version. This is the piece that must keep working exactly
# as it did before.
# ---------------------------------------------------------------------------

def predict_sign(model, feature_vector: np.ndarray, confidence_threshold: float):
    """
    Run the classifier on a single (63,) feature vector.

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


# ---------------------------------------------------------------------------
# Sequence (GRU, J/Z) prediction -- mirrors predict_sequence.py's
# load_model()/predict_window(), just renamed so it can live alongside the
# static model's load_model()/predict_sign() in the same file.
# ---------------------------------------------------------------------------

def load_sequence_model(final_dir: str):
    """
    Load the promoted GRU (models/final/sequence_model.pt +
    sequence_model_meta.json). Raises FileNotFoundError if either file
    is missing, mirroring load_model()'s behavior for the RF model.
    """
    model_path = os.path.join(final_dir, "sequence_model.pt")
    meta_path = os.path.join(final_dir, "sequence_model_meta.json")

    if not os.path.exists(model_path) or not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"Sequence model not found at '{model_path}' / '{meta_path}'. "
            "Run src/models/train_sequence_model.py first."
        )

    with open(meta_path) as f:
        meta = json.load(f)

    model = SignGRU(
        input_size=meta["input_size"],
        hidden_size=meta["hidden_size"],
        num_layers=meta["num_layers"],
        num_classes=meta["num_classes"],
        dropout=meta["dropout"],
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model, meta


def predict_gru_window(model, window: np.ndarray, classes: list, confidence_threshold: float):
    """
    window: (sequence_length, 63). Same as predict_sequence.py's
    predict_window() -- the GRU ALWAYS argmaxes between its known
    classes (just 'j' and 'z'), so this alone is not sufficient to
    decide whether to show a J/Z prediction; see should_accept_gru_label().
    """
    with torch.no_grad():
        x = torch.from_numpy(window[None, ...]).float()  # (1, L, 63)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].numpy()

    idx = int(np.argmax(probs))
    confidence = float(probs[idx])
    label = classes[idx].upper() if confidence >= confidence_threshold else "Uncertain"
    return label, classes[idx].upper(), confidence


def compute_motion_score(window: np.ndarray) -> float:
    """
    Rough measure of how much movement happened across the buffered
    window of normalized (63,) feature vectors. J and Z are the only
    signs that require the hand to move while forming the sign; A-Y are
    held still. Held-still frames -> tiny frame-to-frame deltas ->
    low score. An actual J/Z motion -> much larger score.

    window: (sequence_length, 63)
    Returns: mean L2 distance between consecutive frames.
    """
    if window.shape[0] < 2:
        return 0.0
    deltas = np.diff(window, axis=0)                # (L-1, 63)
    frame_dists = np.linalg.norm(deltas, axis=1)     # (L-1,)
    return float(np.mean(frame_dists))


def should_accept_gru_label(gru_label: str, gru_confidence, motion_score: float,
                             motion_threshold: float) -> bool:
    """
    The gate described at the top of this file. The GRU's prediction is
    only trusted when it is itself confident AND the window shows real
    motion -- otherwise we defer to the static (RF) prediction.
    """
    if gru_label == "Uncertain" or gru_confidence is None:
        return False
    if motion_score < motion_threshold:
        return False
    return True


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def draw_overlay(frame, display_label, confidence, fps, hand_found: bool, source: str = "static"):
    """Draw the fused prediction, confidence, source and FPS text onto the frame."""
    if hand_found:
        conf_text = f"{confidence * 100:.1f}%" if confidence is not None else "N/A"
        color = (0, 255, 0) if source == "static" else (0, 200, 255)
        cv2.putText(frame, f"Sign: {display_label}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, f"Confidence: {conf_text}  ({source})", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
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

    Only used for the static (RF) model's vocabulary -- unchanged.
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


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_prediction_loop(config: dict, static_model=None, sequence_model=None, sequence_meta=None):
    """
    Open the webcam and continuously predict signs until the user quits,
    fusing the static RF model (A-Y) with the GRU sequence model (J/Z).

    Accepts already-loaded models (so app.py can load them once and hand
    them in) or loads whichever ones are missing itself.
    """
    logs_dir = config["paths"].get("outputs_logs", "outputs/logs")
    log_path = attach_file_logging(logger, logs_dir)
    logger.info(f"Logging this run to '{log_path}'")

    # --- Static model loading (A-Y) ------------------------------------
    if static_model is None:
        final_dir = config["paths"]["models_final"]
        model_path = os.path.join(final_dir, "model.joblib")
        logger.info(f"Loading static alphabet model from '{model_path}'...")
        static_model = load_model(model_path)  # raises FileNotFoundError / RuntimeError
    logger.info("Static alphabet model loaded successfully.")

    # --- Sequence model loading (J/Z) -----------------------------------
    if sequence_model is None or sequence_meta is None:
        seq_final_dir = config["sequence_project"]["paths"]["models_final"]
        logger.info(f"Loading J/Z sequence model from '{seq_final_dir}'...")
        sequence_model, sequence_meta = load_sequence_model(seq_final_dir)
    logger.info("J/Z sequence model loaded successfully.")

    seq_cfg = config["sequence_project"]
    sequence_length = seq_cfg["sequence_length"]
    gru_classes = sequence_meta["classes"]
    gru_confidence_threshold = seq_cfg["inference"]["confidence_threshold"]
    # Motion gate threshold: optional config key, sensible default if absent
    # so this works without forcing a config.yaml edit. Tune this in
    # configs/config.yaml under sequence_project.inference.motion_threshold
    # if J/Z is too eager or too reluctant to trigger for your setup.
    motion_threshold = seq_cfg["inference"].get("motion_threshold", 0.08)

    confidence_threshold = config.get("inference", {}).get("confidence_threshold", 0.6)
    smoothing_window = config.get("inference", {}).get("smoothing_window", 5)
    recent_predictions = deque(maxlen=smoothing_window)
    gru_buffer = deque(maxlen=sequence_length)

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

    logger.info("Webcam opened. Press 'q' to quit, 'c' to correct a wrong static prediction.")
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
                logger.error(f"Hand detection failed on this frame: {e}")
                landmarks, results = None, None

            display = detector.draw_landmarks(frame.copy(), results) if results else frame.copy()

            fused_label, fused_confidence, source = "No hand", None, "static"
            hand_found = False
            raw_static_label = None  # for the 'c' correction workflow below

            if landmarks is not None:
                hand_found = True
                try:
                    # Step 3: normalize landmarks (shared by both models)
                    feature_vector = landmarks_to_feature_vector(landmarks)

                    # Step 4: static (RF) prediction -- exactly as before
                    static_label, raw_static_label, static_confidence = predict_sign(
                        static_model, feature_vector, confidence_threshold
                    )

                    # Step 5: feed the GRU's sliding window
                    gru_buffer.append(feature_vector)

                    frame_label, frame_confidence, frame_source = static_label, static_confidence, "static"

                    if len(gru_buffer) == sequence_length:
                        window = np.stack(gru_buffer, axis=0)
                        gru_label, gru_raw_label, gru_confidence = predict_gru_window(
                            sequence_model, window, gru_classes, gru_confidence_threshold
                        )
                        motion_score = compute_motion_score(window)

                        # Step 6: gate -- only accept the GRU's J/Z guess
                        # when it's confident AND there's real motion.
                        if should_accept_gru_label(gru_label, gru_confidence, motion_score, motion_threshold):
                            frame_label, frame_confidence, frame_source = gru_label, gru_confidence, "motion"

                    recent_predictions.append(frame_label)

                    # Majority-vote smoothing over the last N *fused* frame
                    # decisions, so the on-screen label doesn't flicker
                    # between similar signs -- or between static/motion.
                    if len(recent_predictions) == smoothing_window:
                        fused_label = Counter(recent_predictions).most_common(1)[0][0]
                    else:
                        fused_label = frame_label
                    fused_confidence = frame_confidence
                    source = frame_source

                except Exception as e:
                    logger.error(f"Prediction failed on this frame: {e}")
                    fused_label = "Error"
            else:
                # Motion sign broke / no hand -- start a fresh GRU window,
                # same as predict_sequence.py does standalone.
                gru_buffer.clear()
                recent_predictions.clear()

            # FPS (optional, purely for on-screen display)
            now = time.time()
            fps = 1.0 / (now - prev_time) if now > prev_time else 0.0
            prev_time = now

            display = draw_overlay(display, fused_label, fused_confidence, fps, hand_found, source)
            cv2.imshow("SignBridge - Real-Time Prediction", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                logger.info("Quit key pressed. Stopping.")
                break
            elif key == ord("c"):
                if landmarks is not None:
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
    parser = argparse.ArgumentParser(description="Real-time SignBridge-ML prediction (A-Y + J/Z).")
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
        logger.error(str(e))  # missing model file(s)
    except RuntimeError as e:
        logger.error(str(e))  # camera / corrupted model
    except Exception as e:
        logger.error(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()