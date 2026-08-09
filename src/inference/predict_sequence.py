"""
predict_sequence.py
----------------------
Real-time J/Z prediction using a sliding window of the last
`sequence_length` hand-detected frames. Every frame, the buffer is
fed through the GRU and a fresh prediction + confidence is shown.

Usage:
    python src/inference/predict_sequence.py
"""

import argparse
import json
import os
import sys
import time
from collections import deque, Counter

import cv2
import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.inference.mediapipe_detector import HandDetector
from src.models.train_sequence_model import SignGRU
from src.utils.utils import load_config, setup_logger, landmarks_to_feature_vector

logger = setup_logger(__name__)


def load_model(final_dir: str):
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


def predict_window(model, window: np.ndarray, classes: list, confidence_threshold: float):
    """window: (sequence_length, 63)"""
    with torch.no_grad():
        x = torch.from_numpy(window[None, ...]).float()  # (1, L, 63)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].numpy()

    idx = int(np.argmax(probs))
    confidence = float(probs[idx])
    label = classes[idx].upper() if confidence >= confidence_threshold else "Uncertain"
    return label, classes[idx].upper(), confidence


def run(config: dict):
    seq_cfg = config["sequence_project"]
    final_dir = seq_cfg["paths"]["models_final"]
    sequence_length = seq_cfg["sequence_length"]
    classes = seq_cfg["vocabulary"]  # will be overridden by meta["classes"] below for exact order
    infer_cfg = seq_cfg["inference"]

    model, meta = load_model(final_dir)
    classes = meta["classes"]

    detector = HandDetector(
        max_num_hands=config["mediapipe"]["max_num_hands"],
        min_detection_confidence=config["mediapipe"]["min_detection_confidence"],
        min_tracking_confidence=config["mediapipe"]["min_tracking_confidence"],
        static_image_mode=False,
    )

    cap = cv2.VideoCapture(config["camera"]["device_index"])
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config["camera"]["frame_width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config["camera"]["frame_height"])

    if not cap.isOpened():
        detector.close()
        raise RuntimeError("Could not open webcam. Check camera.device_index in configs/config.yaml.")

    buffer = deque(maxlen=sequence_length)
    recent_predictions = deque(maxlen=infer_cfg["smoothing_window"])
    prev_time = time.time()

    logger.info("Webcam opened. Press 'q' to quit.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                logger.error("Failed to read frame from webcam. Stopping.")
                break

            frame = cv2.flip(frame, 1)
            landmarks, results = detector.detect(frame)
            display = detector.draw_landmarks(frame.copy(), results)

            display_label, confidence, hand_found = "Buffering...", None, False

            if landmarks is not None:
                hand_found = True
                feature_vec = landmarks_to_feature_vector(landmarks)  # normalized (63,)
                buffer.append(feature_vec)

                if len(buffer) == sequence_length:
                    window = np.stack(buffer, axis=0)  # (L, 63)
                    label, raw_label, confidence = predict_window(
                        model, window, classes, infer_cfg["confidence_threshold"]
                    )
                    recent_predictions.append(label)
                    if len(recent_predictions) == infer_cfg["smoothing_window"]:
                        display_label = Counter(recent_predictions).most_common(1)[0][0]
                    else:
                        display_label = label
                else:
                    display_label = f"Buffering... {len(buffer)}/{sequence_length}"
            else:
                display_label = "No hand detected"
                buffer.clear()  # motion sign broke -- start a fresh window
                recent_predictions.clear()

            now = time.time()
            fps = 1.0 / (now - prev_time) if now > prev_time else 0.0
            prev_time = now

            conf_text = f"{confidence * 100:.1f}%" if confidence is not None else "N/A"
            cv2.putText(display, f"Sign: {display_label}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(display, f"Confidence: {conf_text}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(display, f"FPS: {fps:.1f}", (10, display.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(display, "q = quit", (display.shape[1] - 100, display.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow("SignBridge - J/Z Sequence Prediction", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                logger.info("Quit key pressed. Stopping.")
                break

    except KeyboardInterrupt:
        logger.info("Interrupted by user (Ctrl+C). Stopping.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.close()
        logger.info("Camera released and windows closed. Clean shutdown complete.")


def main():
    parser = argparse.ArgumentParser(description="Real-time J/Z sequence prediction.")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    try:
        run(config)
    except (FileNotFoundError, RuntimeError) as e:
        logger.error(str(e))


if __name__ == "__main__":
    main()