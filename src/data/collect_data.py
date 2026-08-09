"""
collect_data.py
-----------------
Interactive webcam-based data collection script.

For a given sign, this script captures `--samples` hand-landmark
samples and stores each as a .npy file (shape (21, 3), RAW /
un-normalized coordinates) under:

    dataset/raw/<sign_name>/sample_0000.npy
    dataset/raw/<sign_name>/sample_0001.npy
    ...

We deliberately store RAW landmarks here (not normalized, not images)
so that preprocess.py can change the normalization/feature strategy
later without needing to recollect data from the webcam.

Usage:
    python src/data/collect_data.py --sign a
    python src/data/collect_data.py --sign a --samples 80
    python src/data/collect_data.py                      # prompts for a label
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np

# Allow running this file directly (python src/data/collect_data.py)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.inference.mediapipe_detector import HandDetector
from src.utils.utils import load_config, ensure_dir, setup_logger

logger = setup_logger(__name__)


def landmarks_to_array(landmarks) -> np.ndarray:
    """Convert MediaPipe landmark list to a raw (21, 3) numpy array."""
    return np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)


def collect(sign: str, num_samples: int, config: dict):
    if sign not in config["vocabulary"]:
        logger.warning(
            f"'{sign}' is not in configs/config.yaml vocabulary. "
            "Add it there first if this is a new sign (e.g. an alphabet/number) "
            "before running retrain.py, or the samples collected here won't be "
            "included in training."
        )

    out_dir = os.path.join(config["paths"]["dataset_raw"], sign)
    ensure_dir(out_dir)

    existing = len([f for f in os.listdir(out_dir) if f.endswith(".npy")])
    logger.info(f"Found {existing} existing samples for '{sign}'. New ones will be appended.")

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
        logger.error("Could not open webcam. Check camera.device_index in configs/config.yaml.")
        return

    collected = 0
    logger.info(f"Press SPACE to capture a sample of '{sign}', 'q' to quit.")

    try:
        while collected < num_samples:
            ok, frame = cap.read()
            if not ok:
                logger.error("Failed to read frame from webcam.")
                break

            frame = cv2.flip(frame, 1)  # mirror for a natural selfie-view
            landmarks, results = detector.detect(frame)
            display = detector.draw_landmarks(frame.copy(), results)

            status = "HAND DETECTED" if landmarks else "NO HAND"
            cv2.putText(display, f"Sign: {sign}  Collected: {collected}/{num_samples}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(display, status, (10, 55), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0) if landmarks else (0, 0, 255), 2)
            cv2.putText(display, "SPACE = capture | q = quit", (10, 460),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow("SignBridge - Data Collection", display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord(" ") and landmarks is not None:
                arr = landmarks_to_array(landmarks)
                sample_idx = existing + collected
                out_path = os.path.join(out_dir, f"sample_{sample_idx:04d}.npy")
                np.save(out_path, arr)
                collected += 1
                logger.info(f"Saved {out_path}")
                time.sleep(0.15)  # small debounce so one keypress = one sample

    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.close()

    logger.info(f"Done. Collected {collected} new samples for '{sign}'.")


def main():
    parser = argparse.ArgumentParser(description="Collect sign landmark samples (webcam).")
    parser.add_argument("--sign", default=None,
                         help="Name of the sign (e.g. 'a'). If omitted, you'll be prompted "
                              "interactively -- this also lets you type a brand-new label "
                              "(e.g. a number or symbol) that isn't in vocabulary yet.")
    parser.add_argument("--samples", type=int, default=None,
                         help="Number of samples to collect (default: from config.yaml)")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)

    sign = args.sign
    if sign is None:
        print("Current vocabulary:", ", ".join(config["vocabulary"]))
        sign = input("Enter the label to collect samples for (existing or new): ").strip().lower()
        if not sign:
            logger.error("No label entered. Aborting.")
            return

    num_samples = args.samples or config["data_collection"]["samples_per_sign"]
    collect(sign, num_samples, config)


if __name__ == "__main__":
    main()