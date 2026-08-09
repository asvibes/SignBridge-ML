"""
collect_sequences.py
----------------------
Webcam tool for collecting J/Z motion-sign sequences.

Unlike collect_data.py (single static-pose frame per sample), this
records a variable-length burst of consecutive RAW (21, 3) landmark
frames while the user holds a "recording" toggle on, and saves the
whole burst as one .npy file of shape (T, 21, 3).

Why raw + variable length here (not normalized/fixed-length)?
Mirrors the project's existing raw-then-process convention (see
collect_data.py's docstring) -- so preprocess_sequences.py can change
the normalization/resampling strategy later without needing to
recollect from the webcam.

Controls:
    SPACE  - toggle recording on/off. Press once to start capturing a
             sequence, press again to stop and save it.
    q      - quit early (already-saved sequences are kept).

Usage:
    python src/data/collect_sequences.py --sign j
    python src/data/collect_sequences.py --sign z --sequences 200
"""

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.inference.mediapipe_detector import HandDetector
from src.data.collect_data import landmarks_to_array  # reuse existing conversion
from src.utils.utils import load_config, ensure_dir, setup_logger

logger = setup_logger(__name__)

MIN_FRAMES_TO_SAVE = 5  # discard accidental near-empty recordings


def collect(sign: str, target_sequences: int, config: dict):
    seq_cfg = config["sequence_project"]
    if sign not in seq_cfg["vocabulary"]:
        logger.warning(
            f"'{sign}' is not in configs/config.yaml sequence_project.vocabulary. "
            "Add it there before running preprocessing/training, or these "
            "sequences won't be picked up."
        )

    out_dir = os.path.join(seq_cfg["paths"]["dataset_sequences"], sign)
    ensure_dir(out_dir)

    existing = len([f for f in os.listdir(out_dir) if f.endswith(".npy")])
    logger.info(f"Found {existing} existing sequences for '{sign}'. New ones will be appended.")

    detector = HandDetector(
        max_num_hands=config["mediapipe"]["max_num_hands"],
        min_detection_confidence=config["mediapipe"]["min_detection_confidence"],
        min_tracking_confidence=config["mediapipe"]["min_tracking_confidence"],
        static_image_mode=False,  # tracking mode: motion signs need continuity
    )

    cap = cv2.VideoCapture(config["camera"]["device_index"])
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config["camera"]["frame_width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config["camera"]["frame_height"])

    if not cap.isOpened():
        logger.error("Could not open webcam. Check camera.device_index in configs/config.yaml.")
        detector.close()
        return

    recording = False
    current_seq = []       # list of (21, 3) raw arrays for the in-progress sequence
    saved = 0

    logger.info(f"Press SPACE to start/stop recording a '{sign}' sequence, 'q' to quit.")

    try:
        while saved < target_sequences:
            ok, frame = cap.read()
            if not ok:
                logger.error("Failed to read frame from webcam.")
                break

            frame = cv2.flip(frame, 1)
            landmarks, results = detector.detect(frame)
            display = detector.draw_landmarks(frame.copy(), results)

            if recording:
                if landmarks is not None:
                    current_seq.append(landmarks_to_array(landmarks))
                    status = f"RECORDING  frames={len(current_seq)}"
                    color = (0, 0, 255)
                else:
                    status = "RECORDING  (NO HAND - keep hand in frame)"
                    color = (0, 140, 255)
            else:
                status = "READY - press SPACE to record"
                color = (0, 255, 0)

            cv2.putText(display, f"Sign: {sign}  Saved: {saved}/{target_sequences}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(display, status, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(display, "SPACE = start/stop | q = quit", (10, 460),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow("SignBridge - J/Z Sequence Collection", display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord(" "):
                recording = not recording
                if recording:
                    current_seq = []
                    logger.info("Recording started.")
                else:
                    # Stopped: save if long enough, else discard.
                    if len(current_seq) >= MIN_FRAMES_TO_SAVE:
                        arr = np.stack(current_seq, axis=0)  # (T, 21, 3)
                        idx = existing + saved
                        out_path = os.path.join(out_dir, f"sequence_{idx:04d}.npy")
                        np.save(out_path, arr)
                        saved += 1
                        logger.info(f"Saved {out_path} ({arr.shape[0]} frames).")
                    else:
                        logger.warning(
                            f"Recording too short ({len(current_seq)} frames, "
                            f"need >= {MIN_FRAMES_TO_SAVE}). Discarded."
                        )
                    current_seq = []

    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.close()

    logger.info(f"Done. Saved {saved} new sequences for '{sign}'.")


def main():
    parser = argparse.ArgumentParser(description="Collect J/Z motion-sign sequences (webcam).")
    parser.add_argument("--sign", required=True, choices=["j", "z"],
                         help="Which motion sign to record.")
    parser.add_argument("--sequences", type=int, default=200,
                         help="Target number of sequences to collect this run (default: 200).")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    collect(args.sign, args.sequences, config)


if __name__ == "__main__":
    main()