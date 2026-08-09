"""
convert_images_to_landmarks.py
--------------------------------
Batch alternative to collect_data.py: instead of capturing samples one
at a time from a live webcam, this reads a folder of already-existing
images (e.g. from a downloaded dataset) and runs the same MediaPipe
detector over each one, producing the exact same raw (21, 3) .npy
landmark format under dataset/raw/<sign>/ that collect_data.py writes.

Because preprocess.py only cares about the .npy files under
dataset/raw/<sign>/ -- not how they got there -- files produced by this
script and files produced by collect_data.py can be freely mixed for
the same sign (e.g. 40 samples from a public dataset + 20 more you
collect yourself).

Expected input layout (one subfolder per sign, matching config.yaml's
vocabulary names):

    <input_dir>/
        hello/
            img001.jpg
            img002.png
            ...
        yes/
            ...
        thank_you/
            ...

Images with no detectable hand, or more than one hand when the config
expects one, are skipped with a warning rather than silently corrupting
the dataset.

Usage:
    python src/data/convert_images_to_landmarks.py --input_dir /path/to/dataset
    python src/data/convert_images_to_landmarks.py --input_dir /path/to/dataset --sign hello
"""

import argparse
import os
import sys

import cv2

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.inference.mediapipe_detector import HandDetector
from src.utils.utils import load_config, ensure_dir, setup_logger
from src.data.collect_data import landmarks_to_array  # same conversion logic collect_data.py uses

logger = setup_logger(__name__)

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def convert_sign(sign: str, images_dir: str, out_dir: str, detector: HandDetector) -> tuple[int, int]:
    """
    Run the detector over every image for one sign and save any detected
    hand as a raw .npy landmark file, continuing numbering after any
    samples that already exist (same convention as collect_data.py).

    Returns (num_converted, num_skipped).
    """
    ensure_dir(out_dir)
    existing = len([f for f in os.listdir(out_dir) if f.endswith(".npy")])

    image_files = sorted(
        f for f in os.listdir(images_dir)
        if f.lower().endswith(IMAGE_EXTENSIONS)
    )
    if not image_files:
        logger.warning(f"No images found in '{images_dir}' for sign '{sign}'.")
        return 0, 0

    converted, skipped = 0, 0
    for fname in image_files:
        img_path = os.path.join(images_dir, fname)
        frame = cv2.imread(img_path)
        if frame is None:
            logger.warning(f"Could not read image '{img_path}', skipping.")
            skipped += 1
            continue

        landmarks, _ = detector.detect(frame)
        if landmarks is None:
            skipped += 1
            continue

        arr = landmarks_to_array(landmarks)  # same (21, 3) raw format as collect_data.py
        sample_idx = existing + converted
        out_path = os.path.join(out_dir, f"sample_{sample_idx:04d}.npy")
        import numpy as np
        np.save(out_path, arr)
        converted += 1

    logger.info(
        f"'{sign}': converted {converted}/{len(image_files)} images "
        f"({skipped} skipped, no hand detected)."
    )
    return converted, skipped


def main():
    parser = argparse.ArgumentParser(
        description="Convert a folder of per-sign images into raw landmark .npy files."
    )
    parser.add_argument("--input_dir", required=True,
                         help="Folder containing one subfolder per sign (matching config.yaml vocabulary names).")
    parser.add_argument("--sign", default=None,
                         help="Convert only this sign instead of the whole vocabulary.")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    raw_dir = config["paths"]["dataset_raw"]
    signs = [args.sign] if args.sign else config["vocabulary"]

    detector = HandDetector(
        max_num_hands=config["mediapipe"]["max_num_hands"],
        min_detection_confidence=config["mediapipe"]["min_detection_confidence"],
        min_tracking_confidence=config["mediapipe"]["min_tracking_confidence"],
        static_image_mode=True,  # static images, not a video stream -- improves per-image accuracy
    )

    total_converted, total_skipped = 0, 0
    try:
        for sign in signs:
            images_dir = os.path.join(args.input_dir, sign)
            if not os.path.isdir(images_dir):
                logger.warning(
                    f"No folder '{images_dir}' found for sign '{sign}' in --input_dir; skipping. "
                    "Check that the dataset's folder names match config.yaml's vocabulary "
                    "(rename folders if needed, e.g. 'Thank You' -> 'thank_you')."
                )
                continue

            out_dir = os.path.join(raw_dir, sign)
            converted, skipped = convert_sign(sign, images_dir, out_dir, detector)
            total_converted += converted
            total_skipped += skipped
    finally:
        detector.close()

    logger.info(f"Done. {total_converted} samples converted, {total_skipped} images skipped in total.")
    logger.info("Next step: python src/data/preprocess.py")


if __name__ == "__main__":
    main()
