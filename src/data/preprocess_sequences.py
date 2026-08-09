"""
preprocess_sequences.py
-------------------------
Reads raw J/Z sequences from dataset/sequences/<sign>/*.npy (each of
shape (T, 21, 3), T variable), normalizes each frame, resamples every
sequence to a fixed length, and writes train/val/test .npz files to
dataset/sequences_processed/.

Resampling strategy (per sequence, applied AFTER per-frame normalization):
    - If the sequence is longer than sequence_length: uniformly sample
      sequence_length frame indices across it (keeps the overall motion
      shape rather than just clipping the end).
    - If shorter: pad by repeating the last frame (the sign's "settled"
      final handshape) until it reaches sequence_length.

Output files each contain:
    X: (num_samples, sequence_length, 63) float32
    y: (num_samples,) string array of labels ('j' / 'z')

Usage:
    python src/data/preprocess_sequences.py
    python src/data/preprocess_sequences.py --config configs/config.yaml
"""

import argparse
import os
import sys

import numpy as np
from sklearn.model_selection import train_test_split

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.utils.utils import load_config, ensure_dir, normalize_landmarks, setup_logger

logger = setup_logger(__name__)


def resize_sequence(frames: np.ndarray, target_len: int) -> np.ndarray:
    """
    frames: (T, 63) normalized+flattened per-frame feature vectors.
    Returns (target_len, 63).
    """
    t = frames.shape[0]
    if t == target_len:
        return frames
    if t > target_len:
        idx = np.round(np.linspace(0, t - 1, target_len)).astype(int)
        return frames[idx]
    # t < target_len: pad by repeating the last frame
    pad_block = np.repeat(frames[-1:], target_len - t, axis=0)
    return np.concatenate([frames, pad_block], axis=0)


def load_raw_sequences(sequences_dir: str, vocabulary: list, target_len: int):
    X, y = [], []

    for sign in vocabulary:
        sign_dir = os.path.join(sequences_dir, sign)
        if not os.path.isdir(sign_dir):
            logger.warning(f"No sequence folder found for '{sign}' at {sign_dir}, skipping.")
            continue

        files = [f for f in os.listdir(sign_dir) if f.endswith(".npy")]
        if not files:
            logger.warning(f"No sequences found for '{sign}'.")
            continue

        kept = 0
        for fname in files:
            raw_seq = np.load(os.path.join(sign_dir, fname))  # (T, 21, 3)
            if raw_seq.shape[0] == 0:
                logger.warning(f"Skipping empty sequence file: {fname}")
                continue

            # Normalize every frame, then flatten to (T, 63).
            normalized = np.stack(
                [normalize_landmarks(raw_seq[i]) for i in range(raw_seq.shape[0])],
                axis=0,
            )
            fixed = resize_sequence(normalized, target_len)
            X.append(fixed)
            y.append(sign)
            kept += 1

        logger.info(f"Loaded {kept} sequences for '{sign}'.")

    return np.array(X, dtype=np.float32), np.array(y)


def run(config_path: str = "configs/config.yaml") -> bool:
    config = load_config(config_path)
    seq_cfg = config["sequence_project"]

    sequences_dir = seq_cfg["paths"]["dataset_sequences"]
    processed_dir = seq_cfg["paths"]["dataset_sequences_processed"]
    target_len = seq_cfg["sequence_length"]
    vocabulary = seq_cfg["vocabulary"]
    train_cfg = seq_cfg["training"]

    ensure_dir(processed_dir)

    X, y = load_raw_sequences(sequences_dir, vocabulary, target_len)
    if len(X) == 0:
        logger.error("No sequence data found. Run collect_sequences.py first.")
        return False

    seed = train_cfg["seed"]
    val_ratio = train_cfg["val_ratio"]
    test_ratio = train_cfg["test_ratio"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(val_ratio + test_ratio), stratify=y, random_state=seed,
    )
    relative_val = val_ratio / (val_ratio + test_ratio)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1 - relative_val), stratify=y_temp, random_state=seed,
    )

    np.savez(os.path.join(processed_dir, "train.npz"), X=X_train, y=y_train)
    np.savez(os.path.join(processed_dir, "val.npz"), X=X_val, y=y_val)
    np.savez(os.path.join(processed_dir, "test.npz"), X=X_test, y=y_test)

    logger.info(
        f"Saved train ({len(X_train)}), val ({len(X_val)}), test ({len(X_test)}) "
        f"sequences to {processed_dir}. Shape per sample: ({target_len}, 63)."
    )
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess raw J/Z sequences into train/val/test .npz files."
    )
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()