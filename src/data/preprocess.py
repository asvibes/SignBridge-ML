"""
preprocess.py
--------------
Reads raw landmark samples from dataset/raw/<sign>/*.npy, normalizes
them, and produces train/val/test CSV files in dataset/processed/.

Each row in the output CSVs is:
    x0, y0, z0, x1, y1, z1, ..., x20, y20, z20, label

Usage:
    python src/data/preprocess.py
    python src/data/preprocess.py --config configs/config.yaml
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.utils.utils import load_config, ensure_dir, normalize_landmarks, setup_logger

logger = setup_logger(__name__)


def load_raw_dataset(raw_dir: str, vocabulary: list):
    """Load every .npy sample under dataset/raw/<sign>/ into (X, y) arrays."""
    features, labels = [], []

    for sign in vocabulary:
        sign_dir = os.path.join(raw_dir, sign)
        if not os.path.isdir(sign_dir):
            logger.warning(f"No data folder found for '{sign}' at {sign_dir}, skipping.")
            continue

        files = [f for f in os.listdir(sign_dir) if f.endswith(".npy")]
        if not files:
            logger.warning(f"No samples found for '{sign}'.")
            continue

        for fname in files:
            raw_landmarks = np.load(os.path.join(sign_dir, fname))  # shape (21, 3)
            feature_vec = normalize_landmarks(raw_landmarks)         # shape (63,)
            features.append(feature_vec)
            labels.append(sign)

        logger.info(f"Loaded {len(files)} samples for '{sign}'.")

    return np.array(features), np.array(labels)


def build_dataframe(X: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    """Combine features + labels into a single labeled DataFrame."""
    num_landmarks = X.shape[1] // 3
    columns = []
    for i in range(num_landmarks):
        columns += [f"x{i}", f"y{i}", f"z{i}"]

    df = pd.DataFrame(X, columns=columns)
    df["label"] = y
    return df


def run(config_path: str = "configs/config.yaml") -> bool:
    """
    Run preprocessing end-to-end: dataset/raw/* -> dataset/processed/*.csv.

    Returns:
        True if train/val/test CSVs were written, False if no raw data
        was found (e.g. nothing has been collected/converted yet).
    """
    config = load_config(config_path)
    raw_dir = config["paths"]["dataset_raw"]
    processed_dir = config["paths"]["dataset_processed"]
    ensure_dir(processed_dir)

    X, y = load_raw_dataset(raw_dir, config["vocabulary"])
    if len(X) == 0:
        logger.error("No data found. Run collect_data.py or convert_images_to_landmarks.py first.")
        return False

    df = build_dataframe(X, y)

    split_cfg = config["dataset_split"]
    seed = config["project"]["seed"]

    train_df, temp_df = train_test_split(
        df, test_size=(1 - split_cfg["train_ratio"]),
        stratify=df["label"], random_state=seed,
    )
    relative_val = split_cfg["val_ratio"] / (split_cfg["val_ratio"] + split_cfg["test_ratio"])
    val_df, test_df = train_test_split(
        temp_df, test_size=(1 - relative_val),
        stratify=temp_df["label"], random_state=seed,
    )

    train_df.to_csv(os.path.join(processed_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(processed_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(processed_dir, "test.csv"), index=False)

    logger.info(f"Saved train ({len(train_df)}), val ({len(val_df)}), "
                f"test ({len(test_df)}) rows to {processed_dir}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess raw landmark samples into train/val/test CSVs."
    )
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()