"""
train_model.py
---------------
Trains a classifier (Random Forest / SVM / MLP, chosen via
configs/config.yaml -> model.type) on the processed landmark
features and saves it to models/checkpoints/. If it beats the
previous best validation accuracy, it's promoted to models/final/.

Usage:
    python src/models/train_model.py
    python src/models/train_model.py --config configs/config.yaml
    python src/models/train_model.py --force
        # Always promote to models/final/, even if validation accuracy
        # didn't beat the previous best. Used by retrain.py, since
        # accuracy isn't directly comparable across runs where the
        # vocabulary (class list) itself has changed.
"""

import argparse
import os
import sys
import json
from datetime import datetime

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.utils.utils import load_config, ensure_dir, setup_logger

logger = setup_logger(__name__)


def build_model(model_cfg: dict):
    """Factory: build the classifier chosen in config.yaml (model.type).

    Adding a new model later (e.g. a small neural net for dynamic
    signs) just means adding another branch here plus its params
    block in config.yaml -- nothing else in the pipeline changes.
    """
    model_type = model_cfg["type"]

    if model_type == "random_forest":
        params = model_cfg["random_forest"]
        return RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            random_state=42,
        )
    elif model_type == "svm":
        params = model_cfg["svm"]
        return SVC(kernel=params["kernel"], C=params["C"],
                    probability=params["probability"], random_state=42)
    elif model_type == "mlp":
        params = model_cfg["mlp"]
        return MLPClassifier(
            hidden_layer_sizes=tuple(params["hidden_layer_sizes"]),
            max_iter=params["max_iter"], random_state=42,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def load_split(processed_dir: str, split_name: str):
    df = pd.read_csv(os.path.join(processed_dir, f"{split_name}.csv"))
    X = df.drop(columns=["label"]).values
    y = df["label"].values
    return X, y


def run(config_path: str = "configs/config.yaml", force: bool = False):
    """
    Train a model and, if it beats the previous best validation accuracy
    (or if force=True), promote it to models/final/model.joblib.

    Returns:
        (val_accuracy, promoted) tuple.
    """
    config = load_config(config_path)
    processed_dir = config["paths"]["dataset_processed"]
    ckpt_dir = config["paths"]["models_checkpoints"]
    final_dir = config["paths"]["models_final"]
    ensure_dir(ckpt_dir)
    ensure_dir(final_dir)

    X_train, y_train = load_split(processed_dir, "train")
    X_val, y_val = load_split(processed_dir, "val")

    model = build_model(config["model"])
    logger.info(f"Training {config['model']['type']} on {len(X_train)} samples...")
    model.fit(X_train, y_train)

    val_preds = model.predict(X_val)
    val_acc = accuracy_score(y_val, val_preds)
    logger.info(f"Validation accuracy: {val_acc:.4f}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_path = os.path.join(ckpt_dir, f"model_{config['model']['type']}_{timestamp}.joblib")
    joblib.dump(model, ckpt_path)
    logger.info(f"Saved checkpoint: {ckpt_path}")

    # Promote to models/final/ only if it's the best validation accuracy
    # seen so far -- unless force=True, in which case always promote
    # (used by retrain.py after the vocabulary itself may have changed).
    best_meta_path = os.path.join(final_dir, "best_meta.json")
    best_acc = -1.0
    if os.path.exists(best_meta_path):
        with open(best_meta_path) as f:
            best_acc = json.load(f).get("val_accuracy", -1.0)

    promoted = False
    if force or val_acc >= best_acc:
        final_model_path = os.path.join(final_dir, "model.joblib")
        joblib.dump(model, final_model_path)
        with open(best_meta_path, "w") as f:
            json.dump({
                "val_accuracy": val_acc,
                "model_type": config["model"]["type"],
                "trained_at": timestamp,
                "vocabulary": config["vocabulary"],
            }, f, indent=2)
        promoted = True
        reason = "forced" if (force and val_acc < best_acc) else "beat previous best"
        logger.info(f"Model promoted to {final_model_path} ({reason}, val_acc={val_acc:.4f})")
    else:
        logger.info(f"Checkpoint did not beat best val_acc={best_acc:.4f}; not promoted.")

    return val_acc, promoted


def main():
    parser = argparse.ArgumentParser(description="Train the SignBridge-ML classifier.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--force", action="store_true",
                         help="Always promote to models/final/model.joblib, even if "
                              "validation accuracy didn't beat the previous best.")
    args = parser.parse_args()
    run(args.config, force=args.force)


if __name__ == "__main__":
    main()