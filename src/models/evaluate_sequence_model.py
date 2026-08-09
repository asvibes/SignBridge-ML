"""
evaluate_sequence_model.py
-----------------------------
Evaluates the promoted J/Z sequence model
(models/final/sequence_model.pt + sequence_model_meta.json) against
the held-out test split (dataset/sequences_processed/test.npz).

Reports accuracy, weighted precision/recall/F1, a per-class breakdown,
and saves a confusion matrix PNG to outputs/confusion_matrix/.

Usage:
    python src/models/evaluate_sequence_model.py
    python src/models/evaluate_sequence_model.py --config configs/config.yaml
"""

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.models.train_sequence_model import SignGRU, encode_labels
from src.utils.utils import load_config, ensure_dir, setup_logger

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
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model, meta


def evaluate(config: dict):
    seq_cfg = config["sequence_project"]
    processed_dir = seq_cfg["paths"]["dataset_sequences_processed"]
    final_dir = seq_cfg["paths"]["models_final"]
    cm_dir = seq_cfg["paths"]["outputs_confusion_matrix"]
    ensure_dir(cm_dir)

    model, meta = load_model(final_dir)
    classes = meta["classes"]

    data = np.load(os.path.join(processed_dir, "test.npz"), allow_pickle=True)
    X_test, y_test_raw = data["X"], data["y"]
    y_test = encode_labels(y_test_raw, classes)
    logger.info(f"Loaded {len(X_test)} test sequences.")

    with torch.no_grad():
        logits = model(torch.from_numpy(X_test).float())
        y_pred = logits.argmax(dim=1).numpy()

    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0
    )

    cm = confusion_matrix(y_test, y_pred, labels=list(range(len(classes))))
    report = classification_report(
        y_test, y_pred, labels=list(range(len(classes))),
        target_names=[c.upper() for c in classes], zero_division=0,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cm_path = os.path.join(cm_dir, f"seq_confusion_matrix_{timestamp}.png")

    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=range(len(classes)), yticks=range(len(classes)),
        xticklabels=[c.upper() for c in classes], yticklabels=[c.upper() for c in classes],
        ylabel="True sign", xlabel="Predicted sign", title="J/Z Sequence Model Confusion Matrix",
    )
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"), ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)

    print("\n" + "=" * 60)
    print("SignBridge-ML J/Z Sequence Model -- Evaluation Report")
    print("=" * 60)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}  (weighted)")
    print(f"Recall:    {recall:.4f}  (weighted)")
    print(f"F1 Score:  {f1:.4f}  (weighted)")
    print("-" * 60)
    print(report)
    print(f"Confusion matrix saved to: {cm_path}")
    print("=" * 60 + "\n")

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1}


def main():
    parser = argparse.ArgumentParser(description="Evaluate the J/Z sequence GRU model.")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    try:
        evaluate(config)
    except FileNotFoundError as e:
        logger.error(str(e))


if __name__ == "__main__":
    main()