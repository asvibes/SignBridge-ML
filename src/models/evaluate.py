"""
evaluate.py
------------
Evaluates the promoted model (models/final/model.joblib) against the
held-out test split (dataset/processed/test.csv).

Reports:
    - Accuracy
    - Precision / Recall / F1 (weighted average across signs, plus a
      full per-class breakdown via sklearn's classification_report)
    - A confusion matrix, printed as text and saved as a PNG under
      outputs/confusion_matrix/

Usage:
    python src/models/evaluate.py
    python src/models/evaluate.py --config configs/config.yaml
"""

import argparse
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless-safe backend (no display needed to save PNGs)
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.utils.utils import load_config, ensure_dir, setup_logger, load_model, decode_label

logger = setup_logger(__name__)


def load_test_split(processed_dir: str):
    """
    Load dataset/processed/test.csv, which preprocess.py wrote as:
        x0, y0, z0, ..., x20, y20, z20, label
    Returns (X_test, y_test) as numpy arrays.
    """
    test_path = os.path.join(processed_dir, "test.csv")
    if not os.path.exists(test_path):
        raise FileNotFoundError(
            f"No test split found at '{test_path}'. Run src/data/preprocess.py first."
        )

    df = pd.read_csv(test_path)
    if "label" not in df.columns:
        raise ValueError(f"'{test_path}' has no 'label' column -- is this the right file?")

    X_test = df.drop(columns=["label"]).values
    y_test = df["label"].values
    return X_test, y_test


def compute_metrics(y_true, y_pred):
    """
    Compute accuracy plus weighted precision/recall/F1.

    We use average="weighted" (rather than "macro") because sign classes
    may not be perfectly balanced (some signs may have more collected
    samples than others) -- weighted averaging accounts for class support.
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }


def plot_confusion_matrix(y_true, y_pred, class_labels, out_path: str):
    """
    Build a confusion matrix over class_labels and save it as a heatmap
    PNG at out_path. Rows = true label, columns = predicted label.
    """
    cm = confusion_matrix(y_true, y_pred, labels=class_labels)
    display_labels = [decode_label(c) for c in class_labels]

    fig_size = max(6, 0.6 * len(class_labels))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set(
        xticks=np.arange(len(class_labels)),
        yticks=np.arange(len(class_labels)),
        xticklabels=display_labels,
        yticklabels=display_labels,
        ylabel="True sign",
        xlabel="Predicted sign",
        title="SignBridge-ML Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Annotate each cell with its count, using a contrasting text color.
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=8,
            )

    fig.tight_layout()
    ensure_dir(os.path.dirname(out_path))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    return cm


def evaluate(config: dict):
    processed_dir = config["paths"]["dataset_processed"]
    final_dir = config["paths"]["models_final"]
    cm_dir = config["paths"].get("outputs_confusion_matrix", "outputs/confusion_matrix")

    model_path = os.path.join(final_dir, "model.joblib")

    logger.info(f"Loading model from '{model_path}'...")
    model = load_model(model_path)  # raises FileNotFoundError / RuntimeError with a clear message

    logger.info(f"Loading test split from '{processed_dir}'...")
    X_test, y_test = load_test_split(processed_dir)
    logger.info(f"Loaded {len(X_test)} test samples.")

    logger.info("Running predictions on the test set...")
    y_pred = model.predict(X_test)

    metrics = compute_metrics(y_test, y_pred)

    # Use the union of true + predicted labels (sorted) so the confusion
    # matrix stays well-formed even if a class is missing from the test
    # split or the model never predicts one of the classes.
    class_labels = sorted(set(y_test.tolist()) | set(y_pred.tolist()))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cm_path = os.path.join(cm_dir, f"confusion_matrix_{timestamp}.png")
    cm = plot_confusion_matrix(y_test, y_pred, class_labels, cm_path)
    logger.info(f"Saved confusion matrix image to '{cm_path}'")

    report = classification_report(
        y_test, y_pred,
        labels=class_labels,
        target_names=[decode_label(c) for c in class_labels],
        zero_division=0,
    )

    _print_report(metrics, cm, class_labels, report, model_path)

    return metrics, cm, report


def _print_report(metrics, cm, class_labels, report, model_path):
    print("\n" + "=" * 60)
    print("SignBridge-ML -- Evaluation Report")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Classes evaluated: {len(class_labels)}")
    print("-" * 60)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}  (weighted)")
    print(f"Recall:    {metrics['recall']:.4f}  (weighted)")
    print(f"F1 Score:  {metrics['f1_score']:.4f}  (weighted)")
    print("-" * 60)
    print("Per-class report:")
    print(report)
    print("-" * 60)
    print("Confusion matrix (rows=true, cols=predicted):")
    header = "        " + " ".join(f"{decode_label(c)[:6]:>6}" for c in class_labels)
    print(header)
    for label, row in zip(class_labels, cm):
        row_str = " ".join(f"{v:>6d}" for v in row)
        print(f"{decode_label(label)[:8]:<8}{row_str}")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate the trained SignBridge-ML model.")
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
        evaluate(config)
    except FileNotFoundError as e:
        logger.error(str(e))
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")


if __name__ == "__main__":
    main()
