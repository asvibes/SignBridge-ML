"""
retrain.py
-----------
One-command retraining pipeline for SignBridge-ML:

    dataset/raw/*.npy
          |
          v
    src/data/preprocess.py        -> dataset/processed/{train,val,test}.csv
          |
          v
    src/models/train_model.py --force
                                   -> models/final/model.joblib (always promoted)
          |
          v
    src/models/evaluate.py        -> accuracy/F1/confusion matrix report

Run this after:
    - collecting new webcam samples for an existing letter,
    - adding a brand-new class to configs/config.yaml's `vocabulary`
      and collecting samples for it (see README.md), or
    - accumulating correction samples saved by predict.py's 'c' key.

Training is force-promoted here (unlike a plain `train_model.py` run)
because validation accuracy isn't directly comparable across runs where
the class list itself may have changed -- the point of retrain.py is
"make the live model reflect the current dataset/raw/, whatever that
now contains."

Usage:
    python retrain.py
    python retrain.py --config configs/config.yaml
"""

import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.data.preprocess import run as run_preprocess
from src.models.train_model import run as run_train
from src.models.evaluate import evaluate
from src.utils.utils import load_config, setup_logger

logger = setup_logger("signbridge.retrain")


def retrain(config_path: str = "configs/config.yaml") -> bool:
    logger.info("=" * 60)
    logger.info("SignBridge-ML retraining pipeline starting")
    logger.info("=" * 60)

    # --- 1. Preprocess ----------------------------------------------------
    logger.info("Step 1/3: preprocessing raw landmarks...")
    try:
        ok = run_preprocess(config_path)
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        return False
    if not ok:
        logger.error(
            "Preprocessing produced no data. Aborting retrain. Make sure "
            "dataset/raw/<label>/ has samples for signs in configs/config.yaml's "
            "vocabulary (via collect_data.py, convert_images_to_landmarks.py, "
            "or predict.py corrections)."
        )
        return False

    # --- 2. Train + force-promote ------------------------------------------
    logger.info("Step 2/3: training and promoting model...")
    try:
        val_acc, promoted = run_train(config_path, force=True)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return False

    if not promoted:
        # Shouldn't happen with force=True, but guard anyway.
        logger.error("Training completed but the model was not promoted. Aborting.")
        return False
    logger.info(f"Model promoted to models/final/model.joblib (val_accuracy={val_acc:.4f})")

    # --- 3. Evaluate the freshly promoted model -----------------------------
    logger.info("Step 3/3: evaluating promoted model on the test set...")
    try:
        config = load_config(config_path)
        evaluate(config)
    except Exception as e:
        # Evaluation failing doesn't undo the promotion -- the new model
        # is already live -- but this is surfaced loudly since it usually
        # means something's wrong (e.g. a missing/corrupt test split).
        logger.error(f"Evaluation failed: {e}")
        logger.warning(
            "models/final/model.joblib IS already updated to the new model "
            "despite the evaluation error above."
        )
        return False

    logger.info("=" * 60)
    logger.info("Retraining complete. models/final/model.joblib is up to date.")
    logger.info("=" * 60)
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Retrain SignBridge-ML end-to-end: preprocess -> train -> evaluate."
    )
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    success = retrain(args.config)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()