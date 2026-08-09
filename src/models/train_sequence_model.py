"""
train_sequence_model.py
--------------------------
Trains a lightweight GRU classifier to distinguish J vs Z motion
sequences, using the .npz files produced by preprocess_sequences.py.

Saved separately from the static alphabet model:
    models/final/sequence_model.pt        (torch state_dict)
    models/final/sequence_model_meta.json (architecture + label config,
                                            needed to reconstruct the
                                            model before loading weights)

This does NOT touch models/final/model.joblib (the existing static
Random Forest alphabet model).

Usage:
    python src/models/train_sequence_model.py
    python src/models/train_sequence_model.py --config configs/config.yaml
"""

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.utils.utils import load_config, ensure_dir, setup_logger

logger = setup_logger(__name__)


class SignGRU(nn.Module):
    """
    Small GRU classifier: input_size=63 (flattened normalized landmarks
    per frame) -> GRU -> take final hidden state -> Linear -> logits.

    Kept deliberately small (default hidden_size=64, 1 layer) so
    inference stays fast and CPU-friendly for eventual Raspberry Pi
    deployment.
    """

    def __init__(self, input_size=63, hidden_size=64, num_layers=1,
                 num_classes=2, dropout=0.0):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        _, h_n = self.gru(x)          # h_n: (num_layers, batch, hidden_size)
        last_hidden = h_n[-1]          # (batch, hidden_size) -- final layer's hidden state
        return self.classifier(last_hidden)  # (batch, num_classes)


def load_split(processed_dir: str, split_name: str):
    data = np.load(os.path.join(processed_dir, f"{split_name}.npz"), allow_pickle=True)
    return data["X"], data["y"]


def encode_labels(y: np.ndarray, classes: list) -> np.ndarray:
    class_to_idx = {c: i for i, c in enumerate(classes)}
    return np.array([class_to_idx[label] for label in y], dtype=np.int64)


def run(config_path: str = "configs/config.yaml"):
    config = load_config(config_path)
    seq_cfg = config["sequence_project"]

    processed_dir = seq_cfg["paths"]["dataset_sequences_processed"]
    ckpt_dir = seq_cfg["paths"]["models_checkpoints"]
    final_dir = seq_cfg["paths"]["models_final"]
    ensure_dir(ckpt_dir)
    ensure_dir(final_dir)

    classes = sorted(seq_cfg["vocabulary"])  # ['j', 'z']
    model_cfg = seq_cfg["model"]
    train_cfg = seq_cfg["training"]

    torch.manual_seed(train_cfg["seed"])

    X_train, y_train_raw = load_split(processed_dir, "train")
    X_val, y_val_raw = load_split(processed_dir, "val")

    y_train = encode_labels(y_train_raw, classes)
    y_val = encode_labels(y_val_raw, classes)

    train_ds = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val))

    train_loader = DataLoader(train_ds, batch_size=train_cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=train_cfg["batch_size"], shuffle=False)

    device = torch.device("cpu")  # keep CPU-only: matches Raspberry Pi deployment target
    model = SignGRU(
        input_size=model_cfg["input_size"],
        hidden_size=model_cfg["hidden_size"],
        num_layers=model_cfg["num_layers"],
        num_classes=len(classes),
        dropout=model_cfg["dropout"],
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["learning_rate"])

    best_val_acc = -1.0
    best_state_dict = None

    logger.info(f"Training SignGRU on {len(X_train)} sequences for {train_cfg['epochs']} epochs...")

    for epoch in range(1, train_cfg["epochs"] + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        train_loss = total_loss / len(train_ds)

        model.eval()
        correct = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                preds = logits.argmax(dim=1)
                correct += (preds == yb).sum().item()
        val_acc = correct / len(val_ds)

        if epoch % 5 == 0 or epoch == train_cfg["epochs"]:
            logger.info(f"Epoch {epoch}/{train_cfg['epochs']} - train_loss={train_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save checkpoint of this run
    ckpt_path = os.path.join(ckpt_dir, f"sequence_model_{timestamp}.pt")
    torch.save(best_state_dict, ckpt_path)
    logger.info(f"Saved checkpoint: {ckpt_path}")

    # Promote best-of-run to models/final/ (distinct filenames -- does NOT
    # touch models/final/model.joblib, the existing static alphabet model)
    final_model_path = os.path.join(final_dir, "sequence_model.pt")
    final_meta_path = os.path.join(final_dir, "sequence_model_meta.json")

    torch.save(best_state_dict, final_model_path)
    with open(final_meta_path, "w") as f:
        json.dump({
            "input_size": model_cfg["input_size"],
            "hidden_size": model_cfg["hidden_size"],
            "num_layers": model_cfg["num_layers"],
            "num_classes": len(classes),
            "dropout": model_cfg["dropout"],
            "sequence_length": seq_cfg["sequence_length"],
            "classes": classes,
            "val_accuracy": best_val_acc,
            "trained_at": timestamp,
        }, f, indent=2)

    logger.info(
        f"Model promoted to {final_model_path} (best_val_acc={best_val_acc:.4f}). "
        f"Metadata: {final_meta_path}"
    )
    return best_val_acc


def main():
    parser = argparse.ArgumentParser(description="Train the J/Z sequence GRU classifier.")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()