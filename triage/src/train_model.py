"""
Tri-Check Anemia Triage — Model Trainer
========================================
Loads `tri_check_master.csv`, trains a Deep Neural Network (DNN) for
100 epochs using Binary Crossentropy, and saves `models/deep_anemia_model.h5`.

Run: python src/train_model.py
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # non-interactive backend — safe in any environment
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import StandardScaler
from sklearn.utils            import class_weight
import joblib

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers, callbacks

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent.parent
CSV_PATH    = BASE_DIR / "tri_check_master.csv"
MODELS_DIR  = BASE_DIR / "models"
MODEL_PATH  = MODELS_DIR / "deep_anemia_model.h5"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
PLOT_PATH   = MODELS_DIR / "training_history.png"
MODELS_DIR.mkdir(exist_ok=True)

# ── Hyper-parameters ───────────────────────────────────────────────────────────
EPOCHS       = 100
BATCH_SIZE   = 32
LEARNING_RATE= 1e-3
DROPOUT_RATE = 0.35
L2_REG       = 1e-4
RANDOM_SEED  = 42

tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Feature columns (must match data_engine.py output)
FEATURE_COLS = ["rg_ratio", "lab_a", "mean_r", "mean_g", "mean_b"]


# ══════════════════════════════════════════════════════════════════════════════
# Data loading & preprocessing
# ══════════════════════════════════════════════════════════════════════════════

def load_data():
    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"Master CSV not found at {CSV_PATH}.\n"
            "Please run  python src/data_engine.py  first."
        )

    df = pd.read_csv(CSV_PATH)
    log.info(f"Loaded {len(df)} rows from {CSV_PATH.name}")
    log.info(f"Label distribution:\n{df['label'].value_counts().to_string()}")

    X = df[FEATURE_COLS].values.astype(np.float32)
    y = df["label"].values.astype(np.float32)
    return X, y


def split_and_scale(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_SEED, stratify=y
    )
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    joblib.dump(scaler, SCALER_PATH)
    log.info(f"Scaler saved → {SCALER_PATH}")
    return X_train, X_test, y_train, y_test, scaler


# ══════════════════════════════════════════════════════════════════════════════
# Model architecture
# ══════════════════════════════════════════════════════════════════════════════

def build_model(input_dim: int) -> keras.Model:
    """
    Sequential DNN with:
      • 3 dense blocks (Dense → BatchNorm → LeakyReLU → Dropout)
      • L2 regularisation on all Dense layers
      • Sigmoid output for binary classification
    """
    reg = regularizers.l2(L2_REG)

    model = keras.Sequential(
        [
            keras.Input(shape=(input_dim,), name="features"),

            # Block 1
            layers.Dense(128, kernel_regularizer=reg, name="dense_1"),
            layers.BatchNormalization(name="bn_1"),
            layers.LeakyReLU(negative_slope=0.1, name="act_1"),
            layers.Dropout(DROPOUT_RATE, name="drop_1"),

            # Block 2
            layers.Dense(64, kernel_regularizer=reg, name="dense_2"),
            layers.BatchNormalization(name="bn_2"),
            layers.LeakyReLU(negative_slope=0.1, name="act_2"),
            layers.Dropout(DROPOUT_RATE, name="drop_2"),

            # Block 3
            layers.Dense(32, kernel_regularizer=reg, name="dense_3"),
            layers.BatchNormalization(name="bn_3"),
            layers.LeakyReLU(negative_slope=0.1, name="act_3"),
            layers.Dropout(DROPOUT_RATE / 2, name="drop_3"),

            # Output
            layers.Dense(1, activation="sigmoid", name="output"),
        ],
        name="TriCheckDNN",
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")],
    )
    model.summary(print_fn=log.info)
    return model


# ══════════════════════════════════════════════════════════════════════════════
# Training callbacks
# ══════════════════════════════════════════════════════════════════════════════

def get_callbacks() -> list:
    return [
        callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=15,
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=7,
            min_lr=1e-6,
            verbose=1,
        ),
        callbacks.ModelCheckpoint(
            filepath=str(MODEL_PATH),
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
    ]


# ══════════════════════════════════════════════════════════════════════════════
# Plotting
# ══════════════════════════════════════════════════════════════════════════════

def plot_history(history):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    metrics = [("loss", "Loss"), ("accuracy", "Accuracy"), ("auc", "AUC")]

    for ax, (key, title) in zip(axes, metrics):
        ax.plot(history.history[key],     label="Train", linewidth=2)
        ax.plot(history.history[f"val_{key}"], label="Val",   linewidth=2, linestyle="--")
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle("Tri-Check DNN — Training History", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=120)
    log.info(f"Training plot saved → {PLOT_PATH}")
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation helpers
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(model, X_test, y_test):
    from sklearn.metrics import (
        classification_report, confusion_matrix, roc_auc_score
    )
    y_pred_prob = model.predict(X_test, verbose=0).ravel()
    y_pred      = (y_pred_prob >= 0.5).astype(int)

    log.info("\n─── Classification Report ───")
    log.info("\n" + classification_report(y_test, y_pred, target_names=["Healthy", "Anemic"]))

    cm = confusion_matrix(y_test, y_pred)
    log.info(f"Confusion Matrix:\n{cm}")

    auc = roc_auc_score(y_test, y_pred_prob)
    log.info(f"ROC-AUC: {auc:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    log.info("▶ Tri-Check Model Trainer starting …")

    # 1. Data
    X, y = load_data()
    X_train, X_test, y_train, y_test, _ = split_and_scale(X, y)
    log.info(f"Train: {len(X_train)}  |  Test: {len(X_test)}")

    # 2. Class weights (handle imbalance)
    cw = class_weight.compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    cw_dict = {i: w for i, w in enumerate(cw)}
    log.info(f"Class weights: {cw_dict}")

    # 3. Model
    model = build_model(input_dim=X_train.shape[1])

    # 4. Train
    log.info(f"Training for up to {EPOCHS} epochs …")
    history = model.fit(
        X_train, y_train,
        validation_split=0.15,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=cw_dict,
        callbacks=get_callbacks(),
        verbose=1,
    )

    # 5. Evaluate
    log.info("\n▶ Evaluating on held-out test set …")
    evaluate(model, X_test, y_test)

    # 6. Save & plot
    model.save(str(MODEL_PATH))
    log.info(f"✅ Model saved → {MODEL_PATH}")
    plot_history(history)

    log.info("Done! Launch the app with:  python app.py")


if __name__ == "__main__":
    main()
