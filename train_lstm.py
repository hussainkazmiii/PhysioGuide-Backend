from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from exercise_lstm_pipeline import (
    BASE_DIR,
    FEATURES_PER_FRAME,
    INDEX_TO_LABEL,
    LABELS,
    LABEL_TO_INDEX,
    METADATA_PATH,
    MODEL_PATH,
    SCALER_PATH,
    TIMESTEPS,
    SequenceStandardScaler,
    build_model_features,
    class_distribution,
    class_weights,
    classification_report_text,
    confusion_matrix_np,
    load_labeled_csvs,
    make_grouped_split,
    metadata_payload,
    reshape_flat_features,
)


DEFAULT_CSVS = (
    BASE_DIR / "chin_tuck.csv",
    BASE_DIR / "shoulder_abduction.csv",
)
TRAINING_LOG_PATH = BASE_DIR / "training_log.csv"
METRICS_PATH = BASE_DIR / "training_metrics.json"
REPORT_PATH = BASE_DIR / "classification_report.csv"
CONFUSION_MATRIX_PATH = BASE_DIR / "confusion_matrix.csv"
CURVES_PATH = BASE_DIR / "training_curves.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the PhysioGuide LSTM correctness classifier."
    )
    parser.add_argument(
        "--csv",
        nargs="+",
        type=Path,
        default=list(DEFAULT_CSVS),
        help="Input CSV files. Defaults to chin_tuck.csv and shoulder_abduction.csv.",
    )
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=7e-4)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument(
        "--group-size",
        type=int,
        default=25,
        help=(
            "Consecutive windows per split group. This reduces leakage from "
            "near-identical sliding windows crossing train/test splits."
        ),
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--model-out", type=Path, default=MODEL_PATH)
    parser.add_argument("--scaler-out", type=Path, default=SCALER_PATH)
    parser.add_argument("--metadata-out", type=Path, default=METADATA_PATH)
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Validate CSVs and preprocessing without importing TensorFlow.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip training_curves.png generation.",
    )
    return parser.parse_args()


def import_tensorflow():
    try:
        import tensorflow as tf
    except ImportError as exc:
        message = (
            "TensorFlow is not installed in this Python environment.\n"
            "Create/activate a Python environment for training, install requirements, "
            "then run this script again:\n\n"
            "  pip install -r requirements.txt\n"
            "  python train_lstm.py\n"
        )
        raise SystemExit(message) from exc
    return tf


def prepare_data(args: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, SequenceStandardScaler, Dict[str, object]]:
    X_flat, y, data = load_labeled_csvs(args.csv)
    X_seq = reshape_flat_features(X_flat)
    X_features = build_model_features(X_seq)

    train_idx, val_idx, test_idx = make_grouped_split(
        y=y,
        class_names=LABELS,
        group_size=args.group_size,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.random_state,
    )

    X_train = X_features[train_idx]
    y_train = y[train_idx]
    X_val = X_features[val_idx]
    y_val = y[val_idx]
    X_test = X_features[test_idx]
    y_test = y[test_idx]

    scaler = SequenceStandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    summary: Dict[str, object] = {
        "csv_files": [str(path) for path in args.csv],
        "total_samples": int(len(y)),
        "input_shape": [TIMESTEPS, FEATURES_PER_FRAME],
        "raw_class_distribution": class_distribution(y),
        "train_samples": int(len(y_train)),
        "val_samples": int(len(y_val)),
        "test_samples": int(len(y_test)),
        "train_class_distribution": class_distribution(y_train),
        "val_class_distribution": class_distribution(y_val),
        "test_class_distribution": class_distribution(y_test),
        "group_size": int(args.group_size),
        "label_to_index": LABEL_TO_INDEX,
    }

    print("Data summary")
    print(json.dumps(summary, indent=2))
    print("\nSample rows by exercise/label")
    print(data[["exercise", "label"]].value_counts().sort_index().to_string())

    return X_train, y_train, X_val, y_val, X_test, y_test, scaler, summary


def build_model(tf, learning_rate: float):
    regularizer = tf.keras.regularizers.l2(1e-4)
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(TIMESTEPS, FEATURES_PER_FRAME)),
            tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(
                    64,
                    return_sequences=True,
                    kernel_regularizer=regularizer,
                    recurrent_regularizer=regularizer,
                )
            ),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Dropout(0.35),
            tf.keras.layers.LSTM(
                32,
                kernel_regularizer=regularizer,
                recurrent_regularizer=regularizer,
            ),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Dropout(0.35),
            tf.keras.layers.Dense(32, activation="relu", kernel_regularizer=regularizer),
            tf.keras.layers.Dropout(0.20),
            tf.keras.layers.Dense(len(LABELS), activation="softmax"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    model.summary()
    return model


def train(args: argparse.Namespace) -> None:
    X_train, y_train, X_val, y_val, X_test, y_test, scaler, summary = prepare_data(args)
    if args.prepare_only:
        print("\nPreprocessing check passed. Re-run without --prepare-only to train.")
        return

    tf = import_tensorflow()
    tf.keras.utils.set_random_seed(args.random_state)

    model = build_model(tf, args.learning_rate)
    weights = class_weights(y_train)
    print("\nClass weights")
    print(json.dumps({INDEX_TO_LABEL[k]: round(v, 4) for k, v in weights.items()}, indent=2))

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=18,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=6,
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(args.model_out),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(str(TRAINING_LOG_PATH)),
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weight=weights,
        callbacks=callbacks,
        verbose=1,
    )

    # EarlyStopping restores the best in-memory weights. Save them once more so
    # the file always matches the model used for evaluation.
    model.save(args.model_out)
    scaler.save(args.scaler_out)
    args.metadata_out.write_text(
        json.dumps(metadata_payload(), indent=2),
        encoding="utf-8",
    )

    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    probabilities = model.predict(X_test, verbose=0)
    y_pred = np.argmax(probabilities, axis=1)
    cm = confusion_matrix_np(y_test, y_pred, len(LABELS))

    report = classification_report_text(y_test, y_pred)
    REPORT_PATH.write_text(report + "\n", encoding="utf-8")
    pd.DataFrame(cm, index=LABELS, columns=LABELS).to_csv(CONFUSION_MATRIX_PATH)

    metrics = {
        **summary,
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "model_path": str(args.model_out),
        "scaler_path": str(args.scaler_out),
        "metadata_path": str(args.metadata_out),
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    if not args.no_plots:
        save_training_curves(history)

    print("\nEvaluation")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print("\nClassification report")
    print(report)
    print("\nConfusion matrix")
    print(pd.DataFrame(cm, index=LABELS, columns=LABELS).to_string())

    print("\nArtifacts written")
    print(f"Model:     {args.model_out}")
    print(f"Scaler:    {args.scaler_out}")
    print(f"Metadata:  {args.metadata_out}")
    print(f"Metrics:   {METRICS_PATH}")
    print(f"Report:    {REPORT_PATH}")
    print(f"Confusion: {CONFUSION_MATRIX_PATH}")
    if not args.no_plots:
        print(f"Curves:    {CURVES_PATH}")


def save_training_curves(history) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping training_curves.png")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history.get("accuracy", []), label="train")
    axes[0].plot(history.history.get("val_accuracy", []), label="val")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(history.history.get("loss", []), label="train")
    axes[1].plot(history.history.get("val_loss", []), label="val")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(CURVES_PATH, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    try:
        train(args)
    except Exception as exc:
        print(f"\nTraining failed: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
