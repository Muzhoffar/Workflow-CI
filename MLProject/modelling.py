"""
modelling.py (MLProject)
------------------------
Script training yang dijalankan via MLflow Project (mlflow run).
Dioptimalkan untuk CI/CD pipeline di GitHub Actions.

Artefak yang disimpan:
    - Model Random Forest
    - Confusion matrix plot
    - Classification report CSV
    - Feature importance plot

Cara menjalankan manual:
    mlflow run . -P n_estimators=100
"""

import os
import argparse
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)


# =============================================================================
# KONFIGURASI
# =============================================================================

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "dry_bean_preprocessing")
ARTIFACT_DIR = os.path.join(BASE_DIR, "temp_artifacts")

CLASS_NAMES = [
    "BARBUNYA", "BOMBAY", "CALI",
    "DERMASON", "HOROZ", "SEKER", "SIRA"
]


# =============================================================================
# ARGUMENT PARSER
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Training Dry Bean Classifier")
    parser.add_argument("--n_estimators",      type=int,   default=100)
    parser.add_argument("--max_depth",         type=str,   default="None")
    parser.add_argument("--min_samples_split", type=int,   default=2)
    parser.add_argument("--test_size",         type=float, default=0.2)
    parser.add_argument("--random_state",      type=int,   default=42)
    return parser.parse_args()


# =============================================================================
# LOAD DATA
# =============================================================================

def load_data():
    X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv"))
    X_test  = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv")).squeeze()
    y_test  = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv")).squeeze()

    print(f"Data berhasil dimuat.")
    print(f"  X_train : {X_train.shape}")
    print(f"  X_test  : {X_test.shape}")

    return X_train, X_test, y_train, y_test


# =============================================================================
# ARTEFAK: CONFUSION MATRIX
# =============================================================================

def save_confusion_matrix(y_test, y_pred, output_dir):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=ax,
        linewidths=0.5
    )
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    path = os.path.join(output_dir, "confusion_matrix.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# =============================================================================
# ARTEFAK: CLASSIFICATION REPORT
# =============================================================================

def save_classification_report(y_test, y_pred, output_dir):
    report_dict = classification_report(
        y_test, y_pred,
        target_names=CLASS_NAMES,
        output_dict=True
    )
    report_df = pd.DataFrame(report_dict).transpose().round(4)
    path = os.path.join(output_dir, "classification_report.csv")
    report_df.to_csv(path)
    return path


# =============================================================================
# ARTEFAK: FEATURE IMPORTANCE
# =============================================================================

def save_feature_importance(model, feature_names, output_dir):
    importances = model.feature_importances_
    indices     = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(
        range(len(importances)),
        importances[indices],
        color="steelblue",
        edgecolor="white"
    )
    ax.set_xticks(range(len(importances)))
    ax.set_xticklabels(
        [feature_names[i] for i in indices],
        rotation=45,
        ha="right"
    )
    ax.set_title("Feature Importance", fontsize=14, fontweight="bold")
    ax.set_xlabel("Fitur")
    ax.set_ylabel("Importance Score")
    plt.tight_layout()

    path = os.path.join(output_dir, "feature_importance.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# =============================================================================
# TRAINING
# =============================================================================

def train(args):
    # Parse max_depth
    max_depth = None if args.max_depth == "None" else int(args.max_depth)

    print("\nParameter training:")
    print(f"  n_estimators      : {args.n_estimators}")
    print(f"  max_depth         : {max_depth}")
    print(f"  min_samples_split : {args.min_samples_split}")
    print(f"  random_state      : {args.random_state}")

    # Load data
    X_train, X_test, y_train, y_test = load_data()

    # ------------------------------------------------------------------
    # Deteksi apakah dijalankan via "mlflow run" atau manual
    # Jika via "mlflow run": MLFLOW_RUN_ID sudah ada di environment,
    #   jangan panggil set_experiment() atau start_run() lagi
    # Jika manual: buat experiment dan run baru
    # ------------------------------------------------------------------
    run_from_mlflow_cli = os.environ.get("MLFLOW_RUN_ID") is not None

    if not run_from_mlflow_cli:
        mlflow.set_experiment("dry_bean_ci")
        mlflow.start_run(run_name="RandomForest_CI")

    # --- Training ---
    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=max_depth,
        min_samples_split=args.min_samples_split,
        random_state=args.random_state,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # --- Evaluasi ---
    y_pred          = model.predict(X_test)
    accuracy        = accuracy_score(y_test, y_pred)
    f1_macro        = f1_score(y_test, y_pred, average="macro")
    f1_weighted     = f1_score(y_test, y_pred, average="weighted")
    precision_macro = precision_score(y_test, y_pred, average="macro")
    recall_macro    = recall_score(y_test, y_pred, average="macro")

    print(f"\nHasil Evaluasi:")
    print(f"  Accuracy         : {accuracy:.4f}")
    print(f"  F1 Macro         : {f1_macro:.4f}")
    print(f"  F1 Weighted      : {f1_weighted:.4f}")
    print(f"  Precision Macro  : {precision_macro:.4f}")
    print(f"  Recall Macro     : {recall_macro:.4f}")

    # --- Manual Logging: Parameter ---
    mlflow.log_param("n_estimators",      args.n_estimators)
    mlflow.log_param("max_depth",         str(max_depth))
    mlflow.log_param("min_samples_split", args.min_samples_split)
    mlflow.log_param("random_state",      args.random_state)

    # --- Manual Logging: Metrik ---
    mlflow.log_metric("accuracy",        accuracy)
    mlflow.log_metric("f1_macro",        f1_macro)
    mlflow.log_metric("f1_weighted",     f1_weighted)
    mlflow.log_metric("precision_macro", precision_macro)
    mlflow.log_metric("recall_macro",    recall_macro)

    # --- Manual Logging: Model ---
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        registered_model_name="DryBean_CI"
    )

    # --- Manual Logging: Artefak Tambahan ---
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    cm_path     = save_confusion_matrix(y_test, y_pred, ARTIFACT_DIR)
    report_path = save_classification_report(y_test, y_pred, ARTIFACT_DIR)
    fi_path     = save_feature_importance(model, X_train.columns.tolist(), ARTIFACT_DIR)

    mlflow.log_artifact(cm_path,     artifact_path="plots")
    mlflow.log_artifact(report_path, artifact_path="reports")
    mlflow.log_artifact(fi_path,     artifact_path="plots")

    # Simpan run ID ke file agar bisa digunakan oleh workflow
    run_id = mlflow.active_run().info.run_id
    with open(os.path.join(BASE_DIR, "run_id.txt"), "w") as f:
        f.write(run_id)

    print(f"\nRun ID  : {run_id}")

    if not run_from_mlflow_cli:
        mlflow.end_run()


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 55)
    print(" MODELLING CI - DRY BEAN DATASET")
    print(" Random Forest | MLflow Project | GitHub Actions")
    print("=" * 55)

    args = parse_args()
    train(args)

    print("\n" + "=" * 55)
    print(" Training selesai.")
    print("=" * 55)


if __name__ == "__main__":
    main()