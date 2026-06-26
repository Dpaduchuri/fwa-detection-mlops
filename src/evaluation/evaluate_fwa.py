"""
evaluate_fwa.py

Deeper evaluation of the trained FWA model beyond just top-line accuracy.

Evaluation approach adapted from the AWS fraud-detection-using-machine-learning
reference repo (aws-solutions-library-samples), which uses balanced accuracy,
Cohen's Kappa, confusion matrix, and per-class precision/recall/F1 as their
core evaluation suite. Their repo also uses ROC AUC and threshold tuning, but
those are binary-classification techniques - we have 4 classes, so we skip
those two specifically and keep everything else.

Cohen's Kappa is worth explaining since it's less commonly known:
regular accuracy can look great just because one class dominates the dataset.
Cohen's Kappa corrects for that - it measures how much better the model is
doing compared to just randomly guessing with the same class distribution.
Scores above 0.8 are generally considered strong for fraud detection work.

Same honest caveat as train_fwa.py applies here: with 114 providers total
and WASTE having 0 examples in the test set, these metrics are a pipeline
proof-of-concept, not a real production performance claim.
"""

import json

import mlflow
import mlflow.xgboost
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

FEATURE_COLUMNS = [
    "total_services",
    "total_billed",
    "total_paid",
    "max_billing_ratio",
    "avg_billing_ratio",
    "unique_procedures",
    "procedure_concentration",
    "procedures_per_day",
    "denial_proxy",
]

LABEL_COLUMN = "label"
TEST_SIZE = 0.2
RANDOM_STATE = 42


def load_and_prepare(labeled_path: str):
    print(f"Reading labeled data from {labeled_path}...")
    df = pd.read_parquet(labeled_path)

    X = df[FEATURE_COLUMNS]
    encoder = LabelEncoder()
    y = encoder.fit_transform(df[LABEL_COLUMN])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    return X_train, X_test, y_train, y_test, encoder


def train(X_train, y_train, num_classes: int) -> XGBClassifier:
    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)
    model = XGBClassifier(
        objective="multi:softprob",
        num_class=num_classes,
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train, sample_weight=sample_weights)
    return model


def evaluate(model, X_test, y_test, encoder: LabelEncoder) -> dict:
    preds = model.predict(X_test)

    # balanced accuracy weights each class equally regardless of size -
    # more honest than raw accuracy when classes are imbalanced
    bal_acc = balanced_accuracy_score(y_test, preds)

    # cohen's kappa measures how much better we're doing vs random chance -
    # borrowed directly from the AWS reference repo's evaluation approach.
    # above 0.8 is generally considered strong for fraud detection.
    kappa = cohen_kappa_score(y_test, preds)

    print(f"\nBalanced accuracy: {bal_acc:.3f}")
    print(f"Cohen's Kappa:     {kappa:.3f}")

    if kappa >= 0.8:
        print("→ Kappa looks strong (≥0.8)")
    elif kappa >= 0.6:
        print("→ Kappa is moderate (0.6–0.8), room to improve")
    else:
        print("→ Kappa is weak (<0.6), model needs work")

    print("\nPer-class precision / recall / F1:")
    report = classification_report(
        y_test,
        preds,
        labels=range(len(encoder.classes_)),
        target_names=encoder.classes_,
        zero_division=0,
    )
    print(report)

    print("Confusion matrix:")
    cm = confusion_matrix(y_test, preds)
    cm_df = pd.DataFrame(
        cm,
        index=[f"actual_{c}" for c in encoder.classes_],
        columns=[f"pred_{c}" for c in encoder.classes_],
    )
    print(cm_df)

    return {
        "balanced_accuracy": bal_acc,
        "cohen_kappa": kappa,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "class_names": encoder.classes_.tolist(),
    }


def run_evaluation(labeled_path: str):
    X_train, X_test, y_train, y_test, encoder = load_and_prepare(labeled_path)
    model = train(X_train, y_train, num_classes=len(encoder.classes_))

    mlflow.set_experiment("fwa-detection")

    with mlflow.start_run(run_name="evaluation"):
        metrics = evaluate(model, X_test, y_test, encoder)

        mlflow.log_metric("balanced_accuracy", metrics["balanced_accuracy"])
        mlflow.log_metric("cohen_kappa", metrics["cohen_kappa"])

        # log confusion matrix as a JSON artifact so it's retrievable later
        with open("confusion_matrix.json", "w") as f:
            json.dump({
                "matrix": metrics["confusion_matrix"],
                "classes": metrics["class_names"],
            }, f, indent=2)
        mlflow.log_artifact("confusion_matrix.json")

        print("\nEvaluation run logged to MLflow.")

    return metrics


if __name__ == "__main__":
    run_evaluation(labeled_path="data/gold/provider_labeled_az.parquet")