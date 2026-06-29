"""
train_sagemaker.py

SageMaker-compatible version of train_fwa.py. SageMaker runs this
inside a managed container - data comes in from S3 via environment
variables, model gets saved to /opt/ml/model/ which SageMaker then
uploads back to S3 automatically.

The actual training logic is identical to train_fwa.py - the only
difference is how we read input paths and where we write output.
That's intentional - keeping the training logic the same means we
can test it locally and trust it'll behave the same on SageMaker.
"""

import os
import json
import pickle

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score
from xgboost import XGBClassifier

# SageMaker passes these paths via environment variables
INPUT_PATH = os.environ.get("SM_CHANNEL_TRAIN", "data/gold")
OUTPUT_PATH = os.environ.get("SM_MODEL_DIR", "model")
OUTPUT_DATA_PATH = os.environ.get("SM_OUTPUT_DATA_DIR", "output")

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


def train():
    print(f"Reading labeled data from {INPUT_PATH}...")
    df = pd.read_parquet(
        os.path.join(INPUT_PATH, "provider_labeled_az.parquet")
    )
    print(f"Loaded {len(df):,} providers")

    X = df[FEATURE_COLUMNS]
    encoder = LabelEncoder()
    y = encoder.fit_transform(df[LABEL_COLUMN])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    sample_weights = compute_sample_weight(
        class_weight="balanced", y=y_train
    )

    model = XGBClassifier(
        objective="multi:softprob",
        num_class=len(encoder.classes_),
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train, sample_weight=sample_weights)

    preds = model.predict(X_test)
    bal_acc = balanced_accuracy_score(y_test, preds)
    kappa = cohen_kappa_score(y_test, preds)

    print(f"Balanced accuracy: {bal_acc:.3f}")
    print(f"Cohen's Kappa:     {kappa:.3f}")

    # save model to SageMaker output path
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    model_path = os.path.join(OUTPUT_PATH, "fwa_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to {model_path}")

    # save encoder so inference knows class names
    encoder_path = os.path.join(OUTPUT_PATH, "encoder.pkl")
    with open(encoder_path, "wb") as f:
        pickle.dump(encoder, f)
    print(f"Encoder saved to {encoder_path}")

    # save metrics as JSON for CloudWatch to pick up
    os.makedirs(OUTPUT_DATA_PATH, exist_ok=True)
    metrics = {
        "balanced_accuracy": bal_acc,
        "cohen_kappa": kappa,
    }
    with open(os.path.join(OUTPUT_DATA_PATH, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print("Metrics saved.")


if __name__ == "__main__":
    train()