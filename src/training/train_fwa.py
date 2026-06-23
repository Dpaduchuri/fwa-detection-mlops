"""
train_fwa.py

Trains an XGBoost multi-class classifier on the labeled Gold data to
predict FRAUD / WASTE / ABUSE / LEGITIMATE, with MLflow tracking the
run.

Big caveat worth being honest about: we're training on 114 providers
total, and WASTE only has 2 examples. This is nowhere near enough data
for a model anyone should trust in production - the real value here is
proving the pipeline works end to end (ingestion -> features -> labels
-> training -> tracking), not the model's actual accuracy. Treat any
metrics this spits out as a sanity check, not a real performance claim.

Also not using `specialty` as a feature yet - it's categorical and
would need encoding (one-hot or target encoding), and with this little
data per category it'd probably just add noise right now. Flagging as
a TODO rather than doing it half-heartedly.
"""

import mlflow
import mlflow.xgboost
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

# numeric features only for v1 - see module docstring for why specialty
# isn't in here yet
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

# small dataset, so this split is more about proving the mechanics
# work than producing a statistically solid test set
TEST_SIZE = 0.2
RANDOM_STATE = 42


def load_labeled_data(labeled_path: str) -> pd.DataFrame:
    print(f"Reading labeled Gold data from {labeled_path}...")
    df = pd.read_parquet(labeled_path)
    print(f"Loaded {len(df):,} providers")
    return df


def split_features_and_labels(df: pd.DataFrame):
    X = df[FEATURE_COLUMNS]
    y_raw = df[LABEL_COLUMN]

    # XGBoost wants numeric class labels, not strings - this just maps
    # FRAUD/WASTE/ABUSE/LEGITIMATE to 0/1/2/3 and back
    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)

    return X, y, encoder


def train_test_split_data(X, y):
    # stratify=y keeps class proportions roughly consistent across train
    # and test. With WASTE only having 2 examples total, this is about
    # as small as stratification can go - 1 ends up in train, 1 in test.
    # Worth knowing this split is more symbolic than statistically
    # meaningful at this size, but it proves the mechanics work correctly.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train, num_classes: int) -> XGBClassifier:
    # keeping the model itself simple on purpose - this isn't the part
    # of the project meant to show off, the pipeline around it is
    model = XGBClassifier(
        objective="multi:softprob",
        num_class=num_classes,
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test, encoder: LabelEncoder):
    preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)

    print(f"\nTest accuracy: {accuracy:.3f}")
    print("\nClassification report:")
    # target_names lines the 0/1/2/3 predictions back up with readable
    # FRAUD/WASTE/ABUSE/LEGITIMATE labels for the printout
    report = classification_report(
        y_test,
        preds,
        labels=range(len(encoder.classes_)),
        target_names=encoder.classes_,
        zero_division=0,
    )
    print(report)

    return accuracy, report


def run_training(labeled_path: str):
    df = load_labeled_data(labeled_path)
    X, y, encoder = split_features_and_labels(df)
    X_train, X_test, y_train, y_test = train_test_split_data(X, y)

    mlflow.set_experiment("fwa-detection")

    with mlflow.start_run():
        mlflow.log_param("test_size", TEST_SIZE)
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_param("num_providers", len(df))
        mlflow.log_param("features", FEATURE_COLUMNS)

        model = train_model(X_train, y_train, num_classes=len(encoder.classes_))

        accuracy, report = evaluate_model(model, X_test, y_test, encoder)

        mlflow.log_metric("test_accuracy", accuracy)
        mlflow.xgboost.log_model(model, "model")

        print("\nMLflow run logged.")

    return model, encoder


if __name__ == "__main__":
    run_training(labeled_path="data/gold/provider_labeled_az.parquet")