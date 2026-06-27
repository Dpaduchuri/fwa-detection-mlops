
"""
fwa_service.py

BentoML v1.4+ service for FWA Detection.
Uses the new @bentoml.service decorator pattern instead of the
deprecated bentoml.Service + @svc.api approach.

Pattern adapted from BentoML's official XGBoost fraud detection example
(github.com/bentoml/Fraud-Detection-Model-Serving).
"""

import bentoml
import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd

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

CLASS_NAMES = ["ABUSE", "FRAUD", "LEGITIMATE", "WASTE"]


def save_model_to_bento():
    print("Loading model from MLflow...")
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name("fwa-detection")

    if experiment is None:
        raise ValueError("No fwa-detection experiment found. Run train_fwa.py first.")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.mlflow.runName != 'evaluation'",
        order_by=["start_time DESC"],
        max_results=1,
    )

    if not runs:
        raise ValueError("No training runs found. Run train_fwa.py first.")

    run_id = runs[0].info.run_id
    model = mlflow.xgboost.load_model(f"runs:/{run_id}/model")

    print("Saving model to BentoML model store...")
    saved_model = bentoml.xgboost.save_model(
        "fwa_xgboost",
        model,
        signatures={"predict_proba": {"batchable": True}},
        metadata={
            "feature_columns": FEATURE_COLUMNS,
            "class_names": CLASS_NAMES,
            "mlflow_run_id": run_id,
        },
    )
    print(f"Model saved: {saved_model.tag}")
    return saved_model


@bentoml.service(name="fwa_detection_service")
class FWAService:
    # loads the model once when the service starts,
    # not on every request - same principle as our FastAPI lifespan
    fwa_model = bentoml.models.get("fwa_xgboost:latest")

    def __init__(self):
        self.model = self.fwa_model.load_model()

    @bentoml.api()
    def predict(self, input_data: dict) -> dict:
        provider_id = input_data.get("provider_id", "unknown")
        features = {col: input_data[col] for col in FEATURE_COLUMNS}
        input_df = pd.DataFrame([features])[FEATURE_COLUMNS]

        proba = self.model.predict_proba(input_df)[0]
        predicted_class_idx = int(np.argmax(proba))
        predicted_label = CLASS_NAMES[predicted_class_idx]
        confidence = round(float(proba[predicted_class_idx]), 4)

        return {
            "provider_id": provider_id,
            "predicted_label": predicted_label,
            "confidence": confidence,
            "class_probabilities": {
                label: round(float(p), 4)
                for label, p in zip(CLASS_NAMES, proba)
            },
        }


if __name__ == "__main__":
    save_model_to_bento()
    print("Model saved. Now run:")
    print("  bentoml serve bentoml/fwa_service.py:FWAService --reload")
