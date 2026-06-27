import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd

FEATURE_COLUMNS = ["total_services","total_billed","total_paid","max_billing_ratio","avg_billing_ratio","unique_procedures","procedure_concentration","procedures_per_day","denial_proxy"]
CLASS_NAMES = ["ABUSE","FRAUD","LEGITIMATE","WASTE"]

def load_model(run_id=None):
    if run_id:
        model_uri = f"runs:/{run_id}/model"
    else:
        client = mlflow.tracking.MlflowClient()
        exp = client.get_experiment_by_name("fwa-detection")
        if exp is None:
            raise ValueError("No fwa-detection experiment found. Run train_fwa.py first.")
        runs = client.search_runs(experiment_ids=[exp.experiment_id], filter_string="tags.mlflow.runName != 'evaluation'", order_by=["start_time DESC"], max_results=1)
        if not runs:
            raise ValueError("No training runs found. Run train_fwa.py first.")
        run_id = runs[0].info.run_id
        print(f"Loading model from run: {run_id}")
        model_uri = f"runs:/{run_id}/model"
    model = mlflow.xgboost.load_model(model_uri)
    print("Model loaded successfully.")
    return model

def predict(model, provider_features):
    input_df = pd.DataFrame([provider_features])[FEATURE_COLUMNS]
    proba = model.predict_proba(input_df)[0]
    idx = int(np.argmax(proba))
    return {"predicted_label": CLASS_NAMES[idx], "confidence": round(float(proba[idx]), 4), "class_probabilities": {l: round(float(p), 4) for l, p in zip(CLASS_NAMES, proba)}}

def score_providers(model, gold_path):
    print(f"Loading provider features from {gold_path}...")
    df = pd.read_parquet(gold_path)
    print(f"Scoring {len(df):,} providers...")
    results = []
    for _, row in df.iterrows():
        r = predict(model, row[FEATURE_COLUMNS].to_dict())
        results.append({"provider_id": row["provider_id"], "specialty": row["specialty"], "predicted_label": r["predicted_label"], "confidence": r["confidence"], **r["class_probabilities"]})
    results_df = pd.DataFrame(results)
    print("\nPrediction distribution:")
    print(results_df["predicted_label"].value_counts())
    return results_df

if __name__ == "__main__":
    model = load_model()
    results = score_providers(model, gold_path="data/gold/provider_features_az.parquet")
    print("\nSample predictions (top 10 by confidence):")
    print(results.sort_values("confidence", ascending=False).head(10)[["provider_id","specialty","predicted_label","confidence"]].to_string(index=False))
