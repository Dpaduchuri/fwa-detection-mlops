
"""
fwa_drift_report.py

Monitors feature drift between our training data (reference) and
any new provider data (current) using Evidently AI.

Drift detection approach:
- Uses PSI (Population Stability Index) as the primary method,
  which is the standard in healthcare and finance fraud detection.
  PSI < 0.1  → no significant drift, model still reliable
  PSI 0.1-0.25 → moderate drift, worth investigating
  PSI > 0.25  → significant drift, consider retraining

Why this matters for FWA detection:
Medicare billing patterns change over time - new procedure codes
get introduced, provider behavior shifts, fraud schemes evolve.
Without monitoring, the model silently degrades without anyone knowing.
This file is the early warning system that catches that drift before
it becomes a real problem.

In production (AWS), this would run on a schedule via MWAA and
push alerts to CloudWatch/SNS when drift thresholds are crossed.
Locally it runs as a standalone script and saves an HTML report.
"""

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

# the nine features our model actually uses - only monitoring these,
# not every column in the Gold table, since irrelevant columns would
# just add noise to the drift report
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

# PSI thresholds from the finance/healthcare fraud detection standard
PSI_NO_DRIFT = 0.1
PSI_MODERATE_DRIFT = 0.25


def load_reference_data(gold_path: str) -> pd.DataFrame:
    """
    Reference data = what the model was trained on.
    This is our Gold feature table from the original training run.
    """
    print(f"Loading reference data from {gold_path}...")
    df = pd.read_parquet(gold_path)
    return df[FEATURE_COLUMNS]


def load_current_data(current_path: str) -> pd.DataFrame:
    """
    Current data = new provider data coming in now.
    In production this would be freshly pulled from the CMS API.
    For local testing we simulate it by using a slice of the same
    Gold table - in a real scenario this would be a different
    time period's data.
    """
    print(f"Loading current data from {current_path}...")
    df = pd.read_parquet(current_path)
    return df[FEATURE_COLUMNS]


def run_drift_report(
    reference_path: str,
    current_path: str,
    output_dir: str = "monitoring",
) -> dict:
    """
    Runs the full Evidently drift report comparing reference vs current
    data. Saves an HTML report and returns a summary dict with drift
    status per feature.
    """
    reference_df = load_reference_data(reference_path)
    current_df = load_current_data(current_path)

    print(f"Reference: {len(reference_df)} providers")
    print(f"Current:   {len(current_df)} providers")

    # DataDriftPreset runs PSI on each feature automatically -
    # no need to manually configure a statistical test per column
    report = Report([
        DataDriftPreset(method="psi"),
    ])

    print("Running drift analysis...")
    my_eval = report.run(current_df, reference_df)

    # save HTML report for visual inspection
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(output_dir) / f"drift_report_{timestamp}.html"
    my_eval.save_html(str(output_path))
    print(f"HTML report saved to {output_path}")

    # extract summary from the report dict
    result_dict = my_eval.dict()

    # pull out per-feature drift results and flag anything concerning
    summary = {
        "timestamp": timestamp,
        "reference_size": len(reference_df),
        "current_size": len(current_df),
        "features_checked": FEATURE_COLUMNS,
        "drift_detected": False,
        "drifted_features": [],
        "recommendation": "No action needed",
    }

    print("\nDrift summary per feature:")
    print("-" * 50)

    # check the overall drift result
    metrics = result_dict.get("metrics", [])
    for metric in metrics:
        result = metric.get("result", {})

        # look for dataset-level drift result
        if "dataset_drift" in result:
            summary["drift_detected"] = result["dataset_drift"]

        # look for per-column drift results
        drift_by_columns = result.get("drift_by_columns", {})
        for col, col_result in drift_by_columns.items():
            drift_score = col_result.get("drift_score", 0)
            is_drifted = col_result.get("drift_detected", False)

            status = "DRIFTED" if is_drifted else "OK"
            print(f"  {col:<30} PSI={drift_score:.4f}  {status}")

            if is_drifted:
                summary["drifted_features"].append({
                    "feature": col,
                    "psi_score": drift_score,
                })

    print("-" * 50)

    # set recommendation based on what we found
    if len(summary["drifted_features"]) == 0:
        summary["recommendation"] = "No action needed - model inputs stable"
    elif len(summary["drifted_features"]) <= 2:
        summary["recommendation"] = (
            "Moderate drift detected - monitor closely, "
            "consider retraining if performance drops"
        )
    else:
        summary["recommendation"] = (
            "Significant drift detected across multiple features - "
            "retrain the model with fresh data"
        )

    print(f"\nRecommendation: {summary['recommendation']}")

    # also save summary as JSON for Airflow/automation to pick up
    json_path = Path(output_dir) / f"drift_summary_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"JSON summary saved to {json_path}")

    return summary


if __name__ == "__main__":
    # for local testing we compare the Gold table against itself -
    # in production this would be reference=training_data vs
    # current=new_batch_from_cms_api
    summary = run_drift_report(
        reference_path="data/gold/provider_features_az.parquet",
        current_path="data/gold/provider_features_az.parquet",
    )
    print("\nDrift detection complete.")
