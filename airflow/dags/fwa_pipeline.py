
"""
fwa_pipeline.py

Airflow DAG that orchestrates the full FWA Detection pipeline end to end —
data ingestion through inference — on a weekly schedule.

The AWS reference repo uses AWS Step Functions for orchestration since
their pipeline lives entirely on SageMaker. We're using Airflow here
because it's the industry standard for local and hybrid pipeline
orchestration, and maps cleanly onto what Step Functions does on the
AWS side — same concept (define steps, set dependencies, schedule runs),
different execution environment.

When this eventually moves to AWS, this DAG either runs on Amazon MWAA
(Managed Workflows for Apache Airflow) or gets replaced by a Step
Functions state machine. The task definitions stay the same either way.

Pipeline order:
  ingest → features → label → train → evaluate → score

Each step only runs if the previous one succeeds — Airflow handles
this dependency tracking automatically once you wire up the >> operators
at the bottom.
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

from src.data_ingestion.cms_ingestor import bronze_to_silver
from src.feature_engineering.fwa_features import build_gold_features
from src.labeling.fwa_labeler import label_gold_data
from src.training.train_fwa import run_training
from src.evaluation.evaluate_fwa import run_evaluation
from src.inference.predict_fwa import load_model, score_providers

# these are the default settings applied to every task in the DAG unless
# a task explicitly overrides them
default_args = {
    "owner": "dhanush",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}

# schedule_interval="@weekly" means Airflow runs this every Monday at
# midnight automatically - no manual trigger needed. Change to
# "@daily" or a cron expression like "0 6 * * 1" as needed.
with DAG(
    dag_id="fwa_detection_pipeline",
    description="Weekly FWA detection pipeline - ingest through scoring",
    default_args=default_args,
    start_date=datetime(2025, 1, 1),
    schedule_interval="@weekly",
    catchup=False,
) as dag:

    # Step 1 - pull fresh CMS data and clean it to Silver
    # in production this would pull from S3 instead of a local file,
    # but the function signature stays identical
    ingest_task = PythonOperator(
        task_id="bronze_to_silver",
        python_callable=bronze_to_silver,
        op_kwargs={
            "bronze_path": "data/bronze/cms_az_2022.json",
            "silver_path": "data/silver/claims_az.parquet",
        },
    )

    # Step 2 - build provider-level features from Silver
    features_task = PythonOperator(
        task_id="build_gold_features",
        python_callable=build_gold_features,
        op_kwargs={
            "silver_path": "data/silver/claims_az.parquet",
            "gold_path": "data/gold/provider_features_az.parquet",
        },
    )

    # Step 3 - apply FRAUD/WASTE/ABUSE/LEGITIMATE labels
    label_task = PythonOperator(
        task_id="label_gold_data",
        python_callable=label_gold_data,
        op_kwargs={
            "gold_path": "data/gold/provider_features_az.parquet",
            "labeled_path": "data/gold/provider_labeled_az.parquet",
        },
    )

    # Step 4 - retrain XGBoost on freshly labeled data
    # this is the "self-improving" part - every weekly run trains on
    # whatever fresh data just came in, not on a static snapshot
    train_task = PythonOperator(
        task_id="train_model",
        python_callable=run_training,
        op_kwargs={
            "labeled_path": "data/gold/provider_labeled_az.parquet",
        },
    )

    # Step 5 - evaluate the newly trained model
    evaluate_task = PythonOperator(
        task_id="evaluate_model",
        python_callable=run_evaluation,
        op_kwargs={
            "labeled_path": "data/gold/provider_labeled_az.parquet",
        },
    )

    # Step 6 - score all current providers with the freshly trained model
    def score_all_providers():
        model = load_model()
        results = score_providers(
            model,
            gold_path="data/gold/provider_features_az.parquet",
        )
        print(f"Scored {len(results)} providers.")
        print(results["predicted_label"].value_counts())

    score_task = PythonOperator(
        task_id="score_providers",
        python_callable=score_all_providers,
    )

    # wire up the dependency chain - the >> operator means
    # "this task must complete successfully before the next one starts"
    # if any step fails, everything downstream stops and Airflow alerts you
    ingest_task >> features_task >> label_task >> train_task >> evaluate_task >> score_task
