"""
launch_sagemaker.py

Launches a SageMaker Training Job using boto3 directly instead of
the sagemaker SDK - avoids pulling in heavy GPU dependencies (triton,
torch) that we don't need for XGBoost training on tabular data.

Same result as using the sagemaker SDK estimator, just more explicit
about what's actually happening under the hood.

Run from project root:
  python src/training/launch_sagemaker.py
"""

import boto3
import json
from datetime import datetime

ROLE_ARN = "arn:aws:iam::553752959016:role/FWASageMakerExecutionRole"
BUCKET = "fwa-detection-mlops-dhanush"
REGION = "us-east-1"

# timestamp keeps job names unique across runs
TIMESTAMP = datetime.now().strftime("%Y%m%d-%H%M%S")
JOB_NAME = f"fwa-detection-{TIMESTAMP}"

# s3 paths
TRAIN_DATA_S3 = f"s3://{BUCKET}/gold"
OUTPUT_S3 = f"s3://{BUCKET}/models"
SOURCE_S3 = f"s3://{BUCKET}/source"


def upload_training_script():
    """
    SageMaker needs the training script in S3 before it can run it.
    Upload it now so the training job can find it.
    """
    s3 = boto3.client("s3", region_name=REGION)
    print("Uploading training script to S3...")
    s3.upload_file(
        "src/training/train_sagemaker.py",
        BUCKET,
        "source/train_sagemaker.py",
    )
    print(f"Script uploaded to s3://{BUCKET}/source/train_sagemaker.py")


def launch_training_job():
    upload_training_script()

    sm = boto3.client("sagemaker", region_name=REGION)

    print(f"Launching training job: {JOB_NAME}")
    print(f"Input:  s3://{BUCKET}/gold")
    print(f"Output: {OUTPUT_S3}/{JOB_NAME}/output/model.tar.gz")

    response = sm.create_training_job(
        TrainingJobName=JOB_NAME,
        RoleArn=ROLE_ARN,

        # using the built-in sklearn container - has pandas, numpy,
        # scikit-learn, and xgboost preinstalled, no custom image needed
        AlgorithmSpecification={
            "TrainingImage": f"683313688378.dkr.ecr.{REGION}.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3",
            "TrainingInputMode": "File",
        },

        InputDataConfig=[
            {
                "ChannelName": "train",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": f"s3://{BUCKET}/gold",
                        "S3DataDistributionType": "FullyReplicated",
                    }
                },
                "ContentType": "application/x-parquet",
            }
        ],

        OutputDataConfig={
            "S3OutputPath": OUTPUT_S3,
        },

        ResourceConfig={
            "InstanceType": "ml.m5.large",
            "InstanceCount": 1,
            "VolumeSizeInGB": 5,
        },

        StoppingCondition={
            "MaxRuntimeInSeconds": 3600,
        },

        HyperParameters={
            "sagemaker_program": "train_sagemaker.py",
            "sagemaker_submit_directory": f"s3://{BUCKET}/source",
        },
    )

    print(f"Training job created: {response['TrainingJobArn']}")
    print("Waiting for job to complete...")

    # poll until the job finishes
    waiter = sm.get_waiter("training_job_completed_or_stopped")
    waiter.wait(TrainingJobName=JOB_NAME)

    # check final status
    result = sm.describe_training_job(TrainingJobName=JOB_NAME)
    status = result["TrainingJobStatus"]
    print(f"Job status: {status}")

    if status == "Completed":
        model_s3 = result["ModelArtifacts"]["S3ModelArtifacts"]
        print(f"Model saved to: {model_s3}")
        return model_s3
    else:
        failure = result.get("FailureReason", "Unknown")
        print(f"Job failed: {failure}")
        return None


if __name__ == "__main__":
    model_path = launch_training_job()