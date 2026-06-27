"""
api.py

FastAPI wrapper around predict_fwa.py - turns our local prediction
logic into a proper REST endpoint that external systems can call
over HTTP.

This is the local equivalent of the AWS reference repo's serving pattern:
    Their version:  HTTP request → API Gateway → Lambda → SageMaker endpoint
    Our version:    HTTP request → FastAPI (this file) → predict_fwa.py → response

Three endpoints:
    GET  /health       - confirms the API and model are running
    POST /predict      - scores a single provider, returns label + confidence
    GET  /predict/batch - scores all providers in the Gold feature table

When we eventually move to AWS, this FastAPI layer gets replaced by
API Gateway + Lambda, but predict_fwa.py stays untouched - that's
exactly why we kept the prediction logic separate from the serving layer.
"""

from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.inference.predict_fwa import load_model, predict, score_providers

# model gets loaded once when the server starts, not on every request -
# loading a model on every single incoming request would be extremely slow
# and is one of the most common beginner mistakes in ML serving
MODEL = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # runs once when the server starts up
    global MODEL
    print("Loading FWA model...")
    MODEL = load_model()
    print("Model ready. API is up.")
    yield
    # runs once when the server shuts down (Ctrl+C)
    print("API shutting down.")


app = FastAPI(
    title="FWA Detection API",
    description="Scores Medicare providers for Fraud, Waste, and Abuse",
    version="0.1.0",
    lifespan=lifespan,
)


# Pydantic model defines exactly what a valid request body looks like -
# FastAPI uses this to automatically validate incoming requests and reject
# malformed ones before they ever reach our prediction logic
class ProviderFeatures(BaseModel):
    provider_id: str
    total_services: float
    total_billed: float
    total_paid: float
    max_billing_ratio: float
    avg_billing_ratio: float
    unique_procedures: float
    procedure_concentration: float
    procedures_per_day: float
    denial_proxy: float


@app.get("/health")
def health_check():
    """
    Quick check that the API is running and the model loaded correctly.
    Useful for monitoring tools to ping periodically.
    """
    return {
        "status": "ok",
        "model_loaded": MODEL is not None,
    }


@app.post("/predict")
def predict_provider(features: ProviderFeatures):
    """
    Scores a single provider and returns their predicted FWA label
    plus confidence scores for all four classes.

    Adapted from the AWS reference repo pattern where Lambda invokes
    the SageMaker endpoint with a single transaction payload and gets
    back a fraud probability score.
    """
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    feature_dict = features.model_dump()
    feature_dict.pop("provider_id")

    result = predict(MODEL, feature_dict)

    return {
        "provider_id": features.provider_id,
        **result,
    }


@app.get("/predict/batch")
def predict_batch(gold_path: str = "data/gold/provider_features_az.parquet"):
    """
    Scores all providers in the Gold feature table at once.
    Returns a list of predictions sorted by confidence descending.

    Equivalent to the AWS reference repo's generate_endpoint_traffic.py
    which sends batches of transactions to the REST API for inference.
    """
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    try:
        results_df = score_providers(MODEL, gold_path)
        return {
            "total_providers": len(results_df),
            "predictions": results_df.sort_values(
                "confidence", ascending=False
            ).to_dict(orient="records"),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
