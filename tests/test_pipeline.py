import pandas as pd
import numpy as np
import pytest
from src.feature_engineering.fwa_features import (
    calculate_row_level_billing_ratio,
    calculate_procedure_concentration,
)
from src.labeling.fwa_labeler import (
    apply_fraud_rule,
    apply_waste_rule,
    apply_abuse_rule,
)
from src.inference.predict_fwa import predict, CLASS_NAMES

# ── feature engineering tests ──────────────────────────────────────

def make_sample_silver():
    """minimal silver-layer dataframe for testing"""
    return pd.DataFrame({
        "provider_id": ["A", "A", "B"],
        "procedure_code": ["99213", "99214", "99213"],
        "total_services": [100.0, 50.0, 200.0],
        "total_billed": [5000.0, 3000.0, 8000.0],
        "total_paid": [3000.0, 2000.0, 5000.0],
        "avg_billed_per_service": [50.0, 60.0, 40.0],
        "Avg_Mdcr_Alowd_Amt": [10.0, 10.0, 20.0],
    })


def test_billing_ratio_calculated():
    df = make_sample_silver()
    result = calculate_row_level_billing_ratio(df)
    assert "billing_ratio" in result.columns
    # 50/10 = 5.0 for first row
    assert abs(result["billing_ratio"].iloc[0] - 5.0) < 0.001


def test_billing_ratio_handles_zero_allowed():
    """division by zero should produce NaN not inf"""
    df = make_sample_silver()
    df.loc[0, "Avg_Mdcr_Alowd_Amt"] = 0.0
    result = calculate_row_level_billing_ratio(df)
    assert np.isnan(result["billing_ratio"].iloc[0])


def test_procedure_concentration_single_code():
    """provider billing only one code should have concentration 1.0"""
    df = make_sample_silver()
    # provider B only bills 99213
    conc = calculate_procedure_concentration(df)
    provider_b = conc[conc["provider_id"] == "B"]["procedure_concentration"].values[0]
    assert abs(provider_b - 1.0) < 0.001


# ── labeling rule tests ────────────────────────────────────────────

def make_sample_gold():
    return pd.DataFrame({
        "provider_id": ["A", "B", "C"],
        "max_billing_ratio": [20.0, 5.0, 3.0],
        "procedures_per_day": [200.0, 10.0, 5.0],
        "denial_proxy": [0.95, 0.5, 0.3],
    })


def test_fraud_rule_flags_high_billing_ratio():
    df = make_sample_gold()
    result = apply_fraud_rule(df)
    assert result.iloc[0] == True   # ratio=20 > threshold 15
    assert result.iloc[1] == False  # ratio=5


def test_waste_rule_flags_high_daily_volume():
    df = make_sample_gold()
    result = apply_waste_rule(df)
    assert result.iloc[0] == True   # 200/day > 150 threshold
    assert result.iloc[1] == False  # 10/day


def test_abuse_rule_flags_high_denial_proxy():
    df = make_sample_gold()
    result = apply_abuse_rule(df)
    assert result.iloc[0] == True   # 0.95 > 0.90 threshold
    assert result.iloc[2] == False  # 0.30


# ── inference tests ────────────────────────────────────────────────

def make_dummy_model():
    """lightweight stand-in that returns fixed probabilities"""
    class DummyModel:
        def predict_proba(self, X):
            # always returns LEGITIMATE as most likely
            return np.array([[0.05, 0.05, 0.85, 0.05]])
    return DummyModel()


def test_predict_returns_expected_keys():
    model = make_dummy_model()
    features = {
        "total_services": 100.0,
        "total_billed": 5000.0,
        "total_paid": 3000.0,
        "max_billing_ratio": 5.0,
        "avg_billing_ratio": 3.0,
        "unique_procedures": 10.0,
        "procedure_concentration": 0.3,
        "procedures_per_day": 5.0,
        "denial_proxy": 0.4,
    }
    result = predict(model, features)
    assert "predicted_label" in result
    assert "confidence" in result
    assert "class_probabilities" in result


def test_predict_label_is_valid_class():
    model = make_dummy_model()
    features = {
        "total_services": 100.0, "total_billed": 5000.0,
        "total_paid": 3000.0, "max_billing_ratio": 5.0,
        "avg_billing_ratio": 3.0, "unique_procedures": 10.0,
        "procedure_concentration": 0.3, "procedures_per_day": 5.0,
        "denial_proxy": 0.4,
    }
    result = predict(model, features)
    assert result["predicted_label"] in CLASS_NAMES


def test_predict_confidence_between_0_and_1():
    model = make_dummy_model()
    features = {
        "total_services": 100.0, "total_billed": 5000.0,
        "total_paid": 3000.0, "max_billing_ratio": 5.0,
        "avg_billing_ratio": 3.0, "unique_procedures": 10.0,
        "procedure_concentration": 0.3, "procedures_per_day": 5.0,
        "denial_proxy": 0.4,
    }
    result = predict(model, features)
    assert 0.0 <= result["confidence"] <= 1.0
