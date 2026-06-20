"""
fwa_features.py

Takes cleaned Silver layer claims data (one row per provider + procedure +
place of service) and builds the Gold layer feature table - one row per
provider, with FWA-relevant signals calculated.

Two things worth remembering about this file:
1. billing_ratio gets calculated per ROW first, before any aggregation -
   otherwise averaging across a provider's procedures hides exactly the
   single overpriced charge we're trying to catch.
2. This dataset has no denial data, so denial_proxy is a stand-in based
   on billed vs paid gap, not a true denial rate. Worth remembering if
   this ever gets compared against real claims data with actual denial
   codes down the line.
"""

import numpy as np
import pandas as pd

WORKING_DAYS_PER_YEAR = 240  # rough estimate, not an exact CMS number


def calculate_row_level_billing_ratio(df: pd.DataFrame) -> pd.DataFrame:
    # catches "this ONE charge was priced way above the allowed amount"
    # before any aggregation has a chance to wash it out
    df["billing_ratio"] = df["avg_billed_per_service"] / df["Avg_Mdcr_Alowd_Amt"]
    df["billing_ratio"] = df["billing_ratio"].replace([np.inf, -np.inf], np.nan)
    return df


def calculate_procedure_concentration(df: pd.DataFrame) -> pd.DataFrame:
    # for each provider: what fraction of their total volume comes from
    # their single most-billed procedure code. Catches the "does the same
    # thing over and over" pattern even before looking at raw volume.
    provider_totals = df.groupby("provider_id")["total_services"].sum()
    procedure_totals = (
        df.groupby(["provider_id", "procedure_code"])["total_services"]
        .sum()
    )

    top_procedure = procedure_totals.groupby("provider_id").max()
    concentration = (top_procedure / provider_totals).rename(
        "procedure_concentration"
    )

    return concentration.reset_index()


def aggregate_to_provider_level(df: pd.DataFrame) -> pd.DataFrame:
    # rolling up to one row per provider - but only AFTER we've already
    # captured the row-level red flags above
    provider_df = df.groupby("provider_id").agg(
        specialty=("specialty", "first"),
        state=("state", "first"),
        total_services=("total_services", "sum"),
        total_billed=("total_billed", "sum"),
        total_paid=("total_paid", "sum"),
        max_billing_ratio=("billing_ratio", "max"),
        avg_billing_ratio=("billing_ratio", "mean"),
        unique_procedures=("procedure_code", "nunique"),
    ).reset_index()

    return provider_df


def calculate_provider_level_features(df: pd.DataFrame) -> pd.DataFrame:
    # procedures_per_day needs a TOTAL across everything the provider did -
    # this is the one feature that genuinely can't be calculated row by row
    df["procedures_per_day"] = df["total_services"] / WORKING_DAYS_PER_YEAR

    # no denial data in this dataset, so this is our best proxy - flagging
    # clearly in comments since it's not a real denial rate
    df["denial_proxy"] = 1 - (df["total_paid"] / df["total_billed"])
    df["denial_proxy"] = df["denial_proxy"].clip(lower=0)

    return df


def build_gold_features(silver_path: str, gold_path: str) -> pd.DataFrame:
    print(f"Reading Silver data from {silver_path}...")
    df = pd.read_parquet(silver_path)
    print(f"Loaded {len(df):,} rows")

    print("Calculating row-level billing ratios...")
    df = calculate_row_level_billing_ratio(df)

    print("Calculating procedure concentration...")
    concentration_df = calculate_procedure_concentration(df)

    print("Aggregating to provider level...")
    provider_df = aggregate_to_provider_level(df)
    provider_df = provider_df.merge(concentration_df, on="provider_id", how="left")

    print("Calculating provider-level features...")
    provider_df = calculate_provider_level_features(provider_df)

    print(f"\nDone. {len(provider_df):,} providers in Gold feature table")

    print(f"Writing Gold parquet to {gold_path}...")
    provider_df.to_parquet(gold_path, index=False)

    return provider_df


if __name__ == "__main__":
    build_gold_features(
        silver_path="data/silver/claims_az.parquet",
        gold_path="data/gold/provider_features_az.parquet",
    )