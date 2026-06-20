"""
cms_ingestor.py

Reads raw CMS Medicare Physician & Other Practitioners data from Bronze
and writes a cleaned version to Silver as Parquet.

Source: CMS Data API
        data.cms.gov/data-api/v1/dataset/92396110-2aed-4d63-a6a2-5d6207d46a29/data

Note: this dataset is at the provider + procedure + place-of-service
level. Dollar fields are AVERAGES per service, not totals - so we
calculate total_billed and total_paid ourselves by multiplying by
total services. Found this out the hard way after assuming the field
names matched the old "by Provider" summary file, which they don't.
"""

import pandas as pd

# actual field names confirmed from the live API response,
# not from the CMS docs page which were a bit stale
RAW_COLUMNS = {
    "Rndrng_NPI": "provider_id",
    "Rndrng_Prvdr_Type": "specialty",
    "Rndrng_Prvdr_State_Abrvtn": "state",
    "HCPCS_Cd": "procedure_code",
    "HCPCS_Desc": "procedure_desc",
    "Tot_Benes": "total_beneficiaries",
    "Tot_Srvcs": "total_services",
    "Avg_Sbmtd_Chrg": "avg_billed_per_service",
    "Avg_Mdcr_Pymt_Amt": "avg_paid_per_service",
}

# anything past this is clearly bad data, not real billing.
# got this number from eyeballing the dataset, not official CMS guidance.
MAX_SERVICES_PER_PROVIDER = 50_000


def load_bronze(bronze_path: str) -> pd.DataFrame:
    """Reads the raw JSON pulled from the CMS data API."""
    print(f"Reading raw CMS file from {bronze_path}...")
    df = pd.read_json(bronze_path)
    print(f"Loaded {len(df):,} raw rows")
    return df


def clean_provider_ids(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.dropna(subset=["Rndrng_NPI"])
    dropped = before - len(df)
    if dropped:
        print(f"Dropped {dropped:,} rows with missing provider ID")
    return df


def clean_specialty(df: pd.DataFrame) -> pd.DataFrame:
    df["Rndrng_Prvdr_Type"] = df["Rndrng_Prvdr_Type"].fillna("Unknown")
    return df


def fix_procedure_codes(df: pd.DataFrame) -> pd.DataFrame:
    df["HCPCS_Cd"] = df["HCPCS_Cd"].astype(str).str.strip().str.upper()
    df["HCPCS_Cd"] = df["HCPCS_Cd"].str.replace("O", "0")

    before = len(df)
    df = df[df["HCPCS_Cd"].str.len() == 5]
    dropped = before - len(df)
    if dropped:
        print(f"Dropped {dropped:,} rows with invalid procedure codes")
    return df


def remove_bad_values(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = ["Tot_Srvcs", "Tot_Benes", "Avg_Sbmtd_Chrg", "Avg_Mdcr_Pymt_Amt"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    before = len(df)
    df = df.dropna(subset=["Tot_Srvcs", "Avg_Sbmtd_Chrg"])

    df = df[df["Tot_Srvcs"] > 0]
    df = df[df["Avg_Sbmtd_Chrg"] > 0]
    df = df[df["Tot_Srvcs"] < MAX_SERVICES_PER_PROVIDER]

    dropped = before - len(df)
    if dropped:
        print(f"Dropped {dropped:,} rows with invalid billing values")
    return df


def add_total_columns(df: pd.DataFrame) -> pd.DataFrame:
    # this dataset only gives averages per service, not running totals.
    # multiply by Tot_Srvcs to get what we need for billing_ratio later.
    df["total_billed"] = df["Avg_Sbmtd_Chrg"] * df["Tot_Srvcs"]
    df["total_paid"] = df["Avg_Mdcr_Pymt_Amt"] * df["Tot_Srvcs"]
    return df


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates(keep="first")
    dropped = before - len(df)
    if dropped:
        print(f"Dropped {dropped:,} exact duplicate rows")
    return df


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols_to_rename = {k: v for k, v in RAW_COLUMNS.items() if k in df.columns}
    return df.rename(columns=cols_to_rename)


def bronze_to_silver(bronze_path: str, silver_path: str) -> pd.DataFrame:
    """
    Main entry point. Takes a raw CMS JSON path, runs it through all
    the cleaning steps, writes the result to Silver as parquet.
    """
    df = load_bronze(bronze_path)
    raw_count = len(df)

    df = clean_provider_ids(df)
    df = clean_specialty(df)
    df = fix_procedure_codes(df)
    df = remove_bad_values(df)
    df = add_total_columns(df)
    df = drop_duplicates(df)
    df = rename_columns(df)

    clean_count = len(df)
    quality_pct = (clean_count / raw_count) * 100
    print(f"\nDone. {clean_count:,} clean rows out of {raw_count:,} raw "
          f"({quality_pct:.1f}% kept)")

    print(f"Writing Silver parquet to {silver_path}...")
    df.to_parquet(silver_path, index=False)

    return df


if __name__ == "__main__":
    bronze_to_silver(
        bronze_path="data/bronze/cms_az_2022.json",
        silver_path="data/silver/claims_az.parquet",
    )