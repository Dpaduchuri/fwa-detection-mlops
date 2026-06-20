"""
fwa_labeler.py

Takes the Gold feature table (one row per provider) and applies rule-based
labels: FRAUD, WASTE, ABUSE, or LEGITIMATE.

These aren't arbitrary cutoffs - they came from actually looking at the
percentile distribution and the real top-10 providers for each signal
on our AZ sample. See notes next to each threshold for the reasoning.

PRIORITY ORDER (when a provider trips more than one rule):
FRAUD > WASTE > ABUSE - this is ranked by how much we trust each signal,
not by assumed real-world severity (we don't have real fraud outcomes
to validate severity against).

- FRAUD (max_billing_ratio): strongest signal. Backed by a verified,
  individually-checked extreme outlier pattern - a single charge priced
  wildly above normal is hard to explain away as a coincidence.

- WASTE (procedures_per_day): also showed a real, visible gap in the
  data, but has more legitimate alternative explanations (e.g. some
  specialties like radiology naturally have higher throughput than
  others like oncology).

- ABUSE (denial_proxy): weakest signal. This is a PROXY built on
  billed-vs-paid gap, not actual denial data we don't have access to.
  It's also the noisiest of the three - nearly every provider sits
  high on this metric with no clean separation in the data.
"""

import pandas as pd

# billing_ratio: no single dramatic cliff in the data (unlike
# procedures_per_day), so this is closer to a judgment call. Landed on
# the ~90th percentile from our AZ sample (15.3) rounded to 15.
FRAUD_BILLING_RATIO_THRESHOLD = 15

# procedures_per_day: there's a real, visible gap here - top 3 providers
# (189-348/day) are clearly separated from everyone else (under 85/day).
# 150 sits cleanly in that gap.
WASTE_PROCEDURES_PER_DAY_THRESHOLD = 150

# denial_proxy: this dataset has no actual denial data, so this is a
# stand-in based on billed-vs-paid gap (see fwa_features.py docstring).
# Original guess of 0.4 was way too low - even the 25th percentile was
# 0.64. Recalibrated to ~90th percentile (0.90) after seeing the real
# distribution.
ABUSE_DENIAL_PROXY_THRESHOLD = 0.90


def apply_fraud_rule(df: pd.DataFrame) -> pd.Series:
    return df["max_billing_ratio"] > FRAUD_BILLING_RATIO_THRESHOLD


def apply_waste_rule(df: pd.DataFrame) -> pd.Series:
    return df["procedures_per_day"] > WASTE_PROCEDURES_PER_DAY_THRESHOLD


def apply_abuse_rule(df: pd.DataFrame) -> pd.Series:
    return df["denial_proxy"] > ABUSE_DENIAL_PROXY_THRESHOLD


def assign_labels(df: pd.DataFrame) -> pd.DataFrame:
    is_fraud = apply_fraud_rule(df)
    is_waste = apply_waste_rule(df)
    is_abuse = apply_abuse_rule(df)

    # applied weakest signal first so it gets overwritten by anything
    # stronger - see module docstring for why this order specifically
    df["label"] = "LEGITIMATE"
    df.loc[is_abuse, "label"] = "ABUSE"
    df.loc[is_waste, "label"] = "WASTE"
    df.loc[is_fraud, "label"] = "FRAUD"

    return df


def label_gold_data(gold_path: str, labeled_path: str) -> pd.DataFrame:
    print(f"Reading Gold data from {gold_path}...")
    df = pd.read_parquet(gold_path)
    print(f"Loaded {len(df):,} providers")

    print("Applying FRAUD / WASTE / ABUSE rules...")
    df = assign_labels(df)

    print("\nLabel distribution:")
    print(df["label"].value_counts())

    print(f"\nWriting labeled data to {labeled_path}...")
    df.to_parquet(labeled_path, index=False)

    return df


if __name__ == "__main__":
    label_gold_data(
        gold_path="data/gold/provider_features_az.parquet",
        labeled_path="data/gold/provider_labeled_az.parquet",
    )