
import pandas as pd

file_path = "dataset/india_inr_loans.csv"

df = pd.read_csv(file_path)

print("=" * 60)
print("MICROSEGMENT - FEATURE REVIEW")
print("=" * 60)

# 1. Count records in each sector
print("\n1. RECORDS BY SECTOR")
print(df["sector"].value_counts())

# 2. Top activities in each sector
print("\n2. TOP ACTIVITIES BY SECTOR")

for sector in df["sector"].unique():
    print(f"\n--- {sector} ---")

    activities = (
        df[df["sector"] == sector]["activity"]
        .value_counts()
        .head(5)
    )

    print(activities)

# 3. Inspect actual loan purposes
print("\n3. SAMPLE LOAN PURPOSES BY SECTOR")

for sector in df["sector"].unique():
    print(f"\n--- {sector} ---")

    purposes = (
        df[df["sector"] == sector]["use"]
        .dropna()
        .drop_duplicates()
        .head(3)
    )

    for purpose in purposes:
        print("-", purpose)

# 4. Number of unique activities
print("\n4. UNIQUE ACTIVITIES")
print(df["activity"].nunique())

# 5. Candidate numerical feature correlations
print("\n5. NUMERICAL FEATURE CORRELATIONS")
print(
    df[[
        "loan_amount",
        "funded_amount",
        "term_in_months",
        "lender_count"
    ]].corr().round(3)
)

print("\nFEATURE REVIEW COMPLETED")
