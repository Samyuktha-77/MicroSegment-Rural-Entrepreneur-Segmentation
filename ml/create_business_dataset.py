
import pandas as pd

input_file = "dataset/india_inr_loans.csv"
output_file = "dataset/india_business_loans.csv"

print("=" * 55)
print("MICROSEGMENT - BUSINESS DATASET PREPARATION")
print("=" * 55)

df = pd.read_csv(input_file)

# Exclude clearly personal-use categories
excluded_sectors = [
    "Education",
    "Housing",
    "Personal Use"
]

business_df = df[
    ~df["sector"].isin(excluded_sectors)
].copy()

print("\n1. ORIGINAL INDIA INR RECORDS")
print(len(df))

print("\n2. EXCLUDED RECORDS")
print(
    df[df["sector"].isin(excluded_sectors)]
    ["sector"].value_counts()
)

print("\n3. BUSINESS-FOCUSED RECORD COUNT")
print(len(business_df))

print("\n4. REMAINING SECTOR DISTRIBUTION")
print(business_df["sector"].value_counts())

print("\n5. UNIQUE BUSINESS ACTIVITIES")
print(business_df["activity"].nunique())

print("\n6. MISSING VALUES IN CANDIDATE FEATURES")
candidate_features = [
    "loan_amount",
    "term_in_months",
    "sector",
    "activity",
    "repayment_interval"
]

print(business_df[candidate_features].isnull().sum())

# Save the new dataset
business_df.to_csv(output_file, index=False)

print("\n7. DATASET SAVED")
print(output_file)

print("\nBUSINESS DATASET PREPARATION COMPLETED")
