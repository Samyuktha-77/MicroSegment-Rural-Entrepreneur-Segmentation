
import pandas as pd

file_path = "dataset/india_kiva_loans.csv"
output_file = "dataset/india_inr_loans.csv"

print("=" * 55)
print("MICROSEGMENT - DATA QUALITY CHECK")
print("=" * 55)

df = pd.read_csv(file_path)

# Keep only Indian records denominated in INR
df = df[df["currency"] == "INR"].copy()

print("\n1. DATASET SIZE")
print("Rows:", len(df))
print("Columns:", len(df.columns))

# Check duplicate records
print("\n2. DUPLICATE CHECK")
print("Duplicate IDs:", df["id"].duplicated().sum())
print("Duplicate complete rows:", df.duplicated().sum())

# Numeric feature statistics
print("\n3. NUMERIC FEATURE STATISTICS")
print(
    df[["loan_amount", "funded_amount",
        "term_in_months", "lender_count"]].describe()
)

# Loan amount percentiles
print("\n4. LOAN AMOUNT PERCENTILES")
print(df["loan_amount"].quantile([0, 0.25, 0.50, 0.75, 0.90, 0.95, 1]))

# Business sector distribution
print("\n5. BUSINESS SECTOR DISTRIBUTION")
print(df["sector"].value_counts().head(15))

# Business activity distribution
print("\n6. TOP 15 BUSINESS ACTIVITIES")
print(df["activity"].value_counts().head(15))

# Repayment interval distribution
print("\n7. REPAYMENT INTERVAL DISTRIBUTION")
print(df["repayment_interval"].value_counts())

# Date coverage
print("\n8. DATE COVERAGE")
dates = pd.to_datetime(df["date"], errors="coerce")
print("Earliest date:", dates.min())
print("Latest date:", dates.max())
print("Invalid dates:", dates.isna().sum())

# Funding amount check
print("\n9. FUNDING CHECK")
print("Records where funded amount exceeds loan amount:",
      (df["funded_amount"] > df["loan_amount"]).sum())

print("Records with funding gap:",
      (df["funded_amount"] < df["loan_amount"]).sum())

# Save clean INR dataset
df.to_csv(output_file, index=False)

print("\n10. CLEAN DATASET SAVED")
print(output_file)

print("\nQUALITY CHECK COMPLETED")
