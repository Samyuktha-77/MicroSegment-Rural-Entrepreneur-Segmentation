
import os
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

print("=" * 60)
print("MICROSEGMENT - DATA PREPROCESSING")
print("=" * 60)

# Load business-focused dataset
input_file = "dataset/india_business_loans.csv"
df = pd.read_csv(input_file)

print("\n1. DATASET LOADED")
print("Records:", len(df))

# Prepare numerical features
df["loan_amount_log"] = np.log1p(df["loan_amount"])

numeric_features = [
    "loan_amount_log",
    "term_in_months"
]

# Prepare categorical features
# Group very rare sectors to avoid unstable tiny categories
sector_counts = df["sector"].value_counts()

rare_sectors = sector_counts[sector_counts < 20].index

df["sector_clean"] = df["sector"].replace(
    rare_sectors,
    "Other"
)

categorical_features = [
    "sector_clean",
    "repayment_interval"
]

print("\n2. RARE SECTORS GROUPED")
print(list(rare_sectors))

print("\n3. LOAN AMOUNT SKEWNESS")
print(round(df["loan_amount"].skew(), 3))

# Build preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            StandardScaler(),
            numeric_features
        ),
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),
            categorical_features
        )
    ]
)

# Select input columns
X = df[numeric_features + categorical_features]

# Fit and transform
X_processed = preprocessor.fit_transform(X)

# Get meaningful feature names
feature_names = preprocessor.get_feature_names_out()

processed_df = pd.DataFrame(
    X_processed,
    columns=feature_names
)

# Save processed dataset
output_file = "dataset/processed_business_features.csv"
processed_df.to_csv(output_file, index=False)

# Save preprocessing object for future predictions
joblib.dump(
    preprocessor,
    "ml/preprocessor.joblib"
)

print("\n4. PROCESSED FEATURE MATRIX")
print("Rows:", processed_df.shape[0])
print("Columns:", processed_df.shape[1])

print("\n5. FEATURE NAMES")
print(processed_df.columns.tolist())

print("\n6. FIRST FIVE PROCESSED RECORDS")
print(processed_df.head())

print("\n7. CHECK FOR MISSING VALUES")
print(processed_df.isnull().sum().sum())

print("\n8. CHECK FEATURE SCALING")
print(processed_df.describe().round(2))

print("\n9. FILES SAVED")
print(output_file)
print("ml/preprocessor.joblib")

print("\nPREPROCESSING COMPLETED")
