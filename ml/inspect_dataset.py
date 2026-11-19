
import pandas as pd
import os

file_path = "dataset/kiva_loans.csv"

print("MICROSEGMENT - DATASET INSPECTION")
print("-" * 45)

# Check file size
file_size = os.path.getsize(file_path) / (1024 * 1024)
print(f"Dataset size: {file_size:.2f} MB")

# Read only the first 5 rows initially
df = pd.read_csv(file_path, nrows=5)

print("\nCOLUMN NAMES:")
print(df.columns.tolist())

print("\nFIRST 5 RECORDS:")
print(df.to_string())

print("\nDATA TYPES:")
print(df.dtypes)

print("\nINSPECTION COMPLETED")
