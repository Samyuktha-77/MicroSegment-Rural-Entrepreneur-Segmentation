
import pandas as pd

file_path = "dataset/kiva_loans.csv"

print("=" * 55)
print("MICROSEGMENT - FULL DATASET ANALYSIS")
print("=" * 55)

total_rows = 0
missing_values = None
country_counts = {}
sector_counts = {}
currency_counts = {}
repayment_counts = {}

# Process the large CSV in smaller chunks
for chunk in pd.read_csv(file_path, chunksize=100000):

    total_rows += len(chunk)

    # Count missing values
    current_missing = chunk.isnull().sum()

    if missing_values is None:
        missing_values = current_missing
    else:
        missing_values += current_missing

    # Country distribution
    country_counts.update(
        chunk["country"].value_counts().to_dict()
    ) if False else None

    for column, storage in [
        ("country", country_counts),
        ("sector", sector_counts),
        ("currency", currency_counts),
        ("repayment_interval", repayment_counts)
    ]:
        counts = chunk[column].value_counts()

        for key, value in counts.items():
            storage[key] = storage.get(key, 0) + value

print("\n1. TOTAL RECORDS")
print(total_rows)

print("\n2. MISSING VALUES")
print(missing_values.sort_values(ascending=False))

print("\n3. TOP 15 COUNTRIES")
print(
    pd.Series(country_counts)
    .sort_values(ascending=False)
    .head(15)
)

print("\n4. TOP 15 BUSINESS SECTORS")
print(
    pd.Series(sector_counts)
    .sort_values(ascending=False)
    .head(15)
)

print("\n5. CURRENCY DISTRIBUTION")
print(
    pd.Series(currency_counts)
    .sort_values(ascending=False)
    .head(15)
)

print("\n6. REPAYMENT INTERVALS")
print(pd.Series(repayment_counts))

print("\n7. INDIA RECORD COUNT")
print(country_counts.get("India", 0))

print("\nANALYSIS COMPLETED")
