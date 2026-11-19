
import pandas as pd
import os

input_file = "dataset/kiva_loans.csv"
output_file = "dataset/india_kiva_loans.csv"

print("=" * 55)
print("MICROSEGMENT - INDIA DATASET CREATION")
print("=" * 55)

total_india_rows = 0
currency_counts = {}
missing_values = None
first_chunk = True

# Process the large dataset in smaller parts
for chunk in pd.read_csv(input_file, chunksize=100000):

    # Select only Indian records
    india_chunk = chunk[chunk["country"] == "India"].copy()

    if india_chunk.empty:
        continue

    total_india_rows += len(india_chunk)

    # Count currencies
    counts = india_chunk["currency"].value_counts()
    for currency, count in counts.items():
        currency_counts[currency] = (
            currency_counts.get(currency, 0) + int(count)
        )

    # Calculate missing values
    current_missing = india_chunk.isnull().sum()

    if missing_values is None:
        missing_values = current_missing
    else:
        missing_values += current_missing

    # Save the Indian records
    india_chunk.to_csv(
        output_file,
        mode="w" if first_chunk else "a",
        header=first_chunk,
        index=False
    )

    first_chunk = False

print("\n1. TOTAL INDIAN RECORDS")
print(total_india_rows)

print("\n2. CURRENCY DISTRIBUTION")
print(pd.Series(currency_counts).sort_values(ascending=False))

print("\n3. MISSING VALUES IN INDIAN DATA")
if missing_values is not None:
    print(missing_values.sort_values(ascending=False))

print("\n4. OUTPUT FILE")
if os.path.exists(output_file):
    size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"Created: {output_file}")
    print(f"File size: {size_mb:.2f} MB")
else:
    print("India dataset was not created.")

print("\nINDIA DATASET CREATION COMPLETED")
