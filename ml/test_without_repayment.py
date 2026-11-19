
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)

print("=" * 60)
print("MICROSEGMENT - CLUSTERING WITHOUT REPAYMENT INTERVAL")
print("=" * 60)

X = pd.read_csv("dataset/processed_business_features.csv")

# Remove repayment interval one-hot encoded columns
repayment_columns = [
    col for col in X.columns
    if col.startswith("categorical__repayment_interval_")
]

X_reduced = X.drop(columns=repayment_columns)

print("\n1. FEATURE COMPARISON")
print("Original feature count:", X.shape[1])
print("New feature count:", X_reduced.shape[1])
print("Removed features:", repayment_columns)

results = []

# Test K values from 2 to 10
for k in range(2, 11):

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(X_reduced)

    silhouette = silhouette_score(
        X_reduced,
        labels,
        sample_size=min(3000, len(X_reduced)),
        random_state=42
    )

    db = davies_bouldin_score(X_reduced, labels)
    ch = calinski_harabasz_score(X_reduced, labels)

    results.append({
        "K": k,
        "Inertia": round(model.inertia_, 2),
        "Silhouette": round(silhouette, 4),
        "Davies_Bouldin": round(db, 4),
        "Calinski_Harabasz": round(ch, 2)
    })

    print(
        f"\nK={k} | Silhouette={silhouette:.4f}"
        f" | Davies-Bouldin={db:.4f}"
        f" | Calinski-Harabasz={ch:.2f}"
    )

results_df = pd.DataFrame(results)

print("\n2. COMPLETE RESULTS")
print(results_df.to_string(index=False))

best = results_df.loc[results_df["Silhouette"].idxmax()]

print("\n3. BEST SILHOUETTE RESULT")
print("K:", int(best["K"]))
print("Score:", best["Silhouette"])

print("\n4. K=3 COMPARISON")
print("Original Silhouette: 0.4924")
print(
    "Without repayment interval:",
    results_df.loc[results_df["K"] == 3, "Silhouette"].iloc[0]
)

print("\nEXPERIMENT COMPLETED")
