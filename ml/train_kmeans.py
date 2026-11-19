
import os
import joblib
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)

print("=" * 60)
print("MICROSEGMENT - FINAL K-MEANS TRAINING (INITIAL MODEL)")
print("=" * 60)

# Load original business dataset and processed features
df = pd.read_csv("dataset/india_business_loans.csv")
X = pd.read_csv("dataset/processed_business_features.csv")

# Train K-Means with 3 clusters
kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

df["cluster"] = kmeans.fit_predict(X)

# Evaluate model
print("\n1. MODEL EVALUATION")

silhouette = silhouette_score(
    X,
    df["cluster"],
    sample_size=min(3000, len(X)),
    random_state=42
)

print("Silhouette Score:", round(silhouette, 4))
print(
    "Davies-Bouldin Score:",
    round(davies_bouldin_score(X, df["cluster"]), 4)
)
print(
    "Calinski-Harabasz Score:",
    round(calinski_harabasz_score(X, df["cluster"]), 2)
)

# Cluster sizes
print("\n2. NUMBER OF RECORDS IN EACH CLUSTER")
print(df["cluster"].value_counts().sort_index())

print("\n3. CLUSTER PERCENTAGES")
print(
    (df["cluster"].value_counts(normalize=True)
     .sort_index() * 100).round(2)
)

# Numerical profiles
print("\n4. NUMERICAL PROFILE OF EACH CLUSTER")

numeric_profile = df.groupby("cluster").agg(
    record_count=("id", "count"),
    average_loan=("loan_amount", "mean"),
    median_loan=("loan_amount", "median"),
    average_term=("term_in_months", "mean"),
    median_term=("term_in_months", "median")
).round(2)

print(numeric_profile)

# Most common sector in each cluster
print("\n5. MOST COMMON SECTOR IN EACH CLUSTER")

for cluster in sorted(df["cluster"].unique()):
    print(f"\nCluster {cluster}")
    print(
        df[df["cluster"] == cluster]["sector"]
        .value_counts(normalize=True)
        .head(5)
        .mul(100)
        .round(2)
    )

# Repayment interval distribution
print("\n6. REPAYMENT PATTERN IN EACH CLUSTER")

for cluster in sorted(df["cluster"].unique()):
    print(f"\nCluster {cluster}")
    print(
        df[df["cluster"] == cluster]["repayment_interval"]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
    )

# Most common activities
print("\n7. TOP BUSINESS ACTIVITIES IN EACH CLUSTER")

for cluster in sorted(df["cluster"].unique()):
    print(f"\nCluster {cluster}")
    print(
        df[df["cluster"] == cluster]["activity"]
        .value_counts()
        .head(5)
    )

# Save model and labelled data
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

joblib.dump(kmeans, "models/kmeans_model.joblib")

df.to_csv(
    "results/clustered_business_loans.csv",
    index=False
)

numeric_profile.to_csv(
    "results/cluster_profiles.csv"
)

print("\n8. FILES SAVED")
print("models/kmeans_model.joblib")
print("results/clustered_business_loans.csv")
print("results/cluster_profiles.csv")

print("\nK-MEANS TRAINING COMPLETED")
