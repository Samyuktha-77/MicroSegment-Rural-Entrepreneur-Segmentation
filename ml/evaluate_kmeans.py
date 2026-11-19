
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)

print("=" * 65)
print("MICROSEGMENT - K-MEANS CLUSTER EVALUATION")
print("=" * 65)

# Load preprocessed numerical data
X = pd.read_csv("dataset/processed_business_features.csv")

print("\n1. DATA LOADED")
print("Shape:", X.shape)

# Test different numbers of clusters
k_values = range(2, 11)

results = []

for k in k_values:

    print(f"\nTesting K = {k} ...")

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(X)

    # Silhouette uses a representative sample to reduce computation
    silhouette = silhouette_score(
        X,
        labels,
        sample_size=min(3000, len(X)),
        random_state=42
    )

    db_score = davies_bouldin_score(X, labels)

    ch_score = calinski_harabasz_score(X, labels)

    results.append({
        "K": k,
        "Inertia": model.inertia_,
        "Silhouette": silhouette,
        "Davies_Bouldin": db_score,
        "Calinski_Harabasz": ch_score
    })

    print("Inertia:", round(model.inertia_, 2))
    print("Silhouette:", round(silhouette, 4))
    print("Davies-Bouldin:", round(db_score, 4))
    print("Calinski-Harabasz:", round(ch_score, 2))

# Convert results to DataFrame
results_df = pd.DataFrame(results)

os.makedirs("results", exist_ok=True)

results_df.to_csv(
    "results/kmeans_evaluation.csv",
    index=False
)

print("\n2. COMPLETE EVALUATION TABLE")
print(results_df.round(4).to_string(index=False))

# Find the best K according to silhouette score
best_row = results_df.loc[
    results_df["Silhouette"].idxmax()
]

print("\n3. HIGHEST SILHOUETTE SCORE")
print("K:", int(best_row["K"]))
print("Score:", round(best_row["Silhouette"], 4))

# Plot evaluation charts
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Elbow curve
axes[0].plot(
    results_df["K"],
    results_df["Inertia"],
    marker="o"
)
axes[0].set_title("Elbow Method")
axes[0].set_xlabel("Number of Clusters (K)")
axes[0].set_ylabel("Inertia")
axes[0].set_xticks(list(k_values))
axes[0].grid(True)

# Silhouette curve
axes[1].plot(
    results_df["K"],
    results_df["Silhouette"],
    marker="o"
)
axes[1].set_title("Silhouette Score")
axes[1].set_xlabel("Number of Clusters (K)")
axes[1].set_ylabel("Silhouette Score")
axes[1].set_xticks(list(k_values))
axes[1].grid(True)

plt.tight_layout()
plt.savefig(
    "results/kmeans_evaluation.png",
    dpi=300
)
plt.show()

print("\n4. FILES SAVED")
print("results/kmeans_evaluation.csv")
print("results/kmeans_evaluation.png")

print("\nK-MEANS EVALUATION COMPLETED")
