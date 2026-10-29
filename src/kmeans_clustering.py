
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    confusion_matrix,
)

from load_data import DATA_DIR


# 1. Load the cleaned news dataset.
df = pd.read_csv(DATA_DIR / "clean.csv").dropna(
    subset=["content", "label"]
)

X = df["content"]
y = df["label"]

# 2. Use the same held-out test split.
# Labels are NOT given to K-means during clustering.
_, X_test, _, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

print("Test articles:", len(X_test))

# 3. Convert articles into numerical TF-IDF features.
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=5,
    max_df=0.9,
    max_features=50000,
)

X_test_tfidf = vectorizer.fit_transform(X_test)

# 4. Group the articles into two clusters.
kmeans = KMeans(
    n_clusters=2,
    random_state=42,
    n_init=10,
)

clusters = kmeans.fit_predict(X_test_tfidf)

# 5. Compare clusters with the known labels AFTER clustering.
# Cluster IDs (0 and 1) do not inherently mean fake or real.
ari = adjusted_rand_score(y_test, clusters)
nmi = normalized_mutual_info_score(y_test, clusters)

print("\n===== K-MEANS CLUSTERING RESULTS =====")
print(f"Adjusted Rand Index (ARI): {ari:.4f}")
print(f"Normalized Mutual Information (NMI): {nmi:.4f}")

print("\nCluster sizes:")
print(pd.Series(clusters).value_counts().sort_index())

print("\nConfusion matrix (rows = true labels, columns = clusters):")
print(confusion_matrix(y_test, clusters, labels=[0, 1]))

print(
    "\nNote: Cluster 0 and Cluster 1 have no predefined fake/real "
    "meaning. ARI and NMI compare the grouping with known labels "
    "without requiring cluster IDs to match label IDs."
)
