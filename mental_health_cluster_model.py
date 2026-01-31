# Mental Health Cluster Model by Garv Gursahaney
# Use of a dataset to cluster survey respondents based on mental health-related features

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load the dataset and test if it exists
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_name = "mental-heath-in-tech-2016_20161114.csv"
data_path = os.path.join(script_dir, csv_name)
if not os.path.isfile(data_path):
    data_path = os.path.join(script_dir, "data", csv_name)

if os.path.isfile(data_path):
    data = pd.read_csv(data_path)
    print("Loaded survey data from the file.")
else:

    print("Survey file can unfortunately not be found. Using sample data so you can run the full pipeline.")
    print("To use your own data, place the CSV in the script folder or in a 'data' subfolder next to the script.\n")
    np.random.seed(42)
    n = 200
    data = pd.DataFrame({
        "Age": np.random.randint(22, 60, n),
        "Gender": np.random.choice(["Male", "Female", "Other"], n),
        "work_interfere": np.random.choice(["Often", "Sometimes", "Rarely", "Never", np.nan], n),
        "benefits": np.random.choice(["Yes", "No", "Don't know"], n),
        "care_options": np.random.choice(["Yes", "No", "Not sure"], n),
        "wellness_program": np.random.choice(["Yes", "No", "Don't know"], n),
        "seek_help": np.random.choice(["Yes", "No", "Don't know"], n),
        "anonymity": np.random.choice(["Yes", "No", "Don't know"], n),
        "leave": np.random.choice(["Very easy", "Somewhat easy", "Somewhat difficult", "Very difficult", "Don't know"], n),
        "mental_health_consequence": np.random.choice(["Yes", "No", "Maybe"], n),
        "phys_health_consequence": np.random.choice(["Yes", "No", "Maybe"], n),
        "coworkers": np.random.choice(["Yes", "No", "Some of them"], n),
        "supervisor": np.random.choice(["Yes", "No", "Some of them"], n),
        "mental_health_interview": np.random.choice(["Yes", "No", "Maybe"], n),
        "phys_health_interview": np.random.choice(["Yes", "No", "Maybe"], n),
        "mental_vs_physical": np.random.choice(["Yes", "No", "Don't know"], n),
        "obs_consequence": np.random.choice(["Yes", "No"], n),
    })

print("Dataset shape:", data.shape)
print("\nFirst few rows:")
print(data.head())


# Basic cleaning Task and preprocessing steps
missing_threshold = 0.5
data = data.loc[:, data.isnull().mean() < missing_threshold]

num_cols = data.select_dtypes(include=["int64", "float64"]).columns
cat_cols = data.select_dtypes(include=["object"]).columns

# Missing values are input and handled
for col in num_cols:
    data[col] = data[col].fillna(data[col].median())

for col in cat_cols:
    mode_val = data[col].mode()
    if len(mode_val) > 0:
        data[col] = data[col].fillna(mode_val[0])


# Encode the categorical variables
encoded_data = data.copy()

for col in cat_cols:
    le = LabelEncoder()
    encoded_data[col] = le.fit_transform(encoded_data[col].astype(str))


scaler = StandardScaler()
scaled_data = scaler.fit_transform(encoded_data)

# Dimensionality reduction using PCA
pca = PCA(n_components=0.75, random_state=42)
pca_data = pca.fit_transform(scaled_data)

print("\n--- Dimensionality ---")
print("Original number of features:", scaled_data.shape[1])
print("After PCA (keeping 75% of variance):", pca_data.shape[1])


# Here we determine the most optimal numbers of clusters
silhouette_scores = []
k_range = range(2, 8)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(pca_data)
    score = silhouette_score(pca_data, labels)
    silhouette_scores.append(score)

plt.figure(figsize=(8, 5))
plt.plot(k_range, silhouette_scores, marker="o", linewidth=2, markersize=8)
plt.xlabel("Number of clusters (k)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette analysis — higher is better")
plt.grid(True, alpha=0.3)
plt.tight_layout()
silhouette_path = os.path.join(script_dir, "Silhouette_Analysis.png")
plt.savefig(silhouette_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved silhouette plot to {silhouette_path}")


# Adjust k based on silhouette analysis
optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
clusters = kmeans.fit_predict(pca_data)

data["Cluster"] = clusters

print(f"\nAssigned each respondent to one of {optimal_k} clusters.")


# Visualise clusters in a 2 Dimensional PCA space
pca_2d = PCA(n_components=2)
pca_vis = pca_2d.fit_transform(scaled_data)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(
    pca_vis[:, 0],
    pca_vis[:, 1],
    c=clusters,
    cmap="Set2",
    alpha=0.7,
    edgecolors="w",
    linewidths=0.5,
)
plt.colorbar(scatter, label="Cluster")
plt.title("PCA view of survey respondents by cluster")
plt.xlabel("First principal component")
plt.ylabel("Second principal component")
plt.tight_layout()
pca_path = os.path.join(script_dir, "PCA_Clusters.png")
plt.savefig(pca_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved PCA cluster plot to {pca_path}")


# Summarise what each cluster will finally look like
encoded_data_with_cluster = encoded_data.copy()
encoded_data_with_cluster["Cluster"] = clusters
cluster_summary = encoded_data_with_cluster.groupby("Cluster").mean()
print("\n--- Cluster summary (average encoded values per cluster) ---")
print(cluster_summary)
print("\nDone.")
