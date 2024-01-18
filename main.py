import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

def initialize_centroids(X, k):
    indices = np.random.choice(len(X), k, replace=False)
    return X[indices]

def assign_to_clusters(X, centroids):
    distances = np.linalg.norm(X[:, np.newaxis, :] - centroids, axis=2)
    return np.argmin(distances, axis=1)

def update_centroids(X, clusters, k):
    new_centroids = np.array([X[clusters == i].mean(axis=0) for i in range(k)])
    return new_centroids

def kmeans(X, k, max_iterations=100):
    centroids = initialize_centroids(X, k)

    print(f"Centroid Awal:\n{centroids}")

    new_clusters = None  

    for i in range(max_iterations):
        clusters = assign_to_clusters(X, centroids)

        new_centroids = update_centroids(X, clusters, k)

        print(f"Iterasi {i+1} - Centroids Baru:\n{new_centroids}")

        if new_clusters is not None and np.all(clusters == new_clusters):
            print(f"Cluster Stabil pada Iterasi ke-{i+1}")
            break

        centroids = new_centroids
        new_clusters = clusters

    return centroids, clusters


file_path = './dataset/Diabetes.xlsx'
df = pd.read_excel(file_path)


df_subset = df.head(10).copy() 


selected_features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
X_subset = df_subset[selected_features].values


k = 2


final_centroids, final_clusters = kmeans(X_subset, k)

df_subset['Cluster'] = final_clusters

# Menghitung akurasi
accuracy = accuracy_score(df_subset['ClassOutcome'], df_subset['Cluster'])
print(f"\nAkurasi: {accuracy}")
