import numpy as np
import AutomatingAnalysisModelsAndMisprediction.data.process_data as process
import sklearn.cluster as cluster
from sklearn.preprocessing import StandardScaler


churn_df, X_train, X_test, Y_train, Y_test = process.process_data_churn('Customer_Churn.csv')
# Use k-means clustering to identify groups of similar customers
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)
kmeans = cluster.KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)

# Get the cluster assignments for each customer
cluster_labels = kmeans.predict(X_scaled)

# Identify the examples with the highest misprediction rates
"""mispredictions = (Y_train != kmeans.predict(X_scaled)).astype(int)
misprediction_rates = np.mean(mispredictions, axis=1)
high_misprediction_indices = np.argsort(misprediction_rates)[-10:]

# Print the inputs and outputs of the examples with the highest misprediction rates.
for i in high_misprediction_indices:
    print('Example', i)
    print('Input:', X_train[i])
    print('Output:', Y_train[i])
    print()"""

# Identify the common features contributing to the mispredictions
cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
feature_names = churn_df.drop(['Churn'], axis=1).columns

for i, cluster in enumerate(cluster_centers):
    print('Cluster', i)
    for feature, value in zip(feature_names, cluster):
        print(feature, ':', value)
    print()

# Use feature engineering to create new features that better capture the underlying factors driving mispredictions
"""X_new = np.zeros((X_train.shape[0], X_train.shape[1]+1))
X_new[:,:-1] = X_train
X_new[:,-1] = X_train[:,0] + X_train[:,1]

# Fit the k-means clustering model with the new features
X_new_scaled = scaler.fit_transform(X_new)
kmeans = cluster.KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_new_scaled)

# Get the cluster assignments for each example using the new features
cluster_labels = kmeans.predict(X_new_scaled)

# Identify the examples with the highest misprediction rates using the new features
mispredictions = (Y_train != kmeans.predict(X_new_scaled)).astype(int)
misprediction_rates = np.mean(mispredictions, axis=1)
high_misprediction_indices = np.argsort(misprediction_rates)[-10:]

# Print the inputs and outputs of the examples with the highest misprediction rates using the new features
for i in high_misprediction_indices:
    print('Example', i)
    print('Input:', X_new[i])
    print('Output:', Y_train[i])
    print()

# Identify the common features contributing to the mispredictions using the new features
cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
feature_names = churn_df.drop(['Churn'], axis=1).columns
feature_names_new = list(feature_names) + ['New Feature']

for i, cluster in enumerate(cluster_centers):
    print('Cluster', i)
    for feature, value in zip(feature_names_new, cluster):
        print(feature, ':', value)
    print()"""

