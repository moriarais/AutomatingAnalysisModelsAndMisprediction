from sklearn.metrics import accuracy_score
import AutomatingAnalysisModelsAndMisprediction.src.process_data_churn as process
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression

churn_df, X_train, X_test, Y_train, Y_test = process.process_data_churn('../data/Customer_Churn.csv')

# Collect data on model faults and mispredictions : Logistic Regression
reg_model = LogisticRegression(max_iter=1000)
reg_model.fit(X_train, Y_train)
log_reg_score = reg_model.score(X_test, Y_test)
accuracy_dict = {}
print('Logistic Regression accuracy before changes: ', log_reg_score * 100, '%')

# Predict the target variable for the testing set
Y_pred = reg_model.predict(X_test)

# Identify the mispredictions and their associated feature values
data_test = X_test.copy()
data_test['Churn'] = Y_test
data_test['Prediction'] = Y_pred
mispredictions = data_test[data_test['Churn'] != data_test['Prediction']]

# Use k-means clustering to identify groups of similar customers
n_clusters = 5
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(mispredictions.drop(['Churn', 'Prediction'], axis=1))

# Use feature engineering to identify the common features that are contributing to the mispredictions in each cluster
for i in range(n_clusters):
    print('Cluster ', i, 'common features: ')
    cluster_data = mispredictions[clusters == i].drop(['Churn', 'Prediction'], axis=1)
    common_features = cluster_data.mean().sort_values(ascending=False)
    print(common_features)

    # Make changes to the predictive model to reduce the misprediction rate
    # For example, we might remove the least informative features.
    # So, we will drop the least informative features and retrain the model on the original dataset with these changes
    least_informative_features = common_features[common_features < 0.56].index
    X_train_new = X_train.drop(least_informative_features, axis=1)
    X_test_new = X_test.drop(least_informative_features, axis=1)

    model_new = LogisticRegression(max_iter=1000)
    model_new.fit(X_train_new, Y_train)

    Y_pred_new = model_new.predict(X_test_new)
    accuracy_new = accuracy_score(Y_test, Y_pred_new)
    accuracy_dict.update({i: accuracy_new})
    print('Accuracy of the model with the drop of the least informative features in the cluster', i, 'is',
          accuracy_new * 100, '%')

max_accuracy = max(accuracy_dict.values())
print('The max accuracy we could get is', max_accuracy * 100, '%')
print('We gain a better accuracy of the model of', (max_accuracy - log_reg_score) * 100, '%')
