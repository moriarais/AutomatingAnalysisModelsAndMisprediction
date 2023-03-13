# Automating Analysis Models And Misprediction

The project includes an automated process to streamline the model building, evaluation, and training phases of K-means clustering and the SVM algorithm, aimed at achieving the most accurate analysis of model faults and misprediction results. The process involves data cleaning, conversion, and feature selection, followed by the use of K-means clustering and PyTorch to identify the mispredicted data, group it into similar clusters, and select the most important features. In Addition to K-Means, we used the SVM algorithm and automated it using the H2O AutoML framework. The aim is to prevent overfitting or underfitting of the model and to remove irrelevant or redundant features to achieve a more robust and accurate model. The results for both are visualized using a bar plot, histogram plot, and kernel density estimate plot. The process is applied to several datasets, including Telco Customer Churn, Company Bankruptcy Prediction, Credit card fraud, and Pima Indians Diabetes.

# Data bases links:
1. Telco Customer Churn: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
2. Credit card fraud: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
3. Pima Indians Diabetes: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
4. Company Bankruptcy Prediction: https://www.kaggle.com/datasets/fedesoriano/company-bankruptcy-prediction

# Notes!
For data csv files we use Git’s Large File System (Git LFS)
