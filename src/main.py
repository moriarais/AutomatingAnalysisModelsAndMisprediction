import pandas as pd
import numpy as np
import torch
import sklearn.cluster as cluster
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load the customer churn dataset into dataframe.
churn_df = pd.read_csv('Customer_Churn.csv')
print(churn_df.head(5))

# Total Charges column was categorical but must be a numerical value.
# hence we convert it to categorical and fill with NAN values if there are errors.
churn_df["TotalCharges"] = pd.to_numeric(churn_df["TotalCharges"], errors='coerce')

# Find out how many Nan values are in the dataset now
# print(churn_df.isnull().sum())
# drop NaN values since they are small
churn_df.dropna(subset=["TotalCharges"], inplace=True)
# print(churn_df.isnull().sum())
# As expected there are no missing values in any of the columns. Dataset seems to be clean

# display general statistics of dataset
# print(churn_df.describe())

# maping of male to 1 and female to zero.
churn_df['gender'] = churn_df['gender'].map({'Male': 1, 'Female': 0})
# maping of yes to 1 and No to 0 for the following column
churn_df['Partner'] = churn_df['Partner'].map({'Yes': 1, 'No': 0})
churn_df['Dependents'] = churn_df['Dependents'].map({'Yes': 1, 'No': 0})
churn_df['PhoneService'] = churn_df['PhoneService'].map({'Yes': 1, 'No': 0})
churn_df['PaperlessBilling'] = churn_df['PaperlessBilling'].map({'Yes': 1, 'No': 0})
churn_df['Churn'] = churn_df['Churn'].map({'Yes': 1, 'No': 0})
print(churn_df.head(5))

# we separate the features and target variable
X = churn_df.drop('Churn', axis=1)
y = churn_df['Churn']

# we standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#  we split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

"""# Split the dataset into training and test sets
train_df = churn_df.sample(frac=0.8, random_state=123)
test_df = churn_df.drop(train_df.index)

# Extract the features and labels
train_features = train_df.drop(['Churn'], axis=1).values
train_labels = train_df['Churn'].values"""