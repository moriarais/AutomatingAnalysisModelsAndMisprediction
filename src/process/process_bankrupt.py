import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer, StandardScaler
import numpy as np
from imblearn.over_sampling import RandomOverSampler, SMOTE


def get_process_data_bankrupt(file):
    # Load the customer churn dataset into dataframe.
    bankrupt_df = pd.read_csv(file)
    # print(bankrupt_df.head(5))
    # print(bankrupt_df.info())
    # As we can see no value is categorical, That's good news for us.
    # Let's check if any values are null or not

    """for i in bankrupt_df.columns:
        if bankrupt_df[i].isnull().values.any():
            print(i)
    print("Done")"""
    # As we can see there aren't any null values.

    # Extract the features and labels
    X = bankrupt_df.drop(["Bankrupt?"], axis=1)
    Y = bankrupt_df["Bankrupt?"]

    # print(bankrupt_df["Bankrupt?"].value_counts())
    # Our labels are strongly unbalanced, so we do the oversampling method

    over_sample = SMOTE()
    X_ros, Y_ros = over_sample.fit_resample(X, Y)

    # we split the data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X_ros, Y_ros, test_size=0.2, random_state=42)
    return X_train, X_test, Y_train, Y_test

