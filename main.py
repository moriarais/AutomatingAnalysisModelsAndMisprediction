import pandas as pd
import numpy as np
import torch
import sklearn.cluster as cluster

# Load the customer churn dataset into dataframe
churn_df = pd.read_csv('Customer_Churn.csv')
print(churn_df.head(5))
