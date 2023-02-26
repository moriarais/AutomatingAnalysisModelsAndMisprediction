import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split


def get_process_data_churn(file):
    # Load the customer churn dataset into dataframe.
    churn_df = pd.read_csv(file)
    # print(churn_df.head(5))
    # Total Charges column was categorical but must be a numerical value.
    # hence we convert it to categorical and fill with NAN values if there are errors.
    churn_df["TotalCharges"] = pd.to_numeric(churn_df["TotalCharges"], errors='coerce')

    # Find out how many Nan values are in the dataset now
    # print(churn_df.isnull().sum())
    # drop NaN values since they are small
    churn_df.dropna(subset=["TotalCharges"], inplace=True)
    # print(churn_df.isnull().sum())
    # also we don't need the customer ID columm
    churn_df.drop("customerID", axis=1, inplace=True)
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
    # print(churn_df.head(5))

    # get dummy for categorical columns now
    churn_df = pd.get_dummies(churn_df)
    # churn_df = churn_df.astype(float)
    # print(churn_df.head(5))

    # Extract the features and labels
    X = churn_df.drop(['Churn'], axis=1)
    Y = churn_df['Churn']

    over_sample = SMOTE()
    X_ros, Y_ros = over_sample.fit_resample(X, Y)

    # we split the data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X_ros, Y_ros, test_size=0.2, random_state=42)
    return X_train, X_test, Y_train, Y_test
