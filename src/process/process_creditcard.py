"""Python script to process the data"""

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle

import config.config_creditcard as config_creditcard

def get_raw_data(data_location: str):
    """Read raw data
    """
    return pd.read_csv(data_location)

def get_X_y(data: pd.DataFrame, label: str):
    """Get features and label
    """
    X = data.drop(columns=label)
    y = data[label]
    return X, y

def split_train_test(X: pd.DataFrame, y: pd.DataFrame, test_size: int):
    """_summary_

    Parameters
    ----------
    X : pd.DataFrame
        Features
    y : pd.DataFrame
        Target
    test_size : int
        Size of the test set
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=0
    )
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }


def save_processed_data(data: dict, save_location: str):
    """Save processed data

    Parameters
    ----------
    data : dict
        Data to process
    save_location : str
        Where to save the data
    """
    joblib.dump(data, save_location)

def print_pkl_file(file_path: str):
    # read python dict back from the file
    pkl_file = open(file_path, 'rb')
    mydict2 = pickle.load(pkl_file)
    pkl_file.close()
    print(mydict2)


def process(
    location: config_creditcard.Location = config_creditcard.Location
):
    """Flow to process the Data
    """
    data = get_raw_data(location.data_raw)
    
    # Cleaning the data process:
    
    # 1. Null values handler. There are no null values in DB.
    # print(data.isnull().sum())
    # data.dropna(inplace=True)
    # print(data.isnull().sum())
    
    # 2. All the features are required for the ml process
    
    # 3. Convert to numeric Class feature
    # 0 - non-fraudulent. 1 - fraudulent.
    data["Class"] = pd.to_numeric(data["Class"], errors='coerce')

    processed = data
    # After cleaning and processing the database, display general statistics of dataset
    # print(processed.describe())

    X, y = get_X_y(processed, config_creditcard.ProcessConfig.label)
    split_data = split_train_test(X, y, config_creditcard.ProcessConfig.test_size)
    save_processed_data(split_data, config_creditcard.Location.data_process)
    return split_data
    # print_pkl_file(config_creditcard.Location.data_process)

if __name__ == "__main__":
    process()
