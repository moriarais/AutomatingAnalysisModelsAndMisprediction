"""Python script to process the data"""

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import pickle

import config.config_health_diabetes as config_diabetes


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
    with open(save_location, "wb") as f:
        pickle.dump(data, f)


def print_pkl_file(file_path: str):
    # read python dict back from the file
    with open(file_path, 'rb') as f:
        svm_model, (X_train, X_test, y_train, y_test) = pickle.load(f)
    return X_train, X_test, y_train, y_test


def process(
        location: config_diabetes.Location = config_diabetes.Location
):
    """Flow to process the Data
    """
    data = get_raw_data(location.data_raw)

    # Cleaning the data process:

    # 1. Null values' handler. There are no null values in DB.
    # print(data.isnull().sum())
    # data.dropna(inplace=True)
    # print(data.isnull().sum())

    # 2. All the features are required for the ml process

    # 3. Convert to numeric Outcome feature
    # 0 - no diabetes. 1 - no diabetes.
    data[config_diabetes.ProcessConfig.label] = pd.to_numeric(data[config_diabetes.ProcessConfig.label],
                                                              errors='coerce')

    processed = data
    processed = pd.get_dummies(processed)
    # After cleaning and processing the database, display general statistics of dataset
    # print(processed.describe())

    X, y = get_X_y(processed, config_diabetes.ProcessConfig.label)
    over_sample = SMOTE()
    X_ros, Y_ros = over_sample.fit_resample(X, y)
    split_data = split_train_test(X_ros, Y_ros, config_diabetes.ProcessConfig.test_size)
    save_processed_data(split_data, config_diabetes.Location.data_process)


def getProcessedData(file_path: str):
    # read python dict back from the file
    with open(file_path, 'rb') as f:
        split_dict = pickle.load(f)

    X_train = split_dict["X_train"]
    X_test = split_dict["X_test"]
    y_train = split_dict["y_train"]
    y_test = split_dict["y_test"]
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    process()
    X_train, X_test, y_train, y_test = getProcessedData(config_diabetes.Location.data_process)
