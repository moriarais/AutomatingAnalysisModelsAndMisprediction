import pickle
from sklearn.model_selection import train_test_split
import pandas as pd


def getProcessedData(file_path: str):
    # read python dict back from the file
    with open(file_path, 'rb') as f:
        split_dict = pickle.load(f)

    X_train = split_dict["X_train"]
    X_test = split_dict["X_test"]
    y_train = split_dict["y_train"]
    y_test = split_dict["y_test"]
    return X_train, X_test, y_train, y_test


def getUnprocessedData(url: str):
    dataset = pd.read_csv(url, header=None)
    return dataset


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


def get_raw_data(data_location: str):
    """Read raw data
    """
    return pd.read_csv(data_location)
