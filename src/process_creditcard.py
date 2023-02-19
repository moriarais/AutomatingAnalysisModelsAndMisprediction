"""Python script to process the data"""

import joblib
import pandas as pd

import os, sys; sys.path.append(os.path.dirname(os.path.realpath(__file__)))
import config.config_creditcard as config_creditcard

def get_raw_data(data_location: str):
    """Read raw data

    Parameters
    ----------
    data_location : str
        The location of the raw data
    """
    return pd.read_csv(data_location)



def drop_columns(data: pd.DataFrame, columns: list):
    """Drop unimportant columns

    Parameters
    ----------
    data : pd.DataFrame
        Data to process
    columns : list
        Columns to drop
    """
    return data.drop(columns=columns)



def get_X_y(data: pd.DataFrame, label: str):
    """Get features and label

    Parameters
    ----------
    data : pd.DataFrame
        Data to process
    label : str
        Name of the label
    """
    X = data.drop(columns=label)
    y = data[label]
    return X, y


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



def process(
    location: config_creditcard.Location = config_creditcard.Location()
):
    """Flow to process the ata

    Parameters
    ----------
    location : Location, optional
        Locations of inputs and outputs, by default Location()
    config : ProcessConfig, optional
        Configurations for processing data, by default ProcessConfig()
    """
    data = get_raw_data(location.data_raw)
    processed = drop_columns(data, config_creditcard.ProcessConfig().drop_columns)
    # X, y = get_X_y(processed, config.label)
    # split_data = split_train_test(X, y, config.test_size)
    # save_processed_data(split_data, location.data_process)

