"""Python script to process the data"""

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import pickle
import config.config_creditcard as config_creditcard
import process.process_utils as process_utils
# import AutomatingAnalysisModelsAndMisprediction.src.config.config_creditcard as config_creditcard


def process(
        location: config_creditcard.Location = config_creditcard.Location
):
    """Flow to process the Data
    """
    data = process_utils.get_raw_data(location.data_raw)

    # Cleaning the data process:

    # 1. Null values handler. There are no null values in DB.
    # print(data.isnull().sum())
    # data.dropna(inplace=True)
    # print(data.isnull().sum())

    # 2. All the features are required for the ml process

    # 3. Convert to numeric Class feature
    # 0 - non-fraudulent. 1 - fraudulent.
    data[config_creditcard.ProcessConfig.label] = pd.to_numeric(data[config_creditcard.ProcessConfig.label],
                                                                errors='coerce')

    processed = data
    processed = pd.get_dummies(processed)
    # After cleaning and processing the database, display general statistics of dataset
    # print(processed.describe())

    X, y = get_X_y(processed, config_creditcard.ProcessConfig.label)

    over_sample = SMOTE()
    X_ros, Y_ros = over_sample.fit_resample(X, y)

    split_data = split_train_test(X_ros, Y_ros, config_creditcard.ProcessConfig.test_size)
    save_processed_data(split_data, config_creditcard.Location.data_process)



def getProcessedData(file_path: str):
    # read python dict back from the file
    with open(file_path, 'rb') as f:
        split_dict = pickle.load(f)

    X_train = split_dict["X_train"]
    X_test = split_dict["X_test"]
    y_train = split_dict["y_train"]
    y_test = split_dict["y_test"]
    return X_train, X_test, y_train, y_test

    X, y = process_utils.get_X_y(processed, config_creditcard.ProcessConfig.label)
    split_data = process_utils.split_train_test(X, y, config_creditcard.ProcessConfig.test_size)
    process_utils.save_processed_data(split_data, config_creditcard.Location.data_process)



if __name__ == "__main__":
    process()
    X_train, X_test, y_train, y_test = process_utils.getProcessedData(config_creditcard.Location.data_process)
