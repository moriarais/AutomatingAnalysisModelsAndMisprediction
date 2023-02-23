"""Python script to process the data"""

import pandas as pd

import config.config_health_diabetes as config_diabetes
import process.process_utils as process_utils

def process(
        location: config_diabetes.Location = config_diabetes.Location
):
    """Flow to process the Data
    """
    data = process_utils.get_raw_data(location.data_raw)

    # Cleaning the data process:

    # 1. Null values' handler. There are no null values in DB.
    # print(data.isnull().sum())
    # data.dropna(inplace=True)
    # print(data.isnull().sum())

    # 2. All the features are required for the ml process

    # 3. Convert to numeric Outcome feature
    # 0 - no diabetes. 1 - no diabetes.
    data[config_diabetes.ProcessConfig.label] = pd.to_numeric(data[config_diabetes.ProcessConfig.label], errors='coerce')

    processed = data
    # converting to binary data
    processed = pd.get_dummies(processed[config_diabetes.ProcessConfig.label])
    # After cleaning and processing the database, display general statistics of dataset
    # print(processed.describe())

    X, y = process_utils.get_X_y(processed, config_diabetes.ProcessConfig.label)
    split_data = process_utils.split_train_test(X, y, config_diabetes.ProcessConfig.test_size)
    process_utils.save_processed_data(split_data, config_diabetes.Location.data_process)


if __name__ == "__main__":
    process()
    X_train, X_test, y_train, y_test = process_utils.getProcessedData(config_diabetes.Location.data_process)
