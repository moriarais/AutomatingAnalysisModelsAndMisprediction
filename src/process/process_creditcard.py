"""Python script to process the data"""

import pandas as pd
import config.config_creditcard as config_creditcard
import config.config_utils as config_utils


def process(
        location: config_creditcard.Location = config_creditcard.Location
):
    """Flow to process the Data
    """
    data = config_utils.get_raw_data(location.data_raw)

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

    X, y = config_utils.get_X_y(processed, config_creditcard.ProcessConfig.label)
    split_data = config_utils.split_train_test(X, y, config_creditcard.ProcessConfig.test_size)
    config_utils.save_processed_data(split_data, config_creditcard.Location.data_process)


if __name__ == "__main__":
    process()
    X_train, X_test, y_train, y_test = config_utils.getProcessedData(config_creditcard.Location.data_process)
