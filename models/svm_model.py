import numpy as np
import h2o
from h2o.automl import H2OAutoML
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import process.process_creditcard as process_creditcard
import config.config_creditcard as config_creditcard
import process.process_utils as process_utils


def generate_model(X_train, X_test, Y_train, Y_test):
    sc = StandardScaler()

    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # Init an H2O cluster
    h2o.init()

    # Convert training data to H2OFrame
    train = h2o.H2OFrame(np.concatenate((X_train, Y_train.values.reshape(-1, 1)), axis=1))
    dataset = process_utils.getUnprocessedData(config_creditcard.Location.data_raw)
    train.columns = list(dataset.columns)

    # Specify target variable and predictor variables
    x = train.columns[:-1]
    y = train.columns[-1]

    # Run H2O AutoML to automatically select, train and optimize SVM model
    # aml = H2OAutoML(max_models=10, seed=1)
    aml = H2OAutoML(max_models=2, seed=1)
    aml.train(x=x, y=y, training_frame=train)

    # View the leaderboard of trained models
    lb = aml.leaderboard
    print(lb.head())

    # Use the best model to predict on test data
    model = aml.leader
    X_h2o = h2o.H2OFrame(X_test)
    X_h2o.columns = list(dataset.columns)[:-1]
    y_pred = model.predict(X_h2o).as_data_frame().values.flatten()

    # Use the SVM model to flag potential mispredictions
    svm = SVC(kernel='linear')
    svm.fit(X_train, y_train)
    y_pred_svm = svm.predict(X_test)
    misclassified = np.where(Y_test != y_pred_svm)[0]
    print("Indices of potentially misclassified instances: ", misclassified)



if __name__ == "__main__":
    X_train, X_test, y_train, y_test = process_utils.getProcessedData(config_creditcard.Location.data_process)
    # TODO: generate the model for each of the files
    generate_model(X_train, X_test, y_train, y_test)
