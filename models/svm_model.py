import process.process_creditcard as process_creditcard
import config.config_creditcard as config_creditcard
import h2o
from h2o.automl import H2OAutoML
import pandas as pd


def generate_model(X_train, X_test, Y_train, Y_test):
    # Start an H2O cluster
    h2o.init()

    # Convert training and test sets to H2O Frames
    train_h2o = h2o.H2OFrame(pd.concat([X_train, Y_train], axis=1))
    test_h2o = h2o.H2OFrame(pd.concat([X_test, Y_test], axis=1))

    # Define predictors and response
    predictors = train_h2o.col_names[:-1]
    response = train_h2o.col_names[-1]

    # Define AutoML configuration
    aml = H2OAutoML(max_models=20, seed=1234, exclude_algos=["GLM"], verbosity="info")

    # Train the AutoML model
    aml.train(x=predictors, y=response, training_frame=train_h2o)

    # Evaluate the model on the test set
    leader_model = aml.leader
    perf = leader_model.model_performance(test_h2o)
    print(perf)


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = process_creditcard.getProcessedData(config_creditcard.Location.data_process)
    generate_model(X_train, X_test, y_train, y_test)
