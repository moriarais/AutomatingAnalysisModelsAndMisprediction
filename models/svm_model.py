import numpy as np
import h2o
from h2o.automl import H2OAutoML
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc

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
    print("Run H2O AutoML to automatically select, train and optimize SVM model")
    # aml = H2OAutoML(max_models=10, seed=1)
    aml = H2OAutoML(sort_metric='mse', max_runtime_secs=5 * 60, seed=666)
    aml.train(x=x, y=y, training_frame=train)

    # View the leaderboard of trained models
    lb = aml.leaderboard
    print(lb.head())

    # Use the best model to predict on test data
    print("Use the best model to predict on test data")
    model = aml.leader
    X_h2o = h2o.H2OFrame(X_test)
    X_h2o.columns = list(dataset.columns)[:-1]
    y_pred = model.predict(X_h2o).as_data_frame().values.flatten()

    # Train a PySVM SVM model
    print("Train a PySVM SVM model")
    model = LinearSVC(random_state=0, tol=1e-5)
    model.fit(X_train, y_train)
    # Test the model on the test data
    y_pred = model.predict(X_test)
    misclassified = np.where(Y_test != y_pred)[0]
    print("Indices of potentially misclassified instances: ", misclassified)
    # Confusion matrix - summarizing the performance of a classification algorithm.
    cm = confusion_matrix(y_test, y_pred)
    plt.figure()
    plt.imshow(cm, cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.xticks([0, 1], ["Negative", "Positive"])
    plt.yticks([0, 1], ["Negative", "Positive"])
    plt.tight_layout()
    plt.show()

    # ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color="darkorange", lw=2, label="ROC curve (area = %0.2f)" % roc_auc)
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic (ROC)")
    plt.legend(loc="lower right")
    plt.show()
    print("")



if __name__ == "__main__":
    X_train, X_test, y_train, y_test = process_utils.getProcessedData(config_creditcard.Location.data_process)
    # TODO: generate the model for each of the files
    generate_model(X_train, X_test, y_train, y_test)
