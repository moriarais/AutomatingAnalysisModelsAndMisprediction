import process.process_creditcard as process_creditcard
import config.config_creditcard as config_creditcard
from sklearn import svm


def generate_model(X_train, X_test, Y_train, Y_test):
    # Create a svm Classifier
    clf = svm.SVC(kernel='linear')  # Linear Kernel

    # Train the model using the training sets
    clf.fit(X_train, y_train)

    return ''


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = process_creditcard.getProcessedData(config_creditcard.Location.data_process)
    generate_model(X_train, X_test, y_train, y_test)
