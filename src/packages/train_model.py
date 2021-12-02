from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


def run_model_training(X_train, X_test, y_train, y_test):

    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    clf.score(X_test, y_test)
    y_pred = clf.predict(X_test)

    print("\n3. Model Performance: \n")
    print(classification_report(y_test, y_pred))

    return clf
