from typing import Dict

import pandas as pd
from pandas.core.frame import DataFrame
from pandas.core.series import Series
from sklearn.model_selection import train_test_split


def prepare_data(path_to_data: str) -> Dict:

    # Read data from path
    data = pd.read_csv(path_to_data)

    print("\n1. First 7 rows: \n1")
    print(data.head(7))

    print("\n2. Last 7 rows: \n2")
    print(data.tail(7))

    X = data.drop('Outcome', axis=1)
    y = data['Outcome']

    return {'features': X, 'label': y}


def create_train_test_data(X: DataFrame, y: Series, test_size: any, random_state: any) -> Dict:

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)

    return {'x_train': X_train, 'x_test': X_test, 'y_train': y_train, 'y_test': y_test}
