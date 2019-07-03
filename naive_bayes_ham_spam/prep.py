import pandas as pd
import numpy as np


def get_data():
    data = pd.read_csv('data/spambase.data').values
    np.random.shuffle(data)
    X = data[:, :48]
    y = data[:, -1]

    return X, y


if __name__ == '__main__':
    X, y = get_data()
    print(X[0, :48].sum())
    print(X.shape, y.shape)
