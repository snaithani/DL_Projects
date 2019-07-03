import numpy as np
from prep import get_data


class NB:
    def __init__(self):
        self.spam = []
        self.ham = []

        self.s = 0
        self.h = 0

    def train(self, X, y):
        n = X.shape[0]

        self.spam = (X[y == 1] / 100).sum(axis=0) / n
        self.ham = (X[y == 0] / 100).sum(axis=0) / n

        self.s = len(X[y == 1])
        self.h = len(X[y == 0])

    def score(self, X, y):
        y_pred = np.argmax(np.vstack((X @ np.log(self.ham) +
                                      np.log(self.h), X @ np.log(self.spam) +
                                      np.log(self.s))),
                           axis=0)
        return (y == y_pred).mean()


if __name__ == '__main__':
    X, y = get_data()
    X_train, y_train = X[:3000], y[:3000]
    X_test, y_test = X[3000:], y[3000:]

    nb = NB()

    nb.train(X_train, y_train)
    score = nb.score(X_test, y_test)
    print(score)
