from sklearn.datasets.samples_generator import make_blobs
import numpy as np


def get_data(num_clusters=3, num_samples=100, num_features=2):
    X, y = make_blobs(n_samples=num_samples, n_features=num_features,
                      centers=num_clusters)
    return X, y


class KMeansSoft:
    def __init__(self, num_clusters):
        self.num_clusters = num_clusters

    def fit(self, X, beta=2.0):
        n = len(X)

        X = (X - X.mean()) / X.std()

        m = X[np.random.choice(n, self.num_clusters, False)]

        prev = [-1 for _ in range(n)]

        while True:
            responsibilities = np.array([[x] * self.num_clusters for x in X])
            responsibilities = np.exp(-beta *
                                      np.sum((responsibilities - m) ** 2,
                                             axis=2) ** 0.5)
            responsibilities /= np.sum(responsibilities, 1, keepdims=True)
            forw = np.argmax(responsibilities, axis=1)

            if tuple(forw) == tuple(prev):
                break

            prev = forw
            m = (responsibilities.T @ X) / (responsibilities.T.sum(
                axis=1, keepdims=True))

        return prev, m, responsibilities


if __name__ == '__main__':
    X, y = get_data()
    f, g, h = KMeansSoft(4).fit(X)
