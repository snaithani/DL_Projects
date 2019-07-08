from prep import get_data, KMeansSoft
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans


def get_cost(m, X, responsibilities):
    k = len(m)
    dist = np.array([[x] * k for x in X])
    dist = (np.sum(((dist - m) ** 2), axis=2) ** 0.5) * responsibilities
    return dist.sum()


if __name__ == '__main__':
    X, y = get_data(num_clusters=10, num_samples=200, num_features=100)
    costs = []
    for k in [1, 2, 3, 5, 8, 15, 30]:
        kmeans = KMeansSoft(k)
        KMeans_2 = KMeans(n_clusters=k).fit(X)
        y_pred, m, responsibilities = kmeans.fit(X, 5)
        cost = get_cost(m, X, responsibilities)
        costs.append(cost)
    plt.plot(costs)
