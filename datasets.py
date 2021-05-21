import numpy as np
from sklearn.datasets import make_blobs


def blobs_dataset(samples=1_000, prototypes=2, features=2):
    N = samples
    D = features  # dimensions / features
    M = prototypes  # number or prototypes
    X, y = make_blobs(n_samples=N, centers=M, n_features=D, random_state=2)  # 2,3
    X[:, 1] *= 10
    return f"blobs: {samples} samples, {prototypes} pts, {features} features", (X, y)


def load_uci_segmentation():
    file = "/Users/christoph/PycharmProjects/LVQ_experiments/matlab/LVQ_toolbox/data/segment.dat"
    data = np.loadtxt(file)
    y = data[:, -1] - 1  # zero based
    X = data[:, :-1]
    X = X[:, np.std(X, axis=0) != 0]  # feature 3 is constant -> exclude it
    return "uci_segmentation", (X, y)