# FUCK THIS SHIT; MOST RECENT CUDA, BUT NAN AND DEBUG ARRAY
import math
import time

import matplotlib.pyplot as plt
import numpy as np
from numba import jit, cuda
from numpy.random.mtrand import RandomState
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_X_y, check_is_fitted


def blobs_dataset(samples=1_000):
    N = samples
    D = 2  # dimensions / features
    M = 3  # number or prototypes
    X, y = make_blobs(n_samples=N, centers=M, n_features=D, random_state=2)  # 3
    # X[:, 1] *= 20
    return X, y


@cuda.jit(device=True)
def sigmoid(x):
    return 1. / (1. + np.exp(-x))


@jit(nopython=True)
def pairwise_distance(X, prototypes, omega):
    distance = np.zeros((prototypes.shape[0], X.shape[0]))
    for i in range(prototypes.shape[0]):
        distance[i] = np.sum(((X - prototypes[i]) @ omega.T) ** 2, axis=1)
    return distance.T


class MyCudaGradDescGMLVQ(BaseEstimator, ClassifierMixin):
    """
    https://www.in.tu-clausthal.de/fileadmin/homes/techreports/ifi0614biehl.pdf
    Generalized matrix LVQ
    """

    def __init__(self, iterations, lr, matrix_lr):
        self.iterations = iterations
        self.lr = lr
        self.matrix_lr = matrix_lr

    def fit(self, X, y):

        # initialize prototypes, identical/copy'n'paste from sklearn-lvq to allow better comparison
        random_state = np.random.RandomState(42)
        self.classes_ = unique_labels(y)
        M = len(self.classes_)  # number or prototypes
        D = X.shape[1]  # dimensions / features
        self.omega = np.eye(D)
        self.history = np.zeros(shape=(self.iterations, M, D))

        nb_ppc = np.ones([len(self.classes_)], dtype='int') * 1  # self.prototypes_per_class
        self.w_ = np.empty([M, D], dtype=np.double)
        self.c_w_ = np.empty([M], dtype=self.classes_.dtype)
        pos = 0
        X, y = check_X_y(X, y)

        for actClass in range(M):  # prototype initialization adapted from sklearn-lvq
            nb_prot = nb_ppc[actClass]
            mean = np.mean(X[y == self.classes_[actClass], :], 0)
            self.w_[pos:pos + nb_prot] = mean + (random_state.rand(nb_prot, D) * 2 - 1)
            self.c_w_[pos:pos + nb_prot] = self.classes_[actClass]
            pos += nb_prot
        self.w_ = np.zeros_like(self.w_)

        # cuda things
        rows = X.shape[0]
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)  # TODO change type to int
        dev_X = cuda.to_device(X)
        dev_y = cuda.to_device(y)
        dev_w = cuda.to_device(self.w_)

        block_dim = 1024
        grid_dim = int(rows / block_dim)
        for e in range(self.iterations):
            dev_omega_change = cuda.device_array(self.w_.shape)
            debug = np.zeros(shape=X.shape[0])
            debug = cuda.to_device(debug)
            cuda_fit_iteration[grid_dim, block_dim](dev_w, dev_X, dev_y, self.lr, self.matrix_lr, self.omega, self.omega @ self.omega, self.classes_, dev_omega_change, debug)
            self.w_ = dev_w.copy_to_host()  # TODO only in the last loop
            # print(f"Iteration {e}", self.w_)
            self.history[e] = self.w_.copy()
            debug = debug.copy_to_host()
            #print("DEBUG:", debug.shape)
            #print(debug)
            # omega_change = dev_omega_change.copy_to_host(stream=stream)
            # self.omega += omega_change
            # self.omega /= np.sqrt(np.trace(self.omega @ self.omega))

        return self

    def predict(self, X: np.ndarray):
        check_is_fitted(self)
        res = []
        for i in range(X.shape[0]):
            x_i = X[i]
            dists = np.zeros(self.w_.shape[0])
            for p in range(self.w_.shape[0]):
                dists[p] = (x_i - self.w_[p]) @ self.omega @ self.omega.T @ (x_i - self.w_[p]).T
            res.append(np.argmin(dists))
        return self.classes_[np.array(res)]


@cuda.jit(device=True)
def calc_distance(a, b, omega):
    # return (x_i - w[pt_idx]) @ omega @ omega.T @ (x_i - w[pt_idx]).T
    result = 0
    for ab_i in range(a.shape[0]):
        ab = (a[ab_i] - b[ab_i])
        tmp = 0
        for m in range(omega.shape[0]):
            tmp += ab * omega[ab_i, m]
        result += tmp ** 2
    return result


@cuda.jit(
    f"void(       float64[:, :], float64[:, :],  int64[:],     float64,     float64,         float64[:, :],    float64[:, :],        int64[:],           float64[:, :], float64[:])")
def cuda_fit_iteration(w: np.ndarray, X: np.ndarray, y: np.ndarray, lr: float, matrix_lr: float, omega: np.ndarray, Lambda: np.ndarray, classes: np.ndarray,
                       omega_change: np.ndarray, debug):
    i = cuda.grid(1)

    if i < X.shape[0]:
        x_i = X[i]
        y_i = y[i]

        same_class_dist = np.inf
        other_clas_dist = np.inf
        same_class_pt_idx = -1
        other_clas_pt_idx = -1
        for pt_idx in range(w.shape[0]):
            dist = calc_distance(x_i, w[pt_idx], omega)
#            debug[20*pt_idx + i] = dist
            if y_i == classes[pt_idx] and dist < same_class_dist:
                same_class_dist = dist
                same_class_pt_idx = pt_idx
            if y_i != classes[pt_idx] and dist < other_clas_dist:
                other_clas_dist = dist
                other_clas_pt_idx = pt_idx
        wJ_pt_idx = same_class_pt_idx
        wK_pt_idx = other_clas_pt_idx
        dJ = same_class_dist
        dK = other_clas_dist


        mu_sample = (dJ - dK) / (dJ + dK)
        sigmoid_of_mu = 1. / (1. + math.exp(-mu_sample))
        dfdmu = (1 - sigmoid_of_mu)  # dfdmu = 1  # if Phi is identity (?)

        mu_plus = 2 * dK / ((dJ + dK) ** 2)
        mu_minus = 2 * dJ / ((dJ + dK) ** 2)

        # vorher: w[wJ_pt_idx] += lr * dfdmu * mu_plus * Lambda @ (x_i - w[wJ_pt_idx])
        multiplier = lr * dfdmu * mu_plus
        for ft_idx in range(w.shape[1]):
            matmul = 0
            for mm_idx in range(Lambda.shape[0]):
                matmul += Lambda[ft_idx, mm_idx] * (x_i[mm_idx] - w[wJ_pt_idx, mm_idx])
            w[wJ_pt_idx, ft_idx] += multiplier * matmul

        # vorher: w[wK_pt_idx] -= lr * dfdmu * mu_minus * Lambda @ (x_i - w[wK_pt_idx])
        multiplier = lr * dfdmu * mu_minus
        for ft_idx in range(w.shape[1]):
            matmul = 0
            for mm_idx in range(Lambda.shape[0]):
                matmul += Lambda[ft_idx, mm_idx] * (x_i[mm_idx] - w[wK_pt_idx, mm_idx])
            w[wK_pt_idx, ft_idx] -= multiplier * matmul

        # eps = x_i
        # wJ = w[wJ_pt_idx]
        # wK = w[wK_pt_idx]
        # for l in range(omega.shape[0]):
        #     for m in range(omega.shape[1]):
        #         fst = (omega @ (eps - wJ))[m] * (eps[l] - wJ[l]) + \
        #               (omega @ (eps - wJ))[l] * (eps[m] - wJ[m])
        #         snd = (omega @ (eps - wK))[m] * (eps[l] - wK[l]) + \
        #               (omega @ (eps - wK))[l] * (eps[m] - wK[m])
        #         omega_change[l, m] = -matrix_lr * dfdmu * (mu_plus * fst - mu_minus * snd)


# ex: (1797, 64), (10, 64) !


def test_performance(cls, plot=True):
    duration = 0.0
    tries = 1
    for _ in range(tries):
        begin = time.time()
        cls.fit(X, y)
        end = time.time()
        duration += end - begin
    duration = duration / tries
    y_pred = cls.predict(X)
    title = f"{str(cls):<50} Duration={duration:.5f} Accuracy={accuracy_score(y, y_pred):.3f}"
    print(title)
    print(cls.w_)
    omega = None
    try:
        # print(cls.omega)
        omega = cls.omega
    except:
        # print(cls.omega_)
        omega = cls.omega_

    if plot:

        def project(omega, x, dims, print_variance_covered=False):  # adapted from sklearn-lvq GmlvqModel
            v, u = np.linalg.eig(omega.conj().T.dot(omega))
            idx = v.argsort()[::-1]
            if print_variance_covered:
                print('variance coverd by projection:', v[idx][:dims].sum() / v.sum() * 100)
            return x.dot(u[:, idx][:, :dims].dot(np.diag(np.sqrt(v[idx][:dims]))))

        plt.figure(figsize=(10, 10))
        # plt.subplot(1, 2, 1)
        plt.title(title)
        plt.scatter(X[:, 0], X[:, 1], c=y)
        plt.scatter(cls.w_[:, 0], cls.w_[:, 1], s=2000, marker='*', linewidth=3, edgecolor='r', c=cls.classes_)
        plt.axis('equal')
        try:
            for p in range(cls.history.shape[1]):
                plt.plot(cls.history[:, p, 0], cls.history[:, p, 1])
        except:
            pass

        # plt.subplot(1, 2, 2)
        # p = project(omega, X, 2, print_variance_covered=True)
        # plt.scatter(p[:, 0], p[:, 1], c=y)
        # plt.axis('equal')
        # plt.suptitle(title)

        plt.show()


X, y = blobs_dataset(10_000)

LR = 0.2  # 0.1
MATRIX_LR = 0.001
# for ITER in [0, 1, 100, 5000]:
for ITER in [2000]:
    print(f"{'':<50} ------ {ITER} ------")
    for _ in range(1):
        models = [
            MyCudaGradDescGMLVQ(iterations=ITER, lr=LR, matrix_lr=MATRIX_LR),
            # GmlvqModel(max_iter=ITER, gtol=1e-300),
            # MyGMLVQ(ITER, LR, MATRIX_LR),
            # MyGMLVQ(0, LR, MATRIX_LR),
        ]
        [test_performance(m) for m in models]
        print("")
