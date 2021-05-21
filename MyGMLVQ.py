import numpy as np
from numba import jit, prange
from numpy.random.mtrand import RandomState
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_X_y, check_is_fitted


@jit(nopython=True)
def sigmoid(x):
    return 1. / (1. + np.exp(-x))


@jit(nopython=True, parallel=True)
def pairwise_distance(X, prototypes, omega):
    distance = np.zeros((prototypes.shape[0], X.shape[0]))
    for i in prange(prototypes.shape[0]):
        distance[i] = np.sum(((X - prototypes[i]) @ omega.T) ** 2, axis=1)
    return distance.T


# @jit(nopython=True, cache=True, fastmath=True)
@jit(nopython=True)
def _fit(network: np.ndarray, X: np.ndarray, y: np.ndarray, lr, matrix_lr, iter, omega, classes: np.ndarray, history):
    # network = np.zeros(shape=(M, D))
    for e in range(iter):
        history[e] = network.copy()
        omega_grad_after = np.zeros_like(omega)

        for i in range(X.shape[0]):
            x_i, y_i = X[i], y[i]

            dists = np.zeros(network.shape[0])
            for an_idx in range(network.shape[0]):
                dists[an_idx] = (x_i - network[an_idx]) @ omega @ omega.T @ (x_i - network[an_idx]).T

            pt_idx_sorted_by_dist = np.argsort(dists)
            same_class = y_i == classes

            wJ_idx, wK_idx = None, None
            for p_idx in pt_idx_sorted_by_dist:
                if same_class[p_idx] and wJ_idx is None:
                    wJ_idx = p_idx
                if not same_class[p_idx] and wK_idx is None:
                    wK_idx = p_idx
                if wJ_idx is not None and wK_idx is not None:
                    break
            wJ_idx, wK_idx = int(wJ_idx), int(wK_idx)

            tmp1 = (x_i - network[wJ_idx]).T @ omega
            tmp2 = (x_i - network[wK_idx]).T @ omega
            dJ = tmp1 @ tmp1
            dK = tmp2 @ tmp2

            mu_sample = (dJ - dK) / (dJ + dK)
            dfdmu = (1 - sigmoid(mu_sample))
            # dfdmu = 1  # if Phi is identity (?)

            mu_plus = 2 * dK / ((dJ + dK) ** 2)
            mu_minus = 2 * dJ / ((dJ + dK) ** 2)
            Lambda = omega @ omega
            network[wJ_idx] += lr * dfdmu * mu_plus * Lambda @ (x_i - network[wJ_idx])
            network[wK_idx] -= lr * dfdmu * mu_minus * Lambda @ (x_i - network[wK_idx])

            w = network
            wJ_pt_idx = wJ_idx
            wK_pt_idx = wK_idx

            eps = x_i
            wJ = w[wJ_pt_idx]
            wK = w[wK_pt_idx]
            for l in range(omega.shape[0]):
                for m in range(omega.shape[1]):
                    fst = (omega @ (eps - wJ))[m] * (eps[l] - wJ[l]) + \
                          (omega @ (eps - wJ))[l] * (eps[m] - wJ[m])
                    snd = (omega @ (eps - wK))[m] * (eps[l] - wK[l]) + \
                          (omega @ (eps - wK))[l] * (eps[m] - wK[m])
                    omega_grad_after[l, m] = -matrix_lr * dfdmu * (mu_plus * fst - mu_minus * snd)

            # for l in range(omega.shape[0]):
            #     for m in range(omega.shape[1]):
            #         fst = (omega @ (eps - wJ))[m] * (eps[l] - wJ[l]) + \
            #               (omega @ (eps - wJ))[l] * (eps[m] - wJ[m])
            #         snd = (omega @ (eps - wK))[m] * (eps[l] - wK[l]) + \
            #               (omega @ (eps - wK))[l] * (eps[m] - wK[m])
            #         omega_grad_before[l, m] += -matrix_lr * dfdmu * (mu_plus * fst - mu_minus * snd)
            #         # omega[l, m] += -matrix_lr * dfdmu * (mu_plus * fst - mu_minus * snd)
            #
            #         omega_eps_m_w_jm = 0
            #         omega_eps_l_w_jl = 0
            #         omega_eps_m_w_km = 0
            #         omega_eps_l_w_kl = 0
            #         for o in range(omega.shape[0]):  # matmul loop
            #             omega_eps_m_w_jm += omega[m, o] * (eps[o] - w[wJ_pt_idx, o])
            #             omega_eps_l_w_jl += omega[l, o] * (eps[o] - w[wJ_pt_idx, o])
            #             omega_eps_m_w_km += omega[m, o] * (eps[o] - w[wK_pt_idx, o])
            #             omega_eps_l_w_kl += omega[l, o] * (eps[o] - w[wK_pt_idx, o])
            #
            #         fst = omega_eps_m_w_jm * (eps[l] - w[wJ_pt_idx, l]) + \
            #               omega_eps_l_w_jl * (eps[m] - w[wJ_pt_idx, m])
            #         snd = omega_eps_m_w_km * (eps[l] - w[wK_pt_idx, l]) + \
            #               omega_eps_l_w_kl * (eps[m] - w[wK_pt_idx, m])
            #         omega_grad_after[l, m] += (1 / X.shape[0]) * -matrix_lr * dfdmu * (mu_plus * fst - mu_minus * snd)

                    # omega += (1 / X.shape[0]) * omega_grad_before
            # print(np.sum(np.abs(omega_grad_before - omega_grad_after)))
        omega += omega_grad_after  # * (1 / X.shape[0])
    omega /= np.sqrt(np.trace(omega @ omega))

    return network


class MyGMLVQ(BaseEstimator, ClassifierMixin):
    """
    https://www.in.tu-clausthal.de/fileadmin/homes/techreports/ifi0614biehl.pdf
    Generalized matrix LVQ
    """

    def __init__(self, iterations, lr, matrix_lr):
        self.iterations = iterations
        self.lr = lr
        self.matrix_lr = matrix_lr
        self.n_iter_ = -1

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
        self.w_ = _fit(self.w_, X, y, self.lr, self.matrix_lr, self.iterations, self.omega, self.classes_, self.history)
        return self

    def predict(self, X: np.ndarray):
        check_is_fitted(self)
        # X = check_array(X)

        # begin = time.time()
        # closest = np.argmin(euclidean_distances(X, self.w_), axis=1)
        # end = time.time()
        # print(f"euclidean: {end - begin}")
        # return self.classes_[closest]

        # begin = time.time()
        res = []
        for i in range(X.shape[0]):
            x_i = X[i]
            dists = np.zeros(self.w_.shape[0])
            for p in range(self.w_.shape[0]):
                dists[p] = (x_i - self.w_[p]) @ self.omega @ self.omega.T @ (x_i - self.w_[p]).T
            res.append(np.argmin(dists))
        res = np.array(res)
        # end = time.time()
        # print(f"matrix: {end - begin}")
        return self.classes_[res]



"""
blobs: 30 samples, 2 pts, 2 features
	GmlvqModel(beta=10, max_iter=10000)      Duration=0.00327 Accuracy=100.000 iters=0.00 duration/iters=-1.00
	MyGMLVQ(iterations=1, lr=0.1, matrix_lr=0.1) Duration=0.18098 Accuracy=46.667 iters=-1.00 duration/iters=-0.18
	MyGMLVQ(iterations=10000, lr=0.1, matrix_lr=0.1) Duration=38.28023 Accuracy=96.667 iters=-1.00 duration/iters=-38.28
	MyGMLVQ(iterations=10000, lr=0.01, matrix_lr=0.01) Duration=38.15212 Accuracy=96.667 iters=-1.00 duration/iters=-38.15
	MyGMLVQ(iterations=10000, lr=0.001, matrix_lr=0.001) Duration=38.22048 Accuracy=46.667 iters=-1.00 duration/iters=-38.22
done
blobs: 1000 samples, 20 pts, 10 features
	GmlvqModel(beta=10, max_iter=10000)      Duration=10.29666 Accuracy=100.000 iters=486.00 duration/iters=0.02
	MyGMLVQ(iterations=1, lr=0.1, matrix_lr=0.1) Duration=1.64738 Accuracy=57.400 iters=-1.00 duration/iters=-1.65
"""