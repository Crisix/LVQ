import math

import numpy as np
from numba import cuda
from numpy.random.mtrand import RandomState
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.datasets import make_blobs
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_X_y, check_is_fitted


# from MyGMLVQ import MyGMLVQ


def blobs_dataset(samples=1_000):
    N = samples
    D = 2  # dimensions / features
    M = 3  # number or prototypes
    X, y = make_blobs(n_samples=N, centers=M, n_features=D, random_state=2)  # 2,3
    # X[:, 1] *= 20
    return X, y


@cuda.jit(device=True)
def sigmoid(x):
    return 1. / (1. + np.exp(-x))


class MyBFGSCudaLVQ(BaseEstimator, ClassifierMixin):
    """
    https://www.in.tu-clausthal.de/fileadmin/homes/techreports/ifi0614biehl.pdf
    Generalized matrix LVQ
    """

    def __init__(self, iterations, gtol=1e-5):
        self.iterations = iterations
        self.gtol = gtol

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

        # self.w_ = np.zeros_like(self.w_)

        # cuda things
        stream = cuda.stream()
        rows = X.shape[0]
        num_features = X.shape[1]
        float_X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
        int_y = np.ascontiguousarray(np.asarray(y, dtype=np.int64))
        dev_X = cuda.to_device(float_X, stream=stream)
        dev_y = cuda.to_device(int_y, stream=stream)

        block_dim = 128
        grid_dim = int(rows / block_dim)

        def opt(vars):
            # print(vars)
            dev_w = cuda.to_device(vars.reshape(-1, num_features))

            dev_w_grad = cuda.to_device(np.zeros_like(self.w_))
            dev_cost = cuda.to_device(np.zeros(1))

            cuda_fit_iteration[grid_dim, block_dim](dev_w,
                                                    dev_X, dev_y,
                                                    1.,
                                                    self.omega, self.omega @ self.omega,
                                                    self.classes_,
                                                    dev_w_grad, dev_cost)

            w_grad = dev_w_grad.copy_to_host()
            cost = dev_cost.copy_to_host()[0]

            return cost, - w_grad.reshape(-1)

        options = {'disp': True, 'gtol': self.gtol, 'maxiter': self.iterations}
        res = minimize(fun=opt, jac=True, method='l-bfgs-b', x0=self.w_.reshape(-1), options=options)
        print(f"Number of Iterations: {res.nit}")
        self.w_ = res.x.reshape(-1, num_features)
        self.n_iter_ = res.nit
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


@cuda.jit(f"void(float64[:, :], float64[:, :], int64[:], float64, float64[:, :], float64[:, :], int64[:], float64[:, :], float64[:])")
def cuda_fit_iteration(w: np.ndarray,
                       X: np.ndarray, y: np.ndarray,
                       lr: float,
                       omega: np.ndarray, Lambda: np.ndarray,
                       classes: np.ndarray,
                       w_grad, cost_result):
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

        cuda.atomic.add(cost_result, 0, sigmoid_of_mu)

        dfdmu = (1 - sigmoid_of_mu)  # dfdmu = 1  # if Phi is identity (?)

        mu_plus = 2 * dK / ((dJ + dK) ** 2)
        mu_minus = 2 * dJ / ((dJ + dK) ** 2)

        # vorher: w[wJ_pt_idx] += lr * dfdmu * mu_plus * Lambda @ (x_i - w[wJ_pt_idx])
        multiplier = lr * dfdmu * mu_plus
        for ft_idx in range(w.shape[1]):
            matmul = 0
            for mm_idx in range(Lambda.shape[0]):
                matmul += Lambda[ft_idx, mm_idx] * (x_i[mm_idx] - w[wJ_pt_idx, mm_idx])
            # w[wJ_pt_idx, ft_idx] += multiplier * matmul
            cuda.atomic.add(w_grad, (wJ_pt_idx, ft_idx), multiplier * matmul)  # equal to non-atomic: w_grad[wJ_pt_idx, ft_idx] += multiplier * matmul

        # vorher: w[wK_pt_idx] -= lr * dfdmu * mu_minus * Lambda @ (x_i - w[wK_pt_idx])
        multiplier = lr * dfdmu * mu_minus
        for ft_idx in range(w.shape[1]):
            matmul = 0
            for mm_idx in range(Lambda.shape[0]):
                matmul += Lambda[ft_idx, mm_idx] * (x_i[mm_idx] - w[wK_pt_idx, mm_idx])
            cuda.atomic.add(w_grad, (wK_pt_idx, ft_idx), -multiplier * matmul)  # equal to non-atomic: w_grad[wK_pt_idx, ft_idx] -= multiplier * matmul
