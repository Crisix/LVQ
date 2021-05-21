import math

import numpy as np
from numba import cuda
from numpy.random.mtrand import RandomState
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_X_y, check_is_fitted


@cuda.jit(device=True)
def sigmoid(x):
    return 1. / (1. + np.exp(-x))


class MyBFGSCudaGMLVQ32(BaseEstimator, ClassifierMixin):
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
        self.omega_ = np.eye(D)
        self.omega_ /= np.sqrt(np.trace(self.omega_ @ self.omega_))
        self.history = []

        nb_ppc = np.ones([len(self.classes_)], dtype='int') * 1  # self.prototypes_per_class
        self.w_ = np.empty([M, D], dtype=np.float32)
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

        stream = cuda.stream()
        rows = X.shape[0]
        num_features = X.shape[1]
        float_X = np.ascontiguousarray(np.asarray(X, dtype=np.float32))
        int_y = np.ascontiguousarray(np.asarray(y, dtype=np.int32))
        dev_X = cuda.to_device(float_X, stream=stream)
        dev_y = cuda.to_device(int_y, stream=stream)

        block_dim = 128
        grid_dim = int(rows / block_dim)

        def opt(vars, lr_w, lr_omega):
            # print(vars)
            vars = vars.reshape(-1, num_features).astype(np.float32)
            w, omega = vars[:-num_features], vars[-num_features:]

            dev_omega_grad = cuda.to_device(np.zeros_like(self.omega_, dtype=np.float32))
            dev_w_grad = cuda.to_device(np.zeros_like(self.w_, dtype=np.float32))
            dev_cost = cuda.to_device(np.zeros(1, dtype=np.float32))

            cuda_fit_iteration[grid_dim, block_dim](w,
                                                    dev_X, dev_y,
                                                    lr_w, lr_omega,
                                                    omega, omega @ omega,
                                                    self.classes_,
                                                    dev_omega_grad, dev_w_grad, dev_cost)

            w_grad = dev_w_grad.copy_to_host()
            omega_grad = dev_omega_grad.copy_to_host()
            cost = dev_cost.copy_to_host()[0]
            all_grads = -np.append(w_grad, omega_grad, axis=0).reshape(-1).astype(np.float64)
            all_grads = all_grads * (1 + 0.0001 * (random_state.rand(*all_grads.shape) - 0.5))

            return cost, all_grads

        options = {'disp': True, 'gtol': self.gtol, 'maxiter': self.iterations}
        self.n_iter_ = 0

        def cb(x):
            x = x.reshape(-1, num_features)
            tmp_w, tmp_omega = x[:-num_features], x[-num_features:]
            self.history.append(tmp_w.copy())

        for (lr_w, lr_omega) in [[1, 0], [0, 1], [1, 1]]:
            x0 = np.append(self.w_, self.omega_, axis=0).reshape(-1)
            res = minimize(fun=lambda v: opt(v, lr_w=lr_w, lr_omega=lr_omega), jac=True, method='l-bfgs-b', x0=x0, options=options, callback=cb)
            self.n_iter_ = max(res.nit, self.n_iter_)
            vars = res.x.reshape(-1, num_features)
            self.w_, self.omega_ = vars[:-num_features], vars[-num_features:]
            self.omega_ /= np.sqrt(np.trace(self.omega_ @ self.omega_))

        return self

    def predict(self, X: np.ndarray):
        check_is_fitted(self)
        res = []
        for i in range(X.shape[0]):
            x_i = X[i]
            dists = np.zeros(self.w_.shape[0])
            for p in range(self.w_.shape[0]):
                dists[p] = (x_i - self.w_[p]) @ self.omega_ @ self.omega_.T @ (x_i - self.w_[p]).T
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


@cuda.jit(f"void(float32[:, :], float32[:, :], int32[:], float32, float32, float32[:, :], float32[:, :], int32[:], float32[:, :], float32[:, :], float32[:])")
def cuda_fit_iteration(w: np.ndarray,
                       X: np.ndarray, y: np.ndarray,
                       lr_w: float, lr_omega: float,
                       omega: np.ndarray, Lambda: np.ndarray,
                       classes: np.ndarray,
                       omega_grad: np.ndarray, w_grad: np.ndarray, cost_result: np.ndarray):
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

        # noinspection PyArgumentList
        cuda.atomic.add(cost_result, 0, sigmoid_of_mu)

        dfdmu = (1 - sigmoid_of_mu)  # dfdmu = 1  # if Phi is identity (?)

        mu_plus = 2 * dK / ((dJ + dK) ** 2)
        mu_minus = 2 * dJ / ((dJ + dK) ** 2)

        if lr_w > 0:

            # vorher: w[wJ_pt_idx] += lr * dfdmu * mu_plus * Lambda @ (x_i - w[wJ_pt_idx])
            multiplier = lr_w * dfdmu * mu_plus
            for ft_idx in range(w.shape[1]):
                matmul = 0
                for mm_idx in range(Lambda.shape[0]):
                    matmul += Lambda[ft_idx, mm_idx] * (x_i[mm_idx] - w[wJ_pt_idx, mm_idx])
                # w[wJ_pt_idx, ft_idx] += multiplier * matmul
                # noinspection PyArgumentList
                cuda.atomic.add(w_grad, (wJ_pt_idx, ft_idx), multiplier * matmul)  # equal to non-atomic: w_grad[wJ_pt_idx, ft_idx] += multiplier * matmul

            # vorher: w[wK_pt_idx] -= lr * dfdmu * mu_minus * Lambda @ (x_i - w[wK_pt_idx])
            multiplier = lr_w * dfdmu * mu_minus
            for ft_idx in range(w.shape[1]):
                matmul = 0
                for mm_idx in range(Lambda.shape[0]):
                    matmul += Lambda[ft_idx, mm_idx] * (x_i[mm_idx] - w[wK_pt_idx, mm_idx])
                # noinspection PyArgumentList
                cuda.atomic.add(w_grad, (wK_pt_idx, ft_idx), -multiplier * matmul)  # equal to non-atomic: w_grad[wK_pt_idx, ft_idx] -= multiplier * matmul

        if lr_omega > 0:
            eps = x_i
            for l in range(omega.shape[0]):
                for m in range(omega.shape[1]):
                    omega_eps_m_w_jm = 0
                    omega_eps_l_w_jl = 0
                    omega_eps_m_w_km = 0
                    omega_eps_l_w_kl = 0
                    for o in range(omega.shape[0]):  # matmul loop
                        omega_eps_m_w_jm += omega[m, o] * (eps[o] - w[wJ_pt_idx, o])
                        omega_eps_l_w_jl += omega[l, o] * (eps[o] - w[wJ_pt_idx, o])
                        omega_eps_m_w_km += omega[m, o] * (eps[o] - w[wK_pt_idx, o])
                        omega_eps_l_w_kl += omega[l, o] * (eps[o] - w[wK_pt_idx, o])

                    fst = omega_eps_m_w_jm * (eps[l] - w[wJ_pt_idx, l]) + \
                          omega_eps_l_w_jl * (eps[m] - w[wJ_pt_idx, m])
                    snd = omega_eps_m_w_km * (eps[l] - w[wK_pt_idx, l]) + \
                          omega_eps_l_w_kl * (eps[m] - w[wK_pt_idx, m])
                    # noinspection PyArgumentList
                    cuda.atomic.add(omega_grad, (l, m), -lr_omega * dfdmu * (mu_plus * fst - mu_minus * snd))  # * (1 / X.shape[0])
