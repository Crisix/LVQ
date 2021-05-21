import numpy as np
import pycuda.driver as cuda
from numpy.random.mtrand import RandomState
from pycuda.compiler import SourceModule
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_X_y, check_is_fitted

mod = SourceModule("""
__global__
void cuda_fit_iteration(int num_protos, int num_features, int num_samples,
                        float *w,
                        float *X, int *y,
                        float lr_w, float lr_omega,
                        float *omega, float *Lambda, int *classes,
                        float *w_grad, float *omega_grad, float *cost_result) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < num_samples) {

        float same_class_dist = 0.0; // more TODOs
        float other_clas_dist = 0.0;
        int same_class_pt_idx = -1; // TODO -1;
        int other_clas_pt_idx = -1; // TODO -1;
        for (int proto_idx = 0; proto_idx < num_protos; ++proto_idx) {
            float dist = 0;
            for (int ab_i = 0; ab_i < num_features; ++ab_i) {
                float x_i_minus_pt_i = (X[i * num_features + ab_i] - w[proto_idx * num_features + ab_i]);
                float tmp = 0;
                for (int m = 0; m < num_features; ++m) {
                    tmp += x_i_minus_pt_i * omega[ab_i * num_features + m];
                }
                dist += tmp * tmp;
            }

            if (y[i] == classes[proto_idx] && (dist < same_class_dist || same_class_pt_idx == -1)) {
                same_class_dist = dist;
                same_class_pt_idx = proto_idx;
            }
            if (y[i] != classes[proto_idx] && (dist < other_clas_dist || other_clas_pt_idx == -1)) {
                other_clas_dist = dist;
                other_clas_pt_idx = proto_idx;
            }
        }

        int wJ_pt_idx = same_class_pt_idx;
        int wK_pt_idx = other_clas_pt_idx;

        float dJ = same_class_dist;
        float dK = other_clas_dist;

        float mu_sample = (dJ - dK) / (dJ + dK);
        float sigmoid_of_mu = 1. / (1. + exp(-mu_sample));


        atomicAdd(&cost_result[0], sigmoid_of_mu);


        float dfdmu = (1 - sigmoid_of_mu);

        float mu_plus = 2 * dK / ((dJ + dK) * (dJ + dK));
        float mu_minus = 2 * dJ / ((dJ + dK) * (dJ + dK));

        if (lr_w > 0) {
            float multiplier1 = lr_w * dfdmu * mu_plus;
            for (int ft_idx = 0; ft_idx < num_features; ++ft_idx) {
                float matmul = 0;
                for (int mm_idx = 0; mm_idx < num_features; ++mm_idx) {
                    matmul += Lambda[ft_idx * num_features + mm_idx] * (X[i * num_features + mm_idx] - w[wJ_pt_idx * num_features + mm_idx]);
                }
                atomicAdd(&w_grad[wJ_pt_idx * num_features + ft_idx], multiplier1 * matmul);
            }

            float multiplier2 = lr_w * dfdmu * mu_minus;
            for (int ft_idx = 0; ft_idx < num_features; ++ft_idx) {
                float matmul = 0;
                for (int mm_idx = 0; mm_idx < num_features; ++mm_idx) {
                    matmul += Lambda[ft_idx * num_features + mm_idx] * (X[i * num_features + mm_idx] - w[wK_pt_idx * num_features + mm_idx]);
                }
                atomicAdd(&w_grad[wK_pt_idx * num_features + ft_idx], -multiplier2 * matmul);
            }
        }

        if (lr_omega > 0) {

            extern __shared__ float shared_mem[];  // num_features * 2 for omega[l,*] and omega[m,*]
            for (int l = 0; l < num_features; ++l) {

                __syncthreads();
                if (threadIdx.x < num_features) {
                    shared_mem[threadIdx.x] = omega[l * num_features + threadIdx.x];
                }

                for (int m = 0; m < num_features; ++m) {

                    __syncthreads();
                    if (threadIdx.x < num_features) {
                        shared_mem[num_features + threadIdx.x] = omega[m * num_features + threadIdx.x];
                    }
                    __syncthreads();

                    float omega_eps_m_w_jm = 0;
                    float omega_eps_l_w_jl = 0;
                    float omega_eps_m_w_km = 0;
                    float omega_eps_l_w_kl = 0;

                    for (int o = 0; o < num_features; ++o) {

                        float omega_l_o = shared_mem[o];
                        float omega_m_o = shared_mem[num_features + o];

                        float x_i_o = X[i * num_features + o];
                        float w_j_o = w[wJ_pt_idx * num_features + o];
                        float w_k_o = w[wK_pt_idx * num_features + o];

                        omega_eps_m_w_jm += omega_m_o * (x_i_o - w_j_o);
                        omega_eps_l_w_jl += omega_l_o * (x_i_o - w_j_o);
                        omega_eps_m_w_km += omega_m_o * (x_i_o - w_k_o);
                        omega_eps_l_w_kl += omega_l_o * (x_i_o - w_k_o);

                    }

                    float fst = omega_eps_m_w_jm * (X[i * num_features + l] - w[wJ_pt_idx * num_features + l]) + \
                                omega_eps_l_w_jl * (X[i * num_features + m] - w[wJ_pt_idx * num_features + m]);
                    float snd = omega_eps_m_w_km * (X[i * num_features + l] - w[wK_pt_idx * num_features + l]) + \
                                omega_eps_l_w_kl * (X[i * num_features + m] - w[wK_pt_idx * num_features + m]);

                    atomicAdd(&omega_grad[l * num_features + m], lr_omega * dfdmu * (mu_plus * fst - mu_minus * snd));
                }
            }
        }
    }
}

""", options=["-lineinfo"])

cuda_fit_iteration = mod.get_function("cuda_fit_iteration")


class MyBFGSCudaGMLVQ_PyCuda(BaseEstimator, ClassifierMixin):
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
        self.w_ = np.empty([M, D], dtype=np.float)
        self.c_w_ = np.empty([M], dtype=self.classes_.dtype)
        pos = 0
        X, y = check_X_y(X, y)

        for actClass in range(M):  # prototype initialization adapted from sklearn-lvq
            nb_prot = nb_ppc[actClass]
            mean = np.mean(X[y == self.classes_[actClass], :], 0)
            self.w_[pos:pos + nb_prot] = mean + (random_state.rand(nb_prot, D) * 2 - 1)
            self.c_w_[pos:pos + nb_prot] = self.classes_[actClass]
            pos += nb_prot

        # self.w_ = np.zeros_like(self.w_, dtype=np.float32)  # TODO remove sometime
        # self.history.append(self.w_.copy())

        rows = X.shape[0]
        num_features = X.shape[1]
        X = np.asarray(X, dtype=np.float32)
        int_y = np.asarray(y, dtype=np.int32)

        dev_X = cuda.mem_alloc(X.nbytes)
        dev_y = cuda.mem_alloc(int_y.nbytes)

        cuda.memcpy_htod(dev_X, X)
        cuda.memcpy_htod(dev_y, int_y)

        block_dim = 128
        grid_dim = int(rows / block_dim)

        if num_features > block_dim:
            raise ValueError(f"{num_features} features need {4 * num_features} threads per block but only {block_dim} were given")

        def opt(vars, lr_w, lr_omega):

            vars = vars.reshape(-1, num_features).astype(np.float32)
            w, omega = vars[:-num_features], vars[-num_features:]
            w_grad = np.zeros_like(w, dtype=np.float32)
            omega_grad = np.zeros_like(omega, dtype=np.float32)
            cost = np.zeros(shape=(1,), dtype=np.float32)
            classes = np.array(self.classes_, dtype=np.int32)

            dev_w = cuda.mem_alloc(w.nbytes)
            dev_omega = cuda.mem_alloc(omega.nbytes)
            dev_lambda = cuda.mem_alloc(omega.nbytes)
            dev_w_grad = cuda.mem_alloc(w_grad.nbytes)
            dev_omega_grad = cuda.mem_alloc(omega.nbytes)
            dev_cost = cuda.mem_alloc(np.zeros(shape=1).nbytes)
            dev_classes = cuda.mem_alloc(classes.nbytes)

            cuda.memcpy_htod(dev_w, w)
            cuda.memcpy_htod(dev_omega, omega)
            cuda.memcpy_htod(dev_lambda, omega @ omega)
            cuda.memcpy_htod(dev_w_grad, w_grad)
            cuda.memcpy_htod(dev_omega_grad, omega_grad)
            cuda.memcpy_htod(dev_cost, cost)
            cuda.memcpy_htod(dev_classes, classes)

            cuda_fit_iteration(np.int32(self.w_.shape[0]), np.int32(self.omega_.shape[0]), np.int32(X.shape[0]),
                               dev_w,
                               dev_X, dev_y,
                               np.float32(lr_w), np.float32(lr_omega),
                               dev_omega, dev_lambda,
                               dev_classes,
                               dev_w_grad, dev_omega_grad, dev_cost,
                               shared=num_features * np.float32().itemsize * 2,
                               block=(block_dim, 1, 1), grid=(grid_dim, 1))

            cuda.memcpy_dtoh(w_grad, dev_w_grad)
            cuda.memcpy_dtoh(omega_grad, dev_omega_grad)
            cuda.memcpy_dtoh(cost, dev_cost)

            all_grads = -np.append(w_grad, omega_grad, axis=0).reshape(-1).astype(np.float64)
            # all_grads = all_grads * (1 + 0.0001 * (random_state.rand(*all_grads.shape) - 0.5))
            # print(cost)
            # print(all_grads)
            return cost[0], all_grads

        options = {'disp': False, 'gtol': self.gtol, 'maxiter': self.iterations}
        self.n_iter_ = 0

        def cb(x):
            x = x.reshape(-1, num_features)
            tmp_w, tmp_omega = x[:-num_features], x[-num_features:]
            self.history.append(tmp_w.copy())

        for (lr_w, lr_omega) in [[1, 0], [0, 1], [1, 1]]:
            x0 = np.append(self.w_, self.omega_, axis=0).reshape(-1).astype(np.float32)
            res = minimize(fun=lambda v: opt(v, lr_w=lr_w, lr_omega=lr_omega), jac=True, method='l-bfgs-b', x0=x0, options=options, callback=cb)
            self.n_iter_ = max(res.nit, self.n_iter_)
            vars = res.x.reshape(-1, num_features)
            self.w_, self.omega_ = vars[:-num_features], vars[-num_features:]
            self.omega_ /= np.sqrt(np.trace(self.omega_ @ self.omega_))

        # print("w")
        # print(self.w_)
        # print("omega")
        # print(self.omega_)

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
