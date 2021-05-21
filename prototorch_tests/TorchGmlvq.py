from numba import cuda
from numba.cuda import CudaSupportError

from prototorch_tests.scipy_lbfgs_pytorch import LBFGSScipy

try:
    gpu = cuda.get_current_device()
    print("name = %s" % gpu.name)
except CudaSupportError:
    print("Could not find GPU")

import time

import numpy as np
import torch
import torch.nn.functional as F
from prototorch.functions.distances import squared_euclidean_distance
from prototorch.modules.losses import GLVQLoss
from prototorch.modules.prototypes import Prototypes1D
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.datasets import load_iris, load_wine
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn_lvq import GmlvqModel
from torch.autograd import Variable
import torchvision.datasets as datasets

from datasets import blobs_dataset
from PyTorch_LBFGS.functions.LBFGS import LBFGS, FullBatchLBFGS

"""
https://stackoverflow.com/questions/19041486/how-to-enforce-scipy-optimize-fmin-l-bfgs-b-to-use-dtype-float32
"""

TEST_GRAD_PERFORMANCE = True

PATIENCE = 10000000
HISTORY_SIZE = 10
beta = 2
INPLACE = False
MAX_LS = 100  # DEFAULT 10
batch_size = -1
overlap_ratio = 0.25  # should be in (0, 0.5)


def get_grad(optimizer, X_Sk, y_Sk, opfun, ghost_batch=128):
    """
    Computes objective and gradient of neural network over data sample.

    Implemented by: Hao-Jun Michael Shi and Dheevatsa Mudigere
    Last edited 8/29/18.

    Inputs:
        optimizer (Optimizer): the PBQN optimizer
        X_Sk (nparray): set of training examples over sample Sk
        y_Sk (nparray): set of training labels over sample Sk
        opfun (callable): computes forward pass over network over sample Sk
        ghost_batch (int): maximum size of effective batch (default: 128)

    Outputs:
        grad (tensor): stochastic gradient over sample Sk
        obj (tensor): stochastic function value over sample Sk

    """

    if (torch.cuda.is_available()):
        obj = torch.tensor(0, dtype=torch.float).cuda()
    else:
        obj = torch.tensor(0, dtype=torch.float)

    Sk_size = X_Sk.shape[0]

    optimizer.zero_grad()

    # loop through relevant data
    for idx in np.array_split(np.arange(Sk_size), max(int(Sk_size / ghost_batch), 1)):

        # define ops
        ops = opfun(X_Sk[idx])

        # define targets
        if (torch.cuda.is_available()):
            tgts = Variable(torch.from_numpy(y_Sk[idx]).cuda().long().squeeze())
        else:
            tgts = Variable(torch.from_numpy(y_Sk[idx]).long().squeeze())

        # define loss and perform forward-backward pass
        loss_fn = F.cross_entropy(ops, tgts) * (len(idx) / Sk_size)
        loss_fn.backward()

        # accumulate loss
        obj += loss_fn

    # gather flat gradient
    grad = optimizer._gather_flat_grad()

    return grad, obj


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class TorchGmlvq(torch.nn.Module, BaseEstimator, ClassifierMixin):

    def __init__(self,
                 alg,
                 nclasses,
                 input_dim,
                 prototypes_per_class=1,
                 prototype_initializer="stratified_mean",
                 # dtype=torch.float32,
                 dtype=torch.float64,
                 # distance_fn=pt.functions.distances.euclidean_distance,
                 distance_fn=squared_euclidean_distance,
                 beta=2,
                 max_iter=2500,
                 **kwargs):
        super().__init__()
        self.alg: str = alg
        self.nclasses = nclasses
        self.prototypes_per_class = prototypes_per_class
        self.prototype_initializer = prototype_initializer
        self.dtype = dtype
        self.input_dim = input_dim
        self.distance_fn = distance_fn
        self.max_iter = max_iter

        self.omega = torch.nn.Linear(input_dim, input_dim, bias=False)
        self.omega.weight.data.copy_(torch.eye(input_dim, dtype=dtype))
        if dtype == torch.float64:
            self.omega = self.omega.double()
        self.omega = self.omega.to(device)
        self.p1 = None  # define when dataset is known in fit for mean calculation
        self.criterion = GLVQLoss(squashing='sigmoid_beta', beta=beta).to(device)
        self.elapsed_xs = []
        self.elapsed_accuracy_ys = []
        self.n_iter_ = -1

    def forward(self, x):
        protos = self.p1.prototypes
        plabels = self.p1.prototype_labels
        ox = self.omega(x)
        op = self.omega(protos)
        dis = self.distance_fn(ox, op)
        return dis, plabels

    def fit(self, X, y):

        self.p1 = Prototypes1D(input_dim=self.input_dim,
                               prototypes_per_class=self.prototypes_per_class,
                               nclasses=self.nclasses,
                               prototype_initializer=self.prototype_initializer,
                               data=(X, y),
                               dtype=self.dtype)

        if self.dtype == torch.float64:
            self.p1 = self.p1.double()

        self.p1 = self.p1.to(device)
        self.p1.prototype_labels = self.p1.prototype_labels.to(device)
        self.p1.prototypes = self.p1.prototypes.to(device)

        with torch.no_grad():
            self.p1.prototypes += 0.1 * (torch.rand(self.nclasses * self.prototypes_per_class, self.input_dim,
                                                    dtype=self.dtype) * 2 - 1).to(device)

        self.train()

        self.omega.requires_grad_(False)
        self.p1.requires_grad_(True)
        y_tensor = torch.tensor(y, dtype=self.dtype, device=device)
        X_tensor = torch.tensor(X, dtype=self.dtype, device=device)
        self._fit(X_tensor, y_tensor)

        self.omega.requires_grad_(True)
        self.p1.requires_grad_(False)
        self._fit(X_tensor, y_tensor)

        self.omega.requires_grad_(True)
        self.p1.requires_grad_(True)
        self._fit(X_tensor, y_tensor)

        with torch.no_grad():
            self.omega.weight /= torch.sqrt(torch.sum(torch.diag(self.omega.weight)))

    def _fit(self, X, y):
        if self.alg == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
            b = time.time()
            for epoch in range(self.max_iter):
                # with torch.no_grad():
                #     self.omega.weight /= torch.sqrt(torch.sum(torch.diag(self.omega.weight)))
                loss = self.criterion(self.forward(X), y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                e = time.time()
                self.elapsed_xs.append(time.time() - b)
                # self.plot_xs.append(epoch)
                # self.elapsed_accuracy_ys.append(accuracy_score(y, self.predict(X)))
                # self.plot_ys.append(e - b)
                self.n_iter_ = epoch + 1
        elif self.alg == "pytorch_native":

            def closure():
                distances, plabels = self.forward(X)
                loss = self.criterion([distances, plabels], y)
                optimizer.zero_grad()
                if loss.requires_grad:
                    loss.backward()
                return loss

            optimizer = torch.optim.LBFGS(self.parameters(), lr=0.005, history_size=HISTORY_SIZE)
            last_obj = float("inf")
            b = time.time()
            for epoch in range(int(self.max_iter / 20)):
                # with torch.no_grad():
                #     self.omega.weight /= torch.sqrt(torch.sum(torch.diag(self.omega.weight)))
                # print(f'Epoch: {epoch + 1:03d}')
                obj = optimizer.step(closure=closure)
                e = time.time()
                self.n_iter_ = epoch + 1
                # if abs(last_obj - obj.item()) < TolFun:
                #     return
                # self.plot_xs.append(time)
                # self.elapsed_xs.append(time.time() - b)
                # self.elapsed_accuracy_ys.append(accuracy_score(y, self.predict(X)))
                # self.plot_ys.append(e - b)
        elif self.alg.startswith("pytorch_native_all_at_once"):

            def closure():
                optimizer.zero_grad()
                loss = self.criterion(self.forward(X), y)
                if loss.requires_grad:
                    loss.backward()
                return loss

            # 15000 max_eval like scipy
            # ftol=2.2204460492503131e-09,
            # gtol=1e-5
            # optimizer = torch.optim.LBFGS(self.parameters(), lr=1., history_size=HISTORY_SIZE, max_iter=self.max_iter, max_eval=15000)
            optimizer = torch.optim.LBFGS(self.parameters(),
                                          history_size=HISTORY_SIZE,
                                          tolerance_grad=1e-5, tolerance_change=2.2204460492503131e-09,
                                          max_iter=self.max_iter,
                                          line_search_fn="strong_wolfe", lr=1.,
                                          # line_search_fn=None, lr=0.005,
                                          )
            last_obj = float("inf")
            b = time.time()
            obj = optimizer.step(closure=closure)
            e = time.time()
            self.n_iter_ = max(self.n_iter_,
                               list(optimizer.state.items())[0][1]["n_iter"])  # TODO das geht bestimmt besser ...
            # self.elapsed_xs.append(time.time() - b)
            # self.elapsed_accuracy_ys.append(accuracy_score(y, self.predict(X)))
            # self.plot_ys.append(e - b)
        elif self.alg == "pytorch_scipy_wrapper":

            def closure():
                # with torch.no_grad():
                #     self.omega.weight /= torch.sqrt(torch.sum(torch.diag(self.omega.weight)))
                distances, plabels = self.forward(X)
                loss = self.criterion([distances, plabels], y)
                optimizer.zero_grad()
                if loss.requires_grad:
                    loss.backward()
                return loss

            # 15000 max_eval like scipy
            # ftol=2.2204460492503131e-09,
            # gtol=1e-5
            # optimizer = torch.optim.LBFGS(self.parameters(), lr=1., history_size=HISTORY_SIZE, max_iter=self.max_iter, max_eval=15000)
            optimizer = LBFGSScipy(self.parameters(),
                                   history_size=HISTORY_SIZE,
                                   tolerance_grad=1e-5, tolerance_change=2.2204460492503131e-09,
                                   max_iter=self.max_iter)
            last_obj = float("inf")
            b = time.time()
            obj = optimizer.step(closure=closure)
            e = time.time()
            self.n_iter_ = max(self.n_iter_, optimizer._n_iter)
            # self.elapsed_xs.append(time.time() - b)
            # self.elapsed_accuracy_ys.append(accuracy_score(y, self.predict(X)))
            # self.plot_ys.append(e - b)
        elif self.alg == "pytorch_external":

            def closure():
                optimizer.zero_grad()
                distances, plabels = self.forward(X)
                loss = self.criterion([distances, plabels], y)
                return loss

            optimizer = FullBatchLBFGS(self.parameters(), lr=1., history_size=HISTORY_SIZE, debug=False)
            optimizer.zero_grad()
            b = time.time()
            obj = self.criterion(self.forward(X), y)
            obj.backward()

            n_bad_steps = 0
            best_loss = obj.item()
            for epoch in range(self.max_iter):
                # with torch.no_grad():
                #     self.omega.weight /= torch.sqrt(torch.sum(torch.diag(self.omega.weight)))
                self.n_iter_ = epoch + 1
                # "max_ls": 5
                # reusing calculated obj reduces runtime!
                step_res = optimizer.step(
                        {'closure': closure, 'current_loss': obj, "ls_debug": False, "inplace": INPLACE,
                         "max_ls": MAX_LS})
                F_new, g_new, t, ls_step, closure_eval, grad_eval, desc_dir, fail = step_res
                # print(fail) TODO was macht man damit?
                obj, grad = step_res[0], step_res[1]
                if obj.item() < best_loss - 0.1:
                    best_loss = obj.item()
                    n_bad_steps = 0
                else:
                    n_bad_steps += 1
                if n_bad_steps > PATIENCE:
                    break

                e = time.time()
                # if abs(last_obj - obj.item()) < TolFun:
                #     return
                # self.elapsed_xs.append(time.time() - b)
                # self.elapsed_accuracy_ys.append(accuracy_score(y, self.predict(X)))
                # self.plot_xs.append(epoch)
                # self.plot_ys.append(e - b)
                last_obj = obj.item()
        elif self.alg == "pytorch_external_multibatch":

            # opfun = lambda X: self.forward(torch.from_numpy(X))
            opfun = lambda X: self.forward(X)[0]
            # predsfun = lambda op: np.argmax(op.data.numpy(), 1)
            # accfun = lambda op, y: np.mean(np.equal(predsfun(op), y.squeeze())) * 100  # Do the forward pass, then compute the accuracy

            optimizer = LBFGS(self.parameters(), history_size=HISTORY_SIZE, line_search='None',
                              debug=False)  # TODO Wolfe ausprobieren
            # optimizer = LBFGS(self.parameters(), history_size=HISTORY_SIZE, line_search='Wolfe')

            # Main training loop
            Ok_size = int(overlap_ratio * batch_size)
            Nk_size = int((1 - 2 * overlap_ratio) * batch_size)

            # sample previous overlap gradient
            random_index = np.random.permutation(range(X.shape[0]))
            Ok_prev = random_index[0:Ok_size]
            g_Ok_prev, obj_Ok_prev = get_grad(optimizer, X[Ok_prev], y[Ok_prev], opfun)

            # main loop
            for n_iter in range(self.max_iter):
                # with torch.no_grad():
                #     self.omega.weight /= torch.sqrt(torch.sum(torch.diag(self.omega.weight)))

                # sample current non-overlap and next overlap gradient
                random_index = np.random.permutation(range(X.shape[0]))
                Ok = random_index[0:Ok_size]
                Nk = random_index[Ok_size:(Ok_size + Nk_size)]

                g_Ok, obj_Ok = get_grad(optimizer, X[Ok], y[Ok], opfun)  # compute overlap gradient and objective
                g_Nk, obj_Nk = get_grad(optimizer, X[Nk], y[Nk], opfun)  # compute non-overlap gradient and objective
                g_Sk = overlap_ratio * (g_Ok_prev + g_Ok) + (
                        1 - 2 * overlap_ratio) * g_Nk  # compute accumulated gradient over sample
                p = optimizer.two_loop_recursion(-g_Sk)  # two-loop recursion to compute search direction
                lr = optimizer.step(p, g_Ok, g_Sk=g_Sk, options={"inplace": INPLACE})  # perform line search step
                Ok_prev = Ok  # compute previous overlap gradient for next sample
                g_Ok_prev, obj_Ok_prev = get_grad(optimizer, X[Ok_prev], y[Ok_prev], opfun)

                # curvature update
                optimizer.curvature_update(g_Ok_prev, eps=0.2, damping=True)

                # compute statistics
                # self.eval()
                # train_loss, test_loss, test_acc = compute_stats(X_train, y_train, X_test, y_test, opfun, accfun, ghost_batch=128)
                # print('Iter:', n_iter + 1, 'lr:', lr, 'Training Loss:', train_loss, 'Test Loss:', test_loss, 'Test Accuracy:', test_acc)
                self.n_iter_ = n_iter + 1
        elif self.alg == "pytorch_external_fullovlp":
            torch.manual_seed(2018)

            opfun = lambda X: self.forward(X)[0]
            optimizer = LBFGS(model.parameters(), lr=1., history_size=10, line_search='Wolfe', debug=True)

            for n_iter in range(self.max_iter):
                model.train()
                random_index = np.random.permutation(range(X.shape[0]))
                Sk = random_index[0:batch_size]

                # compute initial gradient and objective
                grad, obj = get_grad(optimizer, X[Sk], y_train[Sk], opfun)

                # two-loop recursion to compute search direction
                p = optimizer.two_loop_recursion(-grad)

                def closure():
                    optimizer.zero_grad()
                    distances, plabels = self.forward(X)
                    loss = self.criterion([distances, plabels], torch.tensor(y))
                    return loss

                # perform line search step
                options = {'closure': closure, 'current_loss': obj, "inplace": INPLACE}
                obj, grad, lr, _, _, _, _, _ = optimizer.step(p, grad, options=options)
                optimizer.curvature_update(grad)
                self.n_iter_ = n_iter

    def predict(self, X):
        # return self(torch.tensor(X, dtype=self.dtype))[0].argmin(axis=1).detach().numpy()
        if type(X) is np.ndarray:
            X = torch.tensor(X, device=device)
        return self(X.type(self.dtype))[0].argmin(axis=1).detach().cpu().numpy()
