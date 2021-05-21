# -*- coding: utf-8 -*-

# Based on work of
# Author: Joris Jensen <jjensen@techfak.uni-bielefeld.de>
# see: https://github.com/MrNuggelz/sklearn-lvq

# License: BSD 3 clause

from __future__ import division

import numpy as np
import torch
from prototorch.functions.distances import squared_euclidean_distance
from prototorch.modules.losses import GLVQLoss
from prototorch.modules.prototypes import Prototypes1D
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import validation
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted


class FastGmlvq(torch.nn.Module, BaseEstimator, ClassifierMixin):
    """
    Generalized Matrix Learning Vector Quantization

    Parameters
    ----------

    prototypes_per_class : int or list of int, optional (default=1)
        Number of prototypes per class. Use list to specify different numbers
        per class.

    initial_prototypes : array-like, (TODO)
     shape =  [n_prototypes, n_features + 1], optional
        Prototypes to start with. If not given initialization near the class
        means. Class label must be placed as last entry of each prototype

    initial_matrix : array-like, shape = [dim, n_features], optional (TODO)
        Relevance matrix to start with.
        If not given random initialization for rectangular matrix and unity
        for squared matrix.

    regularization : float, optional (default=0.0) (TODO)
        Value between 0 and 1. Regularization is done by the log determinant
        of the relevance matrix. Without regularization relevances may
        degenerate to zero.

    initialdim : int, optional (default=nb_features) (TODO)
        Maximum rank or projection dimensions

    max_iter : int, optional (default=2500)
        The maximum number of iterations.

    gtol : float, optional (default=1e-5)
        Gradient norm must be less than gtol before successful
        termination of l-bfgs-b.

    beta : int, optional (default=2)
        Used inside phi.
        1 / (1 + np.math.exp(-beta * x))

    c : array-like, shape = [2,3] ,optional
        Weights for wrong classification of form (y_real,y_pred,weight)
        Per default all weights are one, meaning you only need to specify
        the weights not equal one.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Attributes
    ----------

    w_ : array-like, shape = [n_prototypes, n_features]
        Prototype vector, where n_prototypes in the number of prototypes and
        n_features is the number of features

    c_w_ : array-like, shape = [n_prototypes]
        Prototype classes

    classes_ : array-like, shape = [n_classes]
        Array containing labels.

    omega_ : array-like, shape = [dim, n_features]
        Relevance matrix

    """

    def __init__(self,
                 prototypes_per_class=1,
                 initial_prototypes=None,
                 initial_matrix=None,
                 regularization=0.0,
                 initialdim=None,
                 max_iter=2500,
                 gtol=1e-5,
                 beta=2,
                 c=None,
                 random_state=None,
                 dtype=torch.float32,
                 device=torch.device("cpu")):

        super().__init__()
        self.prototypes_per_class = prototypes_per_class
        self.gtol = gtol
        self.inital_prototypes = initial_prototypes
        self.dtype = dtype
        self.max_iter = max_iter
        self.device = device
        self.n_iter_ = -1
        self.random_state = random_state
        self.beta = beta
        self.c = c
        self.regularization = regularization
        self.initial_matrix = initial_matrix
        if c is not None \
                or regularization > 0 \
                or initialdim is not None \
                or initial_prototypes is not None \
                or initial_matrix is not None:
            raise NotImplementedError()

        self.p = None  # torch prototype array
        self.omega = None  # torch omega
        self.criterion = None  # torch criterion (e.g. sigmoid_beta squashing)

    def forward(self, x):
        protos = self.p.prototypes
        ox = self.omega(x)
        op = self.omega(protos)
        return squared_euclidean_distance(ox, op), self.p.prototype_labels

    def fit(self, x, y):
        """Fit the LVQ model to the given training data and parameters using
        l-bfgs-b.

        Parameters
        ----------
        x : array-like, shape = [n_samples, n_features]
          Training vector, where n_samples in the number of samples and
          n_features is the number of features.
        y : array, shape = [n_samples]
          Target values (integers in classification, real numbers in
          regression)

        Returns
        --------
        self
        """
        x, y, random_state = self._validate_train_params(x, y)
        if len(np.unique(y)) == 1:
            raise ValueError("fitting " + type(
                    self).__name__ + " with only one class is not possible")

        input_dim = x.shape[1]
        nb_features = input_dim
        nclasses = len(unique_labels(y))

        self.omega = torch.nn.Linear(input_dim, input_dim, bias=False)
        self.omega.weight.data.copy_(torch.eye(input_dim, dtype=self.dtype))

        if self.initial_matrix is None:
            # if self.dim_ == nb_features: # TODO
            self.omega.weight.data.copy_(
                    torch.eye(input_dim, dtype=self.dtype))
            # else:
            #     self.omega.weight.data.copy_(
            #             random_state.rand(self.dim_, nb_features) * 2 - 1)
        else:
            self.omega.weight.data.copy_(
                    validation.check_array(self.initial_matrix))
            if self.omega_.shape[1] != nb_features:  # TODO: check dim
                raise ValueError("initial matrix has wrong number of features\n"
                                 "found=%d\nexpected=%d" % (
                                         self.omega_.shape[1], nb_features))

        self.p = Prototypes1D(input_dim=input_dim,
                              prototypes_per_class=self.prototypes_per_class,
                              nclasses=nclasses,
                              prototype_initializer="stratified_mean",
                              data=(x, y),
                              dtype=self.dtype)

        self.p = self.p.to(self.device)
        self.p.prototype_labels = self.p.prototype_labels.to(self.device)
        self.p.prototypes = self.p.prototypes.to(self.device)
        self.criterion = GLVQLoss(squashing='sigmoid_beta', beta=self.beta)

        with torch.no_grad():
            rng = random_state.rand(*self.p.prototypes.shape) - 0.5
            self.p.prototypes += 0.0001 * torch.tensor(rng,
                                                       dtype=self.dtype,
                                                       device=self.device)

        if self.dtype == torch.float64 or self.dtype == np.float64:
            self.p = self.p.double()
            self.omega = self.omega.double()

        self.omega = self.omega.to(self.device)
        self.criterion = self.criterion.to(self.device)

        self._optimize(x, y)

        return self

    def _validate_train_params(self, train_set, train_lab):
        random_state = validation.check_random_state(self.random_state)
        if not isinstance(self.max_iter, int) or self.max_iter < 1:
            raise ValueError("max_iter must be an positive integer")
        if not isinstance(self.gtol, float) or self.gtol <= 0:
            raise ValueError("gtol must be a positive float")
        train_set, train_lab = validation.check_X_y(train_set, train_lab)
        return train_set, train_lab, random_state

    def _optimize(self, X, y):

        if not isinstance(self.regularization,
                          float) or self.regularization < 0:
            raise ValueError("regularization must be a positive float ")

        self.train()

        y_tensor = torch.tensor(y, dtype=self.dtype, device=self.device)
        x_tensor = torch.tensor(X, dtype=self.dtype, device=self.device)

        for (o, p) in [(False, True), (True, False), (True, True)]:
            self.omega.requires_grad_(o)
            self.p.requires_grad_(p)

            def closure():
                optimizer.zero_grad()
                loss = self.criterion(self.forward(x_tensor), y_tensor)
                if loss.requires_grad:
                    loss.backward()
                return loss

            optimizer = torch.optim.LBFGS(self.parameters(),
                                          history_size=10,
                                          tolerance_grad=self.gtol,
                                          tolerance_change=2.22044604925031e-09,
                                          max_iter=self.max_iter,
                                          line_search_fn="strong_wolfe", lr=1.)
            optimizer.step(closure=closure)
            num_iter = list(optimizer.state.items())[0][1]["n_iter"]
            self.n_iter_ = max(self.n_iter_, num_iter)

        with torch.no_grad():
            normalization = torch.sqrt(torch.sum(torch.diag(self.omega.weight)))
            self.omega.weight /= normalization

    def predict(self, X):
        if type(X) is np.ndarray:
            X = torch.tensor(X, device=self.device, dtype=self.dtype)
        else:
            X = X.type(self.dtype)
        proto_idx = self(X)[0].argmin(axis=1)
        return self.p.prototype_labels[proto_idx].detach().cpu().numpy()

    @property
    def classes_(self):
        return unique_labels(self.c_w_)

    @property
    def omega_(self):
        return self.omega.weight.detach().numpy()

    @property
    def c_w_(self):
        return self.p.prototype_labels.detach().numpy()

    @property
    def w_(self):
        return self.p.prototypes.detach().numpy()

    def decision_function(self, x):
        """Predict confidence scores for samples.

        Parameters
        ----------
        x : array-like, shape = [n_samples, n_features]


        Returns
        -------
        T : array-like, shape=(n_samples,) if n_classes == 2
                                           else (n_samples, n_classes)
        """
        check_is_fitted(self, ['w_', 'c_w_'])

        x = validation.check_array(x)
        if x.shape[1] != self.w_.shape[1]:
            raise ValueError("X has wrong number of features\n"
                             "found=%d\n"
                             "expected=%d" % (self.w_.shape[1], x.shape[1]))
        dist = self.forward(torch.tensor(x, dtype=self.dtype))[
            0].detach().numpy()

        foo = lambda cls: dist[:, self.c_w_ != cls].min(1)\
                        - dist[:, self.c_w_ == cls].min(1)
        res = np.vectorize(foo, signature='()->(n)')(self.classes_).T

        if self.classes_.size <= 2:
            return res[:, 1]
        else:
            return res


# Example:
# X, y = load_digits(return_X_y=True)
#
# begin = time.time()
# model = FastGmlvq()
# model.fit(X, y)
# end = time.time()
# print(f"CPU {end - begin}")
#
# begin = time.time()
# model = FastGmlvq(device=torch.device("cuda:0"))
# model.fit(X, y)
# end = time.time()
# print(f"GPU {end - begin}")
