import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.datasets import load_digits, load_wine, load_iris, load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn_lvq import GmlvqModel, GlvqModel

from LibGMLVQ_other_solver import LibGMLVQ_other_solver
from MyGMLVQ import MyGMLVQ
from datasets import load_uci_segmentation, blobs_dataset
from prototorch_tests.TorchGmlvq import TorchGmlvq

CUDA_AVAILABLE = False

if CUDA_AVAILABLE:
    from cuda.numba.BFGS_GMLVQ import MyBFGSCudaGMLVQ
    from cuda.numba.BFGS_GMLVQ_32 import MyBFGSCudaGMLVQ32
    from cuda.numba.BFGS_GMLVQ_shared_memory import MyBFGSCudaGMLVQshared
    from cuda.pycuda.BFGS_GMLVQ_pycuda import MyBFGSCudaGMLVQ_PyCuda

all_datasets = [
        # blobs_dataset(30, prototypes=2, features=2),
        # blobs_dataset(30, prototypes=2, features=2),
        blobs_dataset(500, prototypes=20, features=10),
        # blobs_dataset(500, prototypes=20, features=10),
        # blobs_dataset(500, prototypes=50, features=10),
        # blobs_dataset(5000, prototypes=10, features=20),
        # blobs_dataset(10_000, prototypes=2, features=20),
        # blobs_dataset(10_000, prototypes=10, features=20),
        ("wine", load_wine(return_X_y=True)),
        ("breast cancer", load_breast_cancer(return_X_y=True)),
        ("iris", load_iris(return_X_y=True)),
        load_uci_segmentation(),
        ("digits", load_digits(return_X_y=True)),
]

max_iter = 2_000


def test_performance(models=None, datasets=None, repetitions=1, plot=False, no_accuracy=False):
    if datasets is None:
        datasets = all_datasets

    for name, (X, y) in datasets:
        input_dim = X.shape[-1]
        nclasses = len(set(y))
        MAX_ITER = max_iter
        beta = 10

        all_models = [
                # LibGMLVQ_other_solver(max_iter=max_iter, alg="nlopt"), # nlopt not working as intended

                LibGMLVQ_other_solver(max_iter=max_iter, alg="matlab", matlab_options={"GradConstr": True, "HessUpdate": "lbfgs", "GoalsExactAchieve": 1, "MaxIter": max_iter, "rho": 0.0001}),
                LibGMLVQ_other_solver(max_iter=max_iter, alg="matlab", matlab_options={"GradConstr": True, "HessUpdate": "lbfgs", "GoalsExactAchieve": 1, "MaxIter": max_iter, "rho": 0.0001}),

                GmlvqModel(max_iter=max_iter),
                GmlvqModel(max_iter=max_iter, beta=10),
                GmlvqModel(max_iter=max_iter, beta=50),
                GmlvqModel(max_iter=max_iter, beta=beta),

                LibGMLVQ_other_solver(max_iter=max_iter, alg="scipy", beta=beta),

                TorchGmlvq("adam", nclasses, input_dim, max_iter=MAX_ITER, beta=beta),
                # TorchGmlvq("pytorch_external", nclasses, input_dim, max_iter=MAX_ITER, beta=beta),
                TorchGmlvq("pytorch_native", nclasses, input_dim, max_iter=MAX_ITER, beta=beta),
                TorchGmlvq("pytorch_native", nclasses, input_dim, max_iter=MAX_ITER, beta=beta, dtype=torch.float32),
                TorchGmlvq("pytorch_native_all_at_once", nclasses, input_dim, max_iter=MAX_ITER, beta=beta),
                TorchGmlvq("pytorch_native_all_at_once", nclasses, input_dim, max_iter=MAX_ITER, beta=beta, dtype=torch.float32),

                LibGMLVQ_other_solver(max_iter=max_iter, alg="matlab", matlab_options={"GradConstr": False, "HessUpdate": "lbfgs", "GoalsExactAchieve": 1, "MaxIter": max_iter, "rho": 0.0100}, nb_reiterations=10),
                LibGMLVQ_other_solver(max_iter=max_iter, alg="matlab", matlab_options={"GradConstr": True, "HessUpdate": "lbfgs", "GoalsExactAchieve": 1, "MaxIter": max_iter, "rho": 0.0100}, nb_reiterations=10),
                LibGMLVQ_other_solver(max_iter=max_iter, alg="matlab", matlab_options={"GradConstr": False, "HessUpdate": "lbfgs", "GoalsExactAchieve": 1, "MaxIter": max_iter, "rho": 0.0001}, nb_reiterations=10),

                LibGMLVQ_other_solver(max_iter=max_iter, alg="matlab"),
                LibGMLVQ_other_solver(max_iter=max_iter, alg="matlab", matlab_options={"GradConstr": False, "HessUpdate": "bfgs", "GoalsExactAchieve": 0, "MaxIter": max_iter, "StoreN": 20}),
                LibGMLVQ_other_solver(max_iter=max_iter, alg="matlab", matlab_options={"GradConstr": False, "HessUpdate": "bfgs", "GoalsExactAchieve": 1, "MaxIter": max_iter, "StoreN": 20}),
                LibGMLVQ_other_solver(max_iter=max_iter, alg="matlab", matlab_options={"GradConstr": False, "HessUpdate": "lbfgs", "GoalsExactAchieve": 0, "MaxIter": max_iter, "StoreN": 20}),
                LibGMLVQ_other_solver(max_iter=max_iter, alg="matlab", matlab_options={"GradConstr": False, "HessUpdate": "lbfgs", "GoalsExactAchieve": 1, "MaxIter": max_iter, "StoreN": 20}),
                LibGMLVQ_other_solver(max_iter=max_iter, alg="matlab", matlab_options={"GradConstr": True, "HessUpdate": "bfgs", "GoalsExactAchieve": 0, "MaxIter": max_iter, "StoreN": 20}),
                LibGMLVQ_other_solver(max_iter=max_iter, alg="matlab", matlab_options={"GradConstr": True, "HessUpdate": "bfgs", "GoalsExactAchieve": 1, "MaxIter": max_iter, "StoreN": 20}),
                LibGMLVQ_other_solver(max_iter=max_iter, alg="matlab", matlab_options={"GradConstr": True, "HessUpdate": "lbfgs", "GoalsExactAchieve": 0, "MaxIter": max_iter, "StoreN": 20}),
                LibGMLVQ_other_solver(max_iter=max_iter, alg="matlab", matlab_options={"GradConstr": True, "HessUpdate": "lbfgs", "GoalsExactAchieve": 1, "MaxIter": max_iter, "StoreN": 20}),

                #LibGMLVQ_other_solver(max_iter=max_iter, alg="pylbfgs"), # bad
                #LibGMLVQ_other_solver(max_iter=max_iter, alg="scipy-nojac"), # bad

        ]

        if CUDA_AVAILABLE:
            all_models += [
                    GlvqModel(max_iter=max_iter),
                    MyBFGSCudaGMLVQ(iterations=max_iter),
                    MyBFGSCudaGMLVQ(iterations=max_iter),
                    MyBFGSCudaGMLVQ_PyCuda(iterations=max_iter),
                    MyBFGSCudaGMLVQ32(iterations=max_iter),

                    MyBFGSCudaGMLVQshared(iterations=max_iter),
                    MyGMLVQ(1, 0.1, 0.1),
                    MyGMLVQ(max_iter, 0.1, 0.1),
                    MyGMLVQ(max_iter, 0.01, 0.01),
                    MyGMLVQ(max_iter, 0.001, 0.001),
            ]

        # if models is None:
        #     models = all_models
        models = all_models

        print(name)
        for model in models:
            duration, accuracy, num_iters = 0.0, 0.0, 0
            for _ in range(repetitions):
                begin = time.time()
                model.fit(X, y)
                end = time.time()
                duration += end - begin
                num_iters += model.n_iter_
                if no_accuracy:
                    accuracy += accuracy_score(y, model.predict(X))
            duration = duration / repetitions
            accuracy = accuracy / repetitions
            num_iters = num_iters / repetitions

            try:
                model_name = model.alg
            except:
                model_name = str(model).replace("\n", "").replace("\t", " ").replace("    ", " ").replace("    ", " ").replace("  ", " ").replace("  ", " ").replace("  ", " ")
            duration_per_iter = duration / num_iters if num_iters != 0 else -1
            title = f"\t{model_name:<40} Duration={duration:.5f} Accuracy={accuracy * 100:.3f} iters={num_iters:.2f} duration/iters={(duration_per_iter):.2f}"
            print(title)
            try:
                omega = model.omega
            except:
                omega = model.omega_
            # print(omega)

            if plot:
                plot_result(X, model, omega, title, y)

        print("done")


def plot_result(X, model, omega, title, y):
    def project(omega, x, dims, print_variance_covered=False):  # adapted from sklearn-lvq GmlvqModel
        v, u = np.linalg.eig(omega.conj().T.dot(omega))
        idx = v.argsort()[::-1]
        if print_variance_covered:
            print('variance coverd by projection:', v[idx][:dims].sum() / v.sum() * 100)
        return x.dot(u[:, idx][:, :dims].dot(np.diag(np.sqrt(v[idx][:dims]))))

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.scatter(model.w_[:, 0], model.w_[:, 1], s=2000, marker='*', linewidth=3, edgecolor='r', c=model.classes_)
    plt.axis('equal')
    plt.subplot(1, 2, 2)
    p = project(omega, X, 2, print_variance_covered=False)
    plt.scatter(p[:, 0], p[:, 1], c=y)
    plt.axis('equal')
    plt.suptitle(title)
    try:
        for p in range(model.history.shape[1]):
            plt.plot(model.history[:, p, 0], model.history[:, p, 1])
    except:
        pass
    plt.show()


test_performance(repetitions=2, no_accuracy=True, plot=False)

"""
/Users/christoph/.pyenv/shims/python3 /Users/christoph/PycharmProjects/LVQ_experiments/performance_tester.py
blobs: 30 samples, 2 pts, 2 features
	GmlvqModel(beta=10, max_iter=10000)      Duration=0.00328 Accuracy=100.000 iters=0.00 duration/iters=-1.00
	MyGMLVQ(iterations=1, lr=0.1, matrix_lr=0.1) Duration=6.34504 Accuracy=46.667 iters=-1.00 duration/iters=-6.35
	MyGMLVQ(iterations=10000, lr=0.1, matrix_lr=0.1) Duration=4.39643 Accuracy=96.667 iters=-1.00 duration/iters=-4.40
done
blobs: 500 samples, 20 pts, 10 features
	GmlvqModel(beta=10, max_iter=10000)      Duration=5.66608 Accuracy=100.000 iters=281.00 duration/iters=0.02
	MyGMLVQ(iterations=1, lr=0.1, matrix_lr=0.1) Duration=0.12004 Accuracy=51.000 iters=-1.00 duration/iters=-0.12
	MyGMLVQ(iterations=10000, lr=0.1, matrix_lr=0.1) Duration=1194.54251 Accuracy=100.000 iters=-1.00 duration/iters=-1194.54
done
wine
	GmlvqModel(beta=10, max_iter=10000)      Duration=0.43262 Accuracy=90.449 iters=89.00 duration/iters=0.00
	MyGMLVQ(iterations=1, lr=0.1, matrix_lr=0.1) Duration=0.06796 Accuracy=39.888 iters=-1.00 duration/iters=-0.07
	MyGMLVQ(iterations=10000, lr=0.1, matrix_lr=0.1) Duration=671.61255 Accuracy=39.888 iters=-1.00 duration/iters=-671.61
done
breast cancer
	GmlvqModel(beta=10, max_iter=10000)      Duration=37.34683 Accuracy=98.418 iters=10000.00 duration/iters=0.00
	MyGMLVQ(iterations=1, lr=0.1, matrix_lr=0.1) Duration=1.57574 Accuracy=62.742 iters=-1.00 duration/iters=-1.58
	MyGMLVQ(iterations=10000, lr=0.1, matrix_lr=0.1) Duration=15815.86970 Accuracy=62.742 iters=-1.00 duration/iters=-15815.87
done
iris
	GmlvqModel(beta=10, max_iter=10000)      Duration=0.20262 Accuracy=98.667 iters=64.00 duration/iters=0.00
	MyGMLVQ(iterations=1, lr=0.1, matrix_lr=0.1) Duration=0.00696 Accuracy=52.667 iters=-1.00 duration/iters=-0.01
	MyGMLVQ(iterations=10000, lr=0.1, matrix_lr=0.1) Duration=58.80236 Accuracy=88.667 iters=-1.00 duration/iters=-58.80
done
uci_segmentation
	GmlvqModel(beta=10, max_iter=10000)      Duration=61.45486 Accuracy=91.558 iters=3534.00 duration/iters=0.02
	MyGMLVQ(iterations=1, lr=0.1, matrix_lr=0.1) Duration=5.20351 Accuracy=62.944 iters=-1.00 duration/iters=-5.20
	MyGMLVQ(iterations=10000, lr=0.1, matrix_lr=0.1) Duration=19312.49080 Accuracy=58.961 iters=-1.00 duration/iters=-19312.49
done
digits
	GmlvqModel(beta=10, max_iter=10000)      Duration=121.00277 Accuracy=99.666 iters=2782.00 duration/iters=0.04
	MyGMLVQ(iterations=1, lr=0.1, matrix_lr=0.1) Duration=38.31875 Accuracy=89.594 iters=-1.00 duration/iters=-38.32
s
"""
