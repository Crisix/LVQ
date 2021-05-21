import time

import numpy as np
import torch
from numba.cuda import CudaSupportError
from sklearn.datasets import load_wine, load_iris, load_breast_cancer, load_digits
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn_lvq import GmlvqModel

from datasets import blobs_dataset, load_uci_segmentation
from prototorch_tests.TorchGmlvq import TorchGmlvq

if False:
    from torchvision.datasets import MNIST
    mnist_trainset = MNIST(root='./data', train=True, download=True, transform=None)

all_datasets = [
        blobs_dataset(100, prototypes=2, features=2),
        # blobs_dataset(2000, prototypes=2, features=2),
        #
        # blobs_dataset(100, prototypes=2, features=20),
        # blobs_dataset(2000, prototypes=2, features=20),
        #
        blobs_dataset(100, prototypes=20, features=20),
        # blobs_dataset(2000, prototypes=20, features=20),
        #
        # blobs_dataset(1000, prototypes=2, features=20),
        # blobs_dataset(1000, prototypes=10, features=20),
        # blobs_dataset(1000, prototypes=20, features=20),
        # blobs_dataset(1000, prototypes=50, features=20),
        # blobs_dataset(1000, prototypes=100, features=20),
        # blobs_dataset(1000, prototypes=200, features=20),
        # blobs_dataset(30, prototypes=2, features=2),
        # blobs_dataset(1000, prototypes=20, features=15),
        # blobs_dataset(1000, prototypes=20, features=15),
        # blobs_dataset(2000, prototypes=50, features=10),
        # blobs_dataset(10_000, prototypes=2, features=20),
        # blobs_dataset(100_000, prototypes=100, features=20),

        ("wine", load_wine(return_X_y=True)),
        ("iris", load_iris(return_X_y=True)),
        ("breast cancer", load_breast_cancer(return_X_y=True)),
        load_uci_segmentation(),
        ("digits", load_digits(return_X_y=True)),

        # ("mnist", (mnist_trainset.train_data.numpy().reshape((mnist_trainset.train_data.numpy().shape[0], -1)), mnist_trainset.train_labels.numpy())),
        # ("mnist_test", (mnist_trainset.test_data.numpy().reshape((mnist_trainset.test_data.numpy().shape[0], -1)), mnist_trainset.test_labels.numpy()))
]


# x_train, y_train = load_digits(return_X_y=True)
# name, (x_train, y_train) = load_uci_segmentation()

def color(alg):
    if alg == "pytorch_external":
        return "tab:orange"
    if alg == "pytorch_native":
        return "tab:green"
    if alg == "adam":
        return "tab:blue"


def getGmlvq(max_iter, beta):
    model = GmlvqModel(max_iter=max_iter, beta=beta)
    model.alg = "scipy gmlvq"
    return model


TEST_GRAD_PERFORMANCE = True

PATIENCE = 10000000
HISTORY_SIZE = 10
beta = 2
INPLACE = False
MAX_LS = 100  # DEFAULT 10
batch_size = -1
overlap_ratio = 0.25  # should be in (0, 0.5)

REPS = 4

for _ in range(1):
    for name, (x_train, y_train) in all_datasets:
        x_train = StandardScaler().fit_transform(x_train)
        batch_size = 2048  # int(x_train.shape[0] / 5)
        input_dim = x_train.shape[-1]
        nclasses = len(set(y_train))

        for MAX_ITER in [500]:
            print(f"\n{name} ({x_train.shape})  | {MAX_ITER} iterations")
            for model_gen in [
                    lambda: TorchGmlvq("adam", nclasses, input_dim, max_iter=MAX_ITER, beta=beta),
                    lambda: getGmlvq(MAX_ITER, beta=beta),
                    lambda: TorchGmlvq("pytorch_scipy_wrapper", nclasses, input_dim, max_iter=MAX_ITER, beta=beta),
                    lambda: TorchGmlvq("pytorch_native_all_at_once_32", nclasses, input_dim, max_iter=MAX_ITER, beta=beta, dtype=torch.float32),
                    lambda: TorchGmlvq("pytorch_native_all_at_once_64", nclasses, input_dim, max_iter=MAX_ITER, beta=beta, dtype=torch.float64),
                    # too slow or other drawbacks:
                    # lambda: NotThreeTimes(max_iter=MAX_ITER, beta=beta),
                    # BetaLoopGMLVQ(max_iter=MAX_ITER, beta=beta),
                    # SameFGGmlvqModel(max_iter=MAX_ITER, beta=beta),
                    # TorchGmlvq("pytorch_external_multibatch", nclasses, input_dim, max_iter=10 * MAX_ITER, beta=beta),
                    # TorchGmlvq("pytorch_external_fullovlp", nclasses, input_dim, max_iter=MAX_ITER, beta=beta),
                    # TorchGmlvq("pytorch_external", nclasses, input_dim, max_iter=MAX_ITER, beta=beta),
                    # TorchGmlvq("pytorch_native", nclasses, input_dim, max_iter=MAX_ITER, beta=beta),
            ]:
                # for dev, device in [("   ", torch.device("cpu")), ("GPU", torch.device('cuda:0'))]:
                for dev, device in [("   ", torch.device("cpu"))]:
                    try:
                        total_duration = 0
                        total_accuracy = 0
                        total_nits = 0
                        model = model_gen()
                        for rep in range(REPS):
                            model = model_gen()
                            bbbegin = time.time()
                            model.fit(x_train, y_train)
                            eeend = time.time()
                            total_accuracy += accuracy_score(y_train, model.predict(x_train))
                            total_duration += eeend - bbbegin
                            total_nits += model.n_iter_

                        total_duration /= REPS
                        total_accuracy /= REPS
                        total_nits /= REPS

                        bruteforce_time = -1
                        if TEST_GRAD_PERFORMANCE:
                            R = 2_000
                            if hasattr(model, 'p1'):  # -> is pytorch model
                                begin = time.time()
                                model.omega.requires_grad_(True)
                                model.p1.requires_grad_(True)
                                x_tensor = torch.tensor(x_train, dtype=model.dtype, device=device)
                                y_tensor = torch.tensor(y_train, dtype=model.dtype, device=device)
                                for _ in range(R):
                                    model.criterion(model.forward(x_tensor), y_tensor).backward()
                                    model.zero_grad()
                                end = time.time()
                            else:
                                begin = time.time()
                                variables = np.append(model.w_, model.omega_, axis=0)
                                label_equals_prototype = y_train[np.newaxis].T == model.c_w_
                                random_state = np.random.mtrand._rand
                                for _ in range(R):
                                    # model._optfun(variables, x_train, label_equals_prototype)
                                    model._optgrad(variables, x_train, label_equals_prototype, random_state, lr_relevances=1, lr_prototypes=1)
                                end = time.time()
                            bruteforce_time = end - begin

                        print(f"{dev} {model.alg:<30} "
                              f"\t Accuracy: {total_accuracy:.5f} "
                              f"\t Duration: {total_duration:.5f} "
                              f"\t Iterations={total_nits:>5} "
                              f"\t BF-Time={bruteforce_time}")
                    except CudaSupportError as e:
                        print(e)
                        pass
