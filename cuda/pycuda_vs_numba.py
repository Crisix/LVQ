SUMMARY = """

test_performance(repetitions=20) auf Tesla T4

wine
	GmlvqModel()                                                               Duration=1.03847 	Accuracy=89.270 iters=653.85 	duration/iters=0.001588
	MyBFGSCudaGMLVQ(gtol=1e-05, iterations=1000)                               Duration=0.45298 	Accuracy=67.978 iters=16.00 	duration/iters=0.028311
	MyBFGSCudaGMLVQ32(gtol=1e-05, iterations=1000)                             Duration=0.40552 	Accuracy=67.444 iters=29.05 	duration/iters=0.013959
	MyBFGSCudaGMLVQshared(gtol=1e-05, iterations=1000)                         Duration=0.43372 	Accuracy=67.978 iters=16.00 	duration/iters=0.027108
	MyBFGSCudaGMLVQ_PyCuda(gtol=1e-05, iterations=1000)                        Duration=0.09481 	Accuracy=66.882 iters=27.50 	duration/iters=0.003448
	MyBFGSCudaGMLVQ_PyCuda_nonshared(gtol=1e-05, iterations=1000)              Duration=0.10025 	Accuracy=66.854 iters=28.60 	duration/iters=0.003505


breast cancer
	GmlvqModel()                                                               Duration=4.63509 	Accuracy=96.318 iters=771.55 	duration/iters=0.006008
	MyBFGSCudaGMLVQ(gtol=1e-05, iterations=1000)                               Duration=0.77955 	Accuracy=89.455 iters=3.00 	    duration/iters=0.259849
	MyBFGSCudaGMLVQ32(gtol=1e-05, iterations=1000)                             Duration=0.67052 	Accuracy=89.464 iters=4.65 	    duration/iters=0.144197
	MyBFGSCudaGMLVQshared(gtol=1e-05, iterations=1000)                         Duration=0.79095 	Accuracy=89.455 iters=3.00 	    duration/iters=0.263651
	MyBFGSCudaGMLVQ_PyCuda(gtol=1e-05, iterations=1000)                        Duration=0.12479 	Accuracy=89.104 iters=3.70 	    duration/iters=0.033727
	MyBFGSCudaGMLVQ_PyCuda_nonshared(gtol=1e-05, iterations=1000)              Duration=0.09345 	Accuracy=89.104 iters=3.75 	    duration/iters=0.024921


iris
	GmlvqModel()       								                           Duration=0.02665 	Accuracy=98.667 iters=13.00 	duration/iters=0.002050
	MyBFGSCudaGMLVQ(gtol=1e-05, iterations=1000)                               Duration=0.29244 	Accuracy=96.667 iters=9.90 	    duration/iters=0.029539
	MyBFGSCudaGMLVQ32(gtol=1e-05, iterations=1000)                             Duration=0.29271 	Accuracy=96.667 iters=10.10 	duration/iters=0.028981
	MyBFGSCudaGMLVQshared(gtol=1e-05, iterations=1000)                         Duration=0.28849 	Accuracy=96.667 iters=9.95 	    duration/iters=0.028994
	MyBFGSCudaGMLVQ_PyCuda(gtol=1e-05, iterations=1000)                        Duration=0.02756 	Accuracy=91.333 iters=11.00 	duration/iters=0.002506
	MyBFGSCudaGMLVQ_PyCuda_nonshared(gtol=1e-05, iterations=1000)              Duration=0.02811 	Accuracy=91.333 iters=11.00 	duration/iters=0.002556


digits
	GmlvqModel()                                                               Duration=63.53739 	Accuracy=97.014 iters=1000.00 	duration/iters=0.063537
	MyBFGSCudaGMLVQ(gtol=1e-05, iterations=1000)                               Duration=4.35765 	Accuracy=92.321 iters=12.00 	duration/iters=0.363137
	MyBFGSCudaGMLVQ32(gtol=1e-05, iterations=1000)                             Duration=7.05150 	Accuracy=92.346 iters=13.65 	duration/iters=0.516593
	MyBFGSCudaGMLVQshared(gtol=1e-05, iterations=1000)                         Duration=4.05555 	Accuracy=92.321 iters=12.00 	duration/iters=0.337963
	MyBFGSCudaGMLVQ_PyCuda(gtol=1e-05, iterations=1000)                        Duration=2.07170 	Accuracy=90.707 iters=14.15 	duration/iters=0.146410
	MyBFGSCudaGMLVQ_PyCuda_nonshared(gtol=1e-05, iterations=1000)              Duration=1.98102 	Accuracy=90.707 iters=14.10 	duration/iters=0.140498



"""
RESULTS = """
test_performance(repetitions=20) auf Tesla T4

wine
	GmlvqModel(beta=2, c=None, display=False, gtol=1e-05, initial_matrix=None, initial_prototypes=None, initialdim=None, max_iter=1000, prototypes_per_class=1, random_state=None, regularization=0.0)       Duration=0.78968 	Accuracy=88.230 iters=605.05 	duration/iters=0.001305
	MyBFGSCudaGMLVQ(gtol=1e-05, iterations=1000)                                                                                                                                                             Duration=0.45298 	Accuracy=67.978 iters=16.00 	duration/iters=0.028311
	MyBFGSCudaGMLVQ32(gtol=1e-05, iterations=1000)                                                                                                                                                           Duration=0.40552 	Accuracy=67.444 iters=29.05 	duration/iters=0.013959
	MyBFGSCudaGMLVQshared(gtol=1e-05, iterations=1000)                                                                                                                                                       Duration=0.43372 	Accuracy=67.978 iters=16.00 	duration/iters=0.027108
wine
	GmlvqModel(beta=2, c=None, display=False, gtol=1e-05, initial_matrix=None, initial_prototypes=None, initialdim=None, max_iter=1000, prototypes_per_class=1, random_state=None, regularization=0.0)       Duration=1.03847 	Accuracy=89.270 iters=653.85 	duration/iters=0.001588
	MyBFGSCudaGMLVQ_PyCuda(gtol=1e-05, iterations=1000)                                                                                                                                                      Duration=0.09481 	Accuracy=66.882 iters=27.50 	duration/iters=0.003448
	MyBFGSCudaGMLVQ_PyCuda_nonshared(gtol=1e-05, iterations=1000)                                                                                                                                            Duration=0.10025 	Accuracy=66.854 iters=28.60 	duration/iters=0.003505


breast cancer
	GmlvqModel(beta=2, c=None, display=False, gtol=1e-05, initial_matrix=None, initial_prototypes=None, initialdim=None, max_iter=1000, prototypes_per_class=1, random_state=None, regularization=0.0)       Duration=4.63509 	Accuracy=96.318 iters=771.55 	duration/iters=0.006008
	MyBFGSCudaGMLVQ(gtol=1e-05, iterations=1000)                                                                                                                                                             Duration=0.77955 	Accuracy=89.455 iters=3.00 	duration/iters=0.259849
	MyBFGSCudaGMLVQ32(gtol=1e-05, iterations=1000)                                                                                                                                                           Duration=0.67052 	Accuracy=89.464 iters=4.65 	duration/iters=0.144197
	MyBFGSCudaGMLVQshared(gtol=1e-05, iterations=1000)                                                                                                                                                       Duration=0.79095 	Accuracy=89.455 iters=3.00 	duration/iters=0.263651
breast cancer
	GmlvqModel(beta=2, c=None, display=False, gtol=1e-05, initial_matrix=None, initial_prototypes=None, initialdim=None, max_iter=1000, prototypes_per_class=1, random_state=None, regularization=0.0)       Duration=4.44485 	Accuracy=96.248 iters=609.00 	duration/iters=0.007299
	MyBFGSCudaGMLVQ_PyCuda(gtol=1e-05, iterations=1000)                                                                                                                                                      Duration=0.12479 	Accuracy=89.104 iters=3.70 	duration/iters=0.033727
	MyBFGSCudaGMLVQ_PyCuda_nonshared(gtol=1e-05, iterations=1000)                                                                                                                                            Duration=0.09345 	Accuracy=89.104 iters=3.75 	duration/iters=0.024921


iris
	GmlvqModel(beta=2, c=None, display=False, gtol=1e-05, initial_matrix=None, initial_prototypes=None, initialdim=None, max_iter=1000, prototypes_per_class=1, random_state=None, regularization=0.0)       Duration=0.02665 	Accuracy=98.667 iters=13.00 	duration/iters=0.002050
	MyBFGSCudaGMLVQ(gtol=1e-05, iterations=1000)                                                                                                                                                             Duration=0.29244 	Accuracy=96.667 iters=9.90 	duration/iters=0.029539
	MyBFGSCudaGMLVQ32(gtol=1e-05, iterations=1000)                                                                                                                                                           Duration=0.29271 	Accuracy=96.667 iters=10.10 	duration/iters=0.028981
	MyBFGSCudaGMLVQshared(gtol=1e-05, iterations=1000)                                                                                                                                                       Duration=0.28849 	Accuracy=96.667 iters=9.95 	duration/iters=0.028994
iris
	GmlvqModel(beta=2, c=None, display=False, gtol=1e-05, initial_matrix=None, initial_prototypes=None, initialdim=None, max_iter=1000, prototypes_per_class=1, random_state=None, regularization=0.0)       Duration=0.02796 	Accuracy=98.667 iters=13.00 	duration/iters=0.002151
	MyBFGSCudaGMLVQ_PyCuda(gtol=1e-05, iterations=1000)                                                                                                                                                      Duration=0.02756 	Accuracy=91.333 iters=11.00 	duration/iters=0.002506
	MyBFGSCudaGMLVQ_PyCuda_nonshared(gtol=1e-05, iterations=1000)                                                                                                                                            Duration=0.02811 	Accuracy=91.333 iters=11.00 	duration/iters=0.002556




digits
	GmlvqModel(beta=2, c=None, display=False, gtol=1e-05, initial_matrix=None, initial_prototypes=None, initialdim=None, max_iter=1000, prototypes_per_class=1, random_state=None, regularization=0.0)       Duration=56.28711 	Accuracy=96.967 iters=976.45 	duration/iters=0.057645
	MyBFGSCudaGMLVQ(gtol=1e-05, iterations=1000)                                                                                                                                                             Duration=4.35765 	Accuracy=92.321 iters=12.00 	duration/iters=0.363137
	MyBFGSCudaGMLVQ32(gtol=1e-05, iterations=1000)                                                                                                                                                           Duration=7.05150 	Accuracy=92.346 iters=13.65 	duration/iters=0.516593
	MyBFGSCudaGMLVQshared(gtol=1e-05, iterations=1000)                                                                                                                                                       Duration=4.05555 	Accuracy=92.321 iters=12.00 	duration/iters=0.337963
digits
	GmlvqModel(beta=2, c=None, display=False, gtol=1e-05, initial_matrix=None, initial_prototypes=None, initialdim=None, max_iter=1000, prototypes_per_class=1, random_state=None, regularization=0.0)       Duration=63.53739 	Accuracy=97.014 iters=1000.00 	duration/iters=0.063537
	MyBFGSCudaGMLVQ_PyCuda(gtol=1e-05, iterations=1000)                                                                                                                                                      Duration=2.07170 	Accuracy=90.707 iters=14.15 	duration/iters=0.146410
	MyBFGSCudaGMLVQ_PyCuda_nonshared(gtol=1e-05, iterations=1000)                                                                                                                                            Duration=1.98102 	Accuracy=90.707 iters=14.10 	duration/iters=0.140498






KAPUTT, warum noch unklar
uci_segmentation
	GmlvqModel(beta=2, c=None, display=False, gtol=1e-05, initial_matrix=None, initial_prototypes=None, initialdim=None, max_iter=1000, prototypes_per_class=1, random_state=None, regularization=0.0)       Duration=21.60958 	Accuracy=91.251 iters=976.45 	duration/iters=0.022131
	MyBFGSCudaGMLVQ(gtol=1e-05, iterations=1000)                                                                                                                                                             Duration=0.52755 	Accuracy=72.294 iters=0.00 	duration/iters=-1.000000
	MyBFGSCudaGMLVQ32(gtol=1e-05, iterations=1000)                                                                                                                                                           Duration=0.57080 	Accuracy=72.294 iters=0.00 	duration/iters=-1.000000
	MyBFGSCudaGMLVQshared(gtol=1e-05, iterations=1000)                                                                                                                                                       Duration=0.51265 	Accuracy=72.294 iters=0.00 	duration/iters=-1.000000
uci_segmentation
	GmlvqModel(beta=2, c=None, display=False, gtol=1e-05, initial_matrix=None, initial_prototypes=None, initialdim=None, max_iter=1000, prototypes_per_class=1, random_state=None, regularization=0.0)       Duration=25.99925 	Accuracy=91.037 iters=982.35 	duration/iters=0.026466
	MyBFGSCudaGMLVQ_PyCuda(gtol=1e-05, iterations=1000)                                                                                                                                                      Duration=0.27874 	Accuracy=18.026 iters=302.65 	duration/iters=0.000921
	MyBFGSCudaGMLVQ_PyCuda_nonshared(gtol=1e-05, iterations=1000)                                                                                                                                            Duration=0.24720 	Accuracy=19.268 iters=259.35 	duration/iters=0.000953

"""

import time

import numpy as np
from pc.BFGS_GMLVQ_pycuda import MyBFGSCudaGMLVQ_PyCuda
from pc.BFGS_GMLVQ_pycuda_nonshared import MyBFGSCudaGMLVQ_PyCuda_nonshared
from sklearn.datasets import load_digits, load_iris, load_breast_cancer, load_wine
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score
from working.BFGS_GMLVQ import MyBFGSCudaGMLVQ
from working.BFGS_GMLVQ_32 import MyBFGSCudaGMLVQ32
from working.BFGS_GMLVQ_shared_memory import MyBFGSCudaGMLVQshared


def blobs_dataset(samples=1_000, prototypes=2, features=2):
    N = samples
    D = features  # dimensions / features
    M = prototypes  # number or prototypes
    X, y = make_blobs(n_samples=N, centers=M, n_features=D, random_state=2)  # 2,3
    X[:, 1] *= 10
    return f"blobs: {samples} samples, {prototypes} pts, {features} features", (X, y)


def load_uci_segmentation():
    file = "/Users/christoph/PycharmProjects/LVQ/matlab/LVQ_toolbox/data/segment.dat"
    data = np.loadtxt(file)
    y = data[:, -1] - 1  # zero based
    X = data[:, :-1]
    X = X[:, np.std(X, axis=0) != 0]  # feature 3 is constant -> exclude it
    return "uci_segmentation", (X, y)


all_datasets = [
    ("wine", load_wine(return_X_y=True)),
    ("breast cancer", load_breast_cancer(return_X_y=True)),
    ("iris", load_iris(return_X_y=True)),
    load_uci_segmentation(),
    ("digits", load_digits(return_X_y=True)),
]

max_iter = 100
all_models = [
    MyBFGSCudaGMLVQ(iterations=max_iter),
    MyBFGSCudaGMLVQ_PyCuda(iterations=max_iter),
    MyBFGSCudaGMLVQ_PyCuda_nonshared(iterations=max_iter),
    MyBFGSCudaGMLVQ32(iterations=max_iter),
    MyBFGSCudaGMLVQshared(iterations=max_iter)
]


def test_performance(models=None, datasets=None, repetitions=10, plot=False, no_accuracy=False):
    if models is None:
        models = all_models
    if datasets is None:
        datasets = all_datasets

    for name, (X, y) in datasets:
        for _ in range(repetitions):
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

                # .split("\n")[0]
                model_name = str(model).replace("\n", "").replace("\t", " ").replace("    ", " ").replace("    ", " ").replace("  ", " ").replace("  ", " ").replace("  ", " ")
                duration_per_iter = duration / num_iters if num_iters != 0 else -1
                title = f"\t{model_name:<200} Duration={duration:.5f} Accuracy={accuracy * 100:.3f} iters={num_iters:.2f} duration/iters={(duration_per_iter):.2f}"
                print(title)


test_performance()
