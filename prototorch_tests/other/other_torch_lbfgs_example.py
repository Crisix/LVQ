"""ProtoTorch GLVQ example using 2D Iris data."""
import time

import matplotlib.pyplot as plt
import torch
from prototorch.functions.distances import euclidean_distance
from prototorch.modules.losses import GLVQLoss
from prototorch.modules.prototypes import Prototypes1D
# Prepare and preprocess the data
from PyTorch_LBFGS.functions.LBFGS import LBFGS, FullBatchLBFGS
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# x_train, y_train = load_iris(True)
x_train, y_train = load_digits(return_X_y=True)
# name, (x_train, y_train) = load_uci_segmentation()

x_train = torch.tensor(x_train, dtype=torch.float32)

input_dim = x_train.shape[-1]
nclasses = len(set(y_train))


# Define the GLVQ model
class Model(torch.nn.Module):
    def __init__(self, **kwargs):
        """GLVQ model."""
        super().__init__()
        self.omega = torch.nn.Linear(input_dim, input_dim)
        self.omega.weight.data.copy_(torch.eye(input_dim, dtype=torch.float32))
        self.p1 = Prototypes1D(input_dim=input_dim,
                               prototypes_per_class=1,
                               nclasses=nclasses,
                               prototype_initializer='stratified_mean',
                               dtype=torch.float32,
                               data=(x_train, y_train))

    def forward(self, x):
        protos = self.p1.prototypes
        plabels = self.p1.prototype_labels
        dis = euclidean_distance(self.omega(x), self.omega(protos))
        return dis, plabels


for i in range(1):
    model = Model()

    optimizer = FullBatchLBFGS(model.parameters(), lr=1., history_size=20, line_search='Wolfe', debug=False)
    # optimizer = FullBatchLBFGS(model.parameters(), lr=1., history_size=20, line_search='Armijo', debug=True)
    # optimizer = LBFGS(model.parameters(), lr=1., history_size=20, line_search='Wolfe', debug=False)
    criterion = GLVQLoss(squashing='sigmoid_beta')
    # criterion = GLVQLoss(squashing='identity')

    print(f"Untrained Accuracy: {accuracy_score(y_train, model(x_train)[0].argmin(axis=1)):.3f}")
    begin = time.time()

    # Calculate for 1st iteration
    optimizer.zero_grad()
    obj = criterion(model(x_train), torch.tensor(y_train))
    obj.backward()
    grad = optimizer._gather_flat_grad()


    def closure():
        optimizer.zero_grad()
        distances, plabels = model(x_train)
        loss = criterion([distances, plabels], torch.tensor(y_train))
        return loss


    times = []
    accs = []

    for epoch in range(100):
        model.train()

        options = {'closure': closure, 'current_loss': obj}
        obj, grad, lr, _, _, _, _, _ = optimizer.step(options)
        # obj, grad, lr, _, _, _ = optimizer.step(options)
        times.append(time.time())
        accs.append(accuracy_score(y_train, model(x_train)[0].argmin(axis=1)))

    end = time.time()
    print(f"Accuracy: {accuracy_score(y_train, model(x_train)[0].argmin(axis=1)):.3f}")
    print(f"Duration: {end - begin}")
    plt.plot(times, accs)
    plt.show()
    # TODO UserWarning: `prototype_initializer`: `stratified_mean` requires `data`, but `data` is not provided. Using randomly generated data instead.
    #   warnings.warn(

    #  Epoch: 40000 Loss: 184.80  Accuracy: 0.946
    #  Epoch: 40000 Loss: 186.67  Accuracy: 0.946 Duration: 133.07541298866272
