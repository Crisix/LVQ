import time

import prototorch as pt
import torch
from prototorch.functions.distances import euclidean_distance
from prototorch.modules.losses import GLVQLoss
from prototorch.modules.prototypes import Prototypes1D
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from datasets import load_uci_segmentation

scaler = StandardScaler()

# x_train, y_train = load_iris(True)
# x_train, y_train = load_digits(return_X_y=True)
name, (x_train, y_train) = load_uci_segmentation()

x_train = torch.tensor(x_train, dtype=torch.float32)

input_dim = x_train.shape[-1]
nclasses = len(set(y_train))


# Define the GLVQ model
class GmlvqTorchAdam(torch.nn.Module, BaseEstimator, ClassifierMixin):

    def __init__(self,
                 nclasses,
                 input_dim,
                 prototypes_per_class=1,
                 prototype_initializer="stratified_mean",
                 dtype=torch.float32,
                 distance_fn=pt.functions.distances.euclidean_distance,
                 **kwargs):
        super().__init__()
        self.distance_fn = distance_fn
        self.omega = torch.nn.Linear(input_dim, input_dim)
        self.omega.weight.data.copy_(torch.eye(input_dim, dtype=torch.float32))
        self.p1 = Prototypes1D(input_dim=input_dim,
                               prototypes_per_class=prototypes_per_class,
                               nclasses=nclasses,
                               prototype_initializer=prototype_initializer,
                               data=(x_train, y_train),
                               dtype=dtype)

    def forward(self, x):
        protos = self.p1.prototypes
        plabels = self.p1.prototype_labels
        dis = self.distance_fn(self.omega(x), self.omega(protos))
        return dis, plabels


model = GmlvqTorchAdam(nclasses, input_dim)

# Optimize using SGD optimizer from `torch.optim`
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# optimizer = torch.optim.LBFGS(model.parameters(), lr=0.001)
criterion = GLVQLoss(squashing='sigmoid_beta')

begin = time.time()

# Training loop
# for epoch in range(750):
for epoch in range(1000):
    # Compute loss.
    distances, plabels = model(x_train)
    loss = criterion([distances, plabels], torch.tensor(y_train))
    print(f'Epoch: {epoch + 1:03d} Loss: {loss.item():02.02f}')

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

end = time.time()
print(f"Accuracy: {accuracy_score(y_train, model(x_train)[0].argmin(axis=1)):.3f}")
print(f"Duration: {end - begin}")

#  Epoch: 40000 Loss: 184.80  Accuracy: 0.946
#  Epoch: 40000 Loss: 186.67  Accuracy: 0.946 Duration: 133.07541298866272
