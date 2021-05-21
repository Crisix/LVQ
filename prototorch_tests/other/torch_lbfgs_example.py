"""ProtoTorch GLVQ example using 2D Iris data."""
import time

import torch
from prototorch.functions.distances import euclidean_distance
from prototorch.modules.losses import GLVQLoss
from prototorch.modules.prototypes import Prototypes1D
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from datasets import load_uci_segmentation

# Prepare and preprocess the data

scaler = StandardScaler()

# x_train, y_train = load_iris(True)
# x_train, y_train = load_digits(return_X_y=True)
name, (x_train, y_train) = load_uci_segmentation()

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
                               data=(x_train, y_train),
                               prototype_initializer='stratified_mean',
                               dtype=torch.float32)

    def forward(self, x):
        protos = self.p1.prototypes
        plabels = self.p1.prototype_labels
        dis = euclidean_distance(self.omega(x), self.omega(protos))
        return dis, plabels


model = Model()

# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
optimizer = torch.optim.LBFGS(model.parameters(), lr=0.001)
criterion = GLVQLoss(squashing='sigmoid_beta')

print(f"Untrained Accuracy: {accuracy_score(y_train, model(x_train)[0].argmin(axis=1)):.3f}")

begin = time.time()


def closure():
    distances, plabels = model(x_train)
    loss = criterion([distances, plabels], torch.tensor(y_train))
    optimizer.zero_grad()
    if loss.requires_grad:
        loss.backward()
    return loss


for epoch in range(100):
    print(f'Epoch: {epoch + 1:03d}')
    optimizer.step(closure=closure)

end = time.time()
print(f"Accuracy: {accuracy_score(y_train, model(x_train)[0].argmin(axis=1)):.3f}")
print(f"Duration: {end - begin}")

#  Epoch: 40000 Loss: 184.80  Accuracy: 0.946
#  Epoch: 40000 Loss: 186.67  Accuracy: 0.946 Duration: 133.07541298866272
