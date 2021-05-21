"""ProtoTorch GLVQ example using 2D Iris data."""

import torch
from prototorch.functions.distances import euclidean_distance
from prototorch.modules.losses import GLVQLoss
from prototorch.modules.prototypes import Prototypes1D
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler

# Prepare and preprocess the data
scaler = StandardScaler()

# x_train, y_train = load_iris(True)
x_train, y_train = load_digits(return_X_y=True)


x_train = x_train[:, [0, 2]]
scaler.fit(x_train)
x_train = scaler.transform(x_train)


# Define the GLVQ model
class Model(torch.nn.Module):
    def __init__(self, **kwargs):
        """GLVQ model."""
        super().__init__()
        self.p1 = Prototypes1D(input_dim=2, prototypes_per_class=1, nclasses=3, prototype_initializer='stratified_mean')

    def forward(self, x):
        protos = self.p1.prototypes
        plabels = self.p1.prototype_labels
        dis = euclidean_distance(x, protos)
        return dis, plabels


# Build the GLVQ model
model = Model()

# Optimize using SGD optimizer from `torch.optim`
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = GLVQLoss(squashing='sigmoid_beta')

# Training loop
for epoch in range(300):
    # Compute loss.
    distances, plabels = model(torch.tensor(x_train))
    loss = criterion([distances, plabels], torch.tensor(y_train))
    print(f'Epoch: {epoch + 1:03d} Loss: {loss.item():02.02f}')

    # Take a gradient descent step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
