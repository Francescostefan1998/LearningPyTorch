import numpy as np
import matplotlib.pyplot as plt

X_train = np.arange(10, dtype='float32').reshape((10, 1))
y_train = np.array([1.0, 1.3, 3.1, 2.0, 5.0, 6.3, 6.6, 7.4, 8.0, 9.0], dtype='float32')
plt.plot(X_train, y_train, 'o', markersize=10)
plt.xlabel('x')
plt.ylabel('y')
plt.show()

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch
X_train_norm = (X_train - np.mean(X_train))/ np.std(X_train)
X_train_norm = torch.from_numpy(X_train_norm)
y_train = torch.from_numpy(y_train).float()
train_ds = TensorDataset(X_train_norm, y_train)
batch_size = 1
train_dl = DataLoader(train_ds, batch_size, shuffle=True)


torch.manual_seed(1)
weight = torch.randn(1)
weight.requires_grad_()
# Initialize the bias
#  because of requires_grad=True, PyTorch will track its gradient during training and let it update later.
bias = torch.zeros(1, requires_grad=True)
def model(xb):
    return xb @ weight + bias  # the snail in pythorch means matrix multiplication


def loss_fn(input, target):
    return (input-target).pow(2).mean()

learning_rate = 0.001
num_epochs = 200
log_epochs = 10
for epoch in range(num_epochs):
    for x_batch, y_batch in train_dl:
        pred = model(x_batch)
        loss = loss_fn(pred, y_batch.long())
        loss.backward()
    with torch.no_grad():
        weight -= weight.grad * learning_rate
        bias -= bias.grad * learning_rate
        weight.grad.zero_()
        bias.grad.zero_()
    if epoch % log_epochs == 0:
        print(f'Epoch {epoch}    Loss {loss.item():.4f}')

print('Final Parameters:', weight.item(), bias.item())
X_test = np.linspace(0, 9, num=100, dtype='float32').reshape(-1, 1)
X_test_norm = (X_test -np.mean(X_train))/np.std(X_train)
X_test_norm = torch.from_numpy(X_test_norm)
y_pred = model(X_test_norm)
fig = plt.figure(figsize=(13, 5))
ax = fig.add_subplot(1, 2, 1)
plt.plot(X_train_norm, y_train, 'o', markersize=10)
plt.plot(X_test_norm.numpy(), y_pred.detach().numpy(), '--', lw=3)
plt.legend(['Training examples', 'Linear reg.'], fontsize=15)
ax.set_xlabel('x', size=15)
ax.set_ylabel('y', size=15)
ax.tick_params(axis='both', which='major', labelsize=15)
plt.show()

import torch.nn as nn
loss_fn = nn.MSELoss(reduction='mean')
input_size = 1
output_size = 1
model = nn.Linear(input_size, output_size)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for x_batch, y_batch in train_dl:
        # Generate the prediction
        pred = model(x_batch)[:, 0]
        # Calculate loss
        loss = loss_fn(pred, y_batch)
        # Compute gradients
        loss.backward()
        # Update parameters using gradients
        optimizer.step()
        # Reset the gradients to zero
        optimizer.zero_grad()
    if epoch % log_epochs==0:
        print(f'Epoch {epoch}   Loss {loss.item():.4f}')


print(f'Final Parameters:', model.weight.item(), model.bias.item())
