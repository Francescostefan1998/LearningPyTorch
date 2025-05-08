import numpy as np
X = np.array([1, 1.4, 2.5]) # first value must be 1
w = np.array([0.4, 0.3, 0.5])
def net_input(X, w):
    return np.dot(X, w)
def logistic(z):
    return 1.0/(1.0 + np.exp(-z))
def logistic_activation(X, w):
    z = net_input(X, w)
    return logistic(z)
print(f'P(y=1|x) = {logistic_activation(X, w):.3f}')

# W : array with shape = (n_outputs_unit, n_hidden_units+1)
W = np.array([[1.1, 1.2, 0.8, 0.4],
              [0.2, 0.4, 1.0, 0.2],
              [0.6, 1.5, 1.2, 0.7]])

A  = np.array([[1, 0.1, 0.4, 0.6]])
Z = np.dot(W, A[0])
y_probas= logistic(Z)
print('Net Input: \n', Z)
print('Output Units:\n', y_probas)

y_class = np.argmax(Z, axis=0)
print('Predicted class label:', y_class)

# Softmax
def softmax(z):
    return np.exp(z) / np.sum(np.exp(z))
y_probas = softmax(Z)
print('Probabilities:\n', y_probas)
np.sum(y_probas)
print(np.sum(y_probas))

import torch
torch.softmax(torch.from_numpy(Z), dim=0)

import matplotlib.pyplot as plt

def tanh(z):
    e_p = np.exp(z)
    e_m = np.exp(-z)
    return (e_p - e_m) / (e_p + e_m)

z = np.arange(-5, 5, 0.005)
log_act = logistic(z)
tanh_act = tanh(z)
plt.ylim([-1.5, 1.5])
plt.xlabel('net input $z$')
plt.ylabel('activation $\phi(z)$')
plt.axhline(1, color='black', linestyle=':')
plt.axhline(0.5, color='black', linestyle=':')
plt.axhline(0, color='black', linestyle=':')
plt.axhline(-0.5, color='black', linestyle=':')
plt.axhline(-1, color='black', linestyle=':')
plt.plot(z, tanh_act, linewidth=3, linestyle='--', label='tanh')
plt.plot(z, log_act, linewidth=3, label='logistic')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()

np.tanh(z)
print(np.tanh(z))
torch.tanh(torch.from_numpy(z))
print(torch.tanh(torch.from_numpy(z)))
from scipy.special import expit
expit(z)
print(expit(z))
torch.sigmoid(torch.from_numpy(z))
print(torch.sigmoid(torch.from_numpy(z)))
torch.relu(torch.from_numpy(z))
print(torch.relu(torch.from_numpy(z)))