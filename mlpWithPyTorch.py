import numpy as np 
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split 
import torch
import torch.nn as nn
iris = load_iris()
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
X = iris['data']
y = iris['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1./3, random_state=1)
X_train_norm = (X_train - np.mean(X_train))/np.std(X_train)
X_train_norm = torch.from_numpy(X_train_norm).float()
y_train = torch.from_numpy(y_train)
train_ds = TensorDataset(X_train_norm, y_train)
torch.manual_seed(1)
batch_size = 2
train_dl = DataLoader(train_ds, batch_size, shuffle=True)


class Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        x = self.layer1(x)
        x = nn.Sigmoid()(x)
        x = self.layer2(x)
        return x
    
input_size = X_train_norm.shape[1]
hidden_size = 16
output_size = 3
model = Model(input_size, hidden_size, output_size)
# Set the learning rate for the optimizer
learning_rate = 0.001
# Define the loss function: CrossEntropyLoss is used for multi-class classification
loss_fn = nn.CrossEntropyLoss()
# Create an Adam optimizer to update model weights using gradients
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# Set the number of times the entire training dataset will be passed through the model
num_epochs = 100
# Initialize a list to store loss values for each epoch (used for plotting later)
loss_hist = [0] * num_epochs
# Initialize a list to store accuracy values for each epoch
accuracy_hist = [0] * num_epochs
# Start the training loop over the defined number of epochs
for epoch in range(num_epochs):

    # Loop over batches of data from the training dataloader
    for x_batch, y_batch in train_dl:

        # Forward pass: get predictions from the model for the input batch
        pred = model(x_batch)
        # Compute the loss between predictions and true labels
        loss = loss_fn(pred, y_batch)
        # Backward pass: compute gradients of the loss w.r.t. model parameters
        loss.backward()
        # Update model parameters using computed gradients
        optimizer.step()
        # Reset gradients to zero (important, otherwise they accumulate)
        optimizer.zero_grad()
        # Accumulate the total loss for this epoch (scaled by batch size)
        loss_hist[epoch] += loss.item() * y_batch.size(0)
        # Check which predictions were correct (argmax picks the class with highest score)
        is_correct = (torch.argmax(pred, dim=1) == y_batch).float()
        # Accumulate the total number of correct predictions for this epoch
        accuracy_hist[epoch] += is_correct.sum()

    # Average the total loss over the entire dataset size to get mean loss for the epoch
    loss_hist[epoch] /= len(train_dl.dataset)
    # Average the total correct predictions to get accuracy for the epoch
    accuracy_hist[epoch] /= len(train_dl.dataset)

# Create a new figure for plotting with a custom size (there’s a typo in figsize)
fig = plt.figure(figsize=(12, 5))
# Add the first subplot for loss
ax = fig.add_subplot(1, 2, 1)
# Plot the training loss history
ax.plot(loss_hist, lw=3)
# Set the title of the loss plot
ax.set_title('Training loss', size=15)
# Label the x-axis as 'Epoch'
ax.set_xlabel('Epoch', size=15)
# Adjust tick label size
ax.tick_params(axis='both', which='major', labelsize=15)
# Add the second subplot for accuracy
ax = fig.add_subplot(1, 2, 2)
# Plot the training accuracy history
ax.plot(accuracy_hist, lw=3)
# Set the title of the accuracy plot
ax.set_title('Training accuracy', size=15)
# Label the x-axis as 'Epoch'
ax.set_xlabel('Epoch', size=15)
# Adjust tick label size
ax.tick_params(axis='both', which='major', labelsize=15)
# Show the plot
plt.show()


X_test_norm = (X_test - np.mean(X_train))/ np.std(X_train)
X_test_norm = torch.from_numpy(X_test_norm).float()
y_test = torch.from_numpy(y_test)
pred_test = model(X_test_norm)
correct = (torch.argmax(pred_test, dim=1) == y_test).float()
accuracy = correct.mean()
print(f'Test Acc.: {accuracy:.4f}')

path = 'iris_classifier.pt'
torch.save(model, path)
model_new = torch.load(path)
model_new.eval()


pred_test = model_new(X_test_norm)
correct = (torch.argmax(pred_test, dim=1) == y_test).float()
accuracy = correct.mean()
print(f'Test Acc.: {accuracy:.4f}')
path = 'iris_classifier_state.pt'
torch.save(model.state_dict(), path)
model_new = Model(input_size, hidden_size, output_size)
model_new.load_state_dict(torch.load(path))