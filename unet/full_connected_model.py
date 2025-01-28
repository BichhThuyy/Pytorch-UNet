import torch
import torch.nn as nn
import torch.optim as optim


class FullyConnectedModel(nn.Module):
    def __init__(self, in_channels):
        super(FullyConnectedModel, self).__init__()

        # Input size: 2 channels of 512x512 = 2 * 512 * 512 = 524288
        self.fc1 = nn.Linear(in_channels * 512 * 512, 1024)  # First fully connected layer
        self.fc2 = nn.Linear(1024, 512)  # Second fully connected layer
        self.fc3 = nn.Linear(512, 256)  # Third fully connected layer
        self.fc4 = nn.Linear(256, 128)  # Fourth fully connected layer
        self.fc5 = nn.Linear(128, 1)  # Fifth layer, output layer

        # dropout
        self.dropout = nn.Dropout(0.5)

        # Activation function
        self.relu = nn.ReLU()
        # Sigmoid for binary classification
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Flatten the input from (batch_size, 2, 512, 512) to (batch_size, 524288)
        x = x.view(x.size(0), -1)  # Flattening

        # Pass through each fully connected layer with ReLU activations
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.dropout(self.relu(self.fc3(x)))
        x = self.dropout(self.relu(self.fc4(x)))

        # Output layer with sigmoid activation
        x = self.sigmoid(self.fc5(x))

        return x
