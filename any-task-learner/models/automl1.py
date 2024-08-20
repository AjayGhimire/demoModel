import torch
import torch.nn as nn
import torch.optim as optim


class AutoML1_Unsupervised(nn.Module):
    def __init__(self, input_dim, hidden_units):
        super(AutoML1_Unsupervised, self).__init__()

        # Convolutional layers for feature extraction
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # Fully connected layers for the output
        self.fc1 = nn.Linear(32 * (input_dim // 4) * (input_dim // 4), hidden_units)
        self.fc2 = nn.Linear(hidden_units, input_dim)

    def forward(self, x):
        # Feature extraction with convolutional layers
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * (x.shape[2] // 2) * (x.shape[3] // 2))

        # Fully connected layers
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
