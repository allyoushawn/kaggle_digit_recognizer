import torch.nn as nn
import torch.nn.functional as F
import pdb

class NeuralNetwork(nn.Module):
    def __init__(self):
        # If the parent class has defined __init__, use it first
        super(NeuralNetwork, self).__init__()

        # #in-channels, #out-channels, kernel_size
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 50, 5)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
