#pytorch

import torch
from torch import nn
import torch.nn.functional as F


class neuralnetwork(nn.Module):
    def __init__(self):
        super(neuralnetwork, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)
        self.fc3 = nn.Linear(10, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x))
        return x

model = neuralnetwork()
print(model)
