import torch.nn as nn
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 10)  # Reduced the number of neurons in the first layer
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        return x

    def get_weights(self):
        return [param.data for param in self.parameters()]

    def set_weights(self, weights):
        for param, weight in zip(self.parameters(), weights):
            param.data = weight
