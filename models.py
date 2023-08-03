import torch
from torch import nn

class Model_V0(nn.Module):
    """
    Simple Dense(fully conected) neural network
    """
    def __init__(self, in_units, hidden_units, out_units, dropout_prob=0.5):
        super().__init__()

        self.linear_layer = nn.Sequential(
            nn.Linear(in_units, hidden_units),
            nn.Linear(hidden_units, hidden_units),
            nn.Linear(hidden_units, hidden_units),
            nn.Linear(hidden_units, hidden_units),
            nn.Linear(hidden_units, out_units)
        )

    def forward(self, x):
        return self.linear_layer(x)

class Model_V1(nn.Module):
    """
    Simple Dense(fully conected) neural network
    """
    def __init__(self, in_units, hidden_units, out_units, dropout_prob=0.5):
        super().__init__()

        self.linear_layer = nn.Sequential(
            nn.Linear(in_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, out_units)
        )

    def forward(self, x):
        return self.linear_layer(x)

class Model_V2(nn.Module):
    """
    Simple Dense(fully conected) neural network
    """
    def __init__(self, in_units, hidden_units, out_units, dropout_prob=0.5):
        super().__init__()

        self.linear_layer = nn.Sequential(
            nn.Linear(in_units, hidden_units),
            nn.Dropout(p=dropout_prob),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.Dropout(p=dropout_prob),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.Dropout(p=dropout_prob),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.Dropout(p=dropout_prob),
            nn.ReLU(),
            nn.Linear(hidden_units, out_units)
        )

    def forward(self, x):
        return self.linear_layer(x)
