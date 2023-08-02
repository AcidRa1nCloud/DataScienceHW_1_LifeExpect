import torch
from torch import nn

class Model_V0(nn.Module):
    """
    Simple Dense(fully conected) neural network
    """
    def __init__(self, in_units, hidden_units, out_units):
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
