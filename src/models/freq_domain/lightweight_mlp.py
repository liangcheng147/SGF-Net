
import torch
import torch.nn as nn


class LightweightMLP(nn.Module):

    def __init__(self, input_dim=1024, hidden_dim=None):
        super(LightweightMLP, self).__init__()

        if hidden_dim is None:
            hidden_dim = input_dim

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        output = self.mlp(x)

        return output
