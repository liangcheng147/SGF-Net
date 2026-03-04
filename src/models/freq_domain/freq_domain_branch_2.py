
import torch
import torch.nn as nn
from .high_freq_branch import HighFreqBranch


class FreqDomainBranch2(nn.Module):

    def __init__(self):
        super(FreqDomainBranch2, self).__init__()

        self.high_freq_branch = HighFreqBranch()

        self.projection = nn.Linear(512, 512)

        self.alpha = nn.Parameter(torch.ones(3))

        self.head = nn.Sequential(
            *[
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(512, 1),
            ]
        )

    def forward(self, x):
        batch_size = x.size(0)

        x_lh = x[:, 0]
        x_hl = x[:, 1]
        x_hh = x[:, 2]

        features = []
        for variant in [x_lh, x_hl, x_hh]:
            high_freq_output = self.high_freq_branch(variant)
            high_freq_feature = high_freq_output['output']
            features.append(high_freq_feature)

        weights = torch.softmax(self.alpha, dim=0)

        z_512 = (
            weights[0] * features[0] +
            weights[1] * features[1] +
            weights[2] * features[2]
        )

        z = self.projection(z_512)

        p = self.head(z)

        return p, z
