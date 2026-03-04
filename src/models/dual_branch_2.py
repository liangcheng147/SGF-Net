
import torch
import torch.nn as nn


class RINEWithDWTBranch(nn.Module):

    def __init__(self, rine_model, dwt_branch, classifier):
        super(RINEWithDWTBranch, self).__init__()

        self.rine_model = rine_model
        self.dwt_branch = dwt_branch
        self.classifier = classifier

        self.norm_rine = nn.LayerNorm(1024)
        self.norm_dwt = nn.LayerNorm(512)

        self.dwt_projection = nn.Linear(512, 1024)

        self.gate = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, t, c, h, w = x.shape

        tokens = x[:, 0]

        dwt_input = x[:, 1:]

        rine_p, rine_z = self.rine_model(tokens)

        _, dwt_z = self.dwt_branch(dwt_input)

        rine_z = self.norm_rine(rine_z)
        dwt_z = self.norm_dwt(dwt_z)

        projected_dwt_z = self.dwt_projection(dwt_z)

        gate_weight = self.gate(rine_z)

        fused_z = rine_z * gate_weight + projected_dwt_z * (1 - gate_weight)

        p = self.classifier(fused_z)

        return p, fused_z
