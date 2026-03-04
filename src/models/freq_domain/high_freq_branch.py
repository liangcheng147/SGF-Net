
import torch
import torch.nn as nn
import torch.fft as fft
import numpy as np
from .srm_filter import HPF
import torchvision.models as models


class CrossDifferenceFilter(nn.Module):

    def __init__(self):
        super(CrossDifferenceFilter, self).__init__()

        kernel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32)
        kernel_x = kernel_x.view(1, 1, 3, 3)

        kernel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32)
        kernel_y = kernel_y.view(1, 1, 3, 3)

        self.conv_x = nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=False, groups=3)
        self.conv_y = nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=False, groups=3)

        self.conv_x.weight = nn.Parameter(kernel_x.repeat(3, 1, 1, 1), requires_grad=False)
        self.conv_y.weight = nn.Parameter(kernel_y.repeat(3, 1, 1, 1), requires_grad=False)

    def forward(self, x):
        x_filtered = self.conv_x(x)
        y_filtered = self.conv_y(x)

        output = torch.sqrt(x_filtered ** 2 + y_filtered ** 2)

        return output


class HighFreqBranch(nn.Module):

    def __init__(self, k=4):
        super(HighFreqBranch, self).__init__()

        self.cross_diff_filter = CrossDifferenceFilter()
        self.srm_hpf = HPF()
        self.k = k

        self.resnet18 = models.resnet18(pretrained=True)

        self.resnet18.conv1 = nn.Conv2d(30, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        self.resnet18.fc = nn.Identity()

        for param in self.resnet18.parameters():
            param.requires_grad = True

    def _fft_transform(self, x):
        fft_result = fft.fft2(x, dim=(-2, -1))
        fft_result = fft.fftshift(fft_result, dim=(-2, -1))
        mag = torch.abs(fft_result)
        phase = torch.angle(fft_result)

        return mag, phase

    def _top_k_selection(self, mag, phase):
        batch_size, channels, height, width = mag.shape
        mag_flat = mag.view(batch_size, channels, -1)
        topk_values, topk_indices = torch.topk(mag_flat, self.k, dim=-1)
        mask = torch.zeros_like(mag_flat)
        mask.scatter_(-1, topk_indices, 1.0)
        mask = mask.view(batch_size, channels, height, width)
        selected_mag = mag * mask
        selected_phase = phase * mask

        return selected_mag, selected_phase

    def _ifft_transform(self, mag, phase):
        complex_spec = mag * torch.exp(1j * phase)
        complex_spec = fft.ifftshift(complex_spec, dim=(-2, -1))
        output = fft.ifft2(complex_spec, dim=(-2, -1))
        output = output.real

        return output

    def forward(self, x):
        x_srm = self.srm_hpf(x)

        output = self.resnet18(x_srm)

        return {
            'output': output,
            'srm_output': x_srm
        }
