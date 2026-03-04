

import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from torchvision.transforms import functional as F


class DWTPreprocessor:
    
    def __init__(self):
        self.max_size_small = 512
        self.max_size_medium = 1024
        self.max_size_large = 1536
    
    def _haar_dwt_2d(self, x):
        B, C, H, W = x.shape
        
        assert H % 2 == 0 and W % 2 == 0, "输入尺寸必须是偶数"
        
        x0 = x[:, :, :, 0::2]
        x1 = x[:, :, :, 1::2]
        
        x_ll = (x0[:, :, 0::2] + x0[:, :, 1::2] + x1[:, :, 0::2] + x1[:, :, 1::2]) / 4
        x_lh = (x0[:, :, 0::2] - x0[:, :, 1::2] + x1[:, :, 0::2] - x1[:, :, 1::2]) / 4
        x_hl = (x0[:, :, 0::2] + x0[:, :, 1::2] - x1[:, :, 0::2] - x1[:, :, 1::2]) / 4
        x_hh = (x0[:, :, 0::2] - x0[:, :, 1::2] - x1[:, :, 0::2] + x1[:, :, 1::2]) / 4
        
        return x_ll, x_lh, x_hl, x_hh
    
    def preprocess(self, image):
        if isinstance(image, torch.Tensor):
            image = F.to_pil_image(image)
        
        img_tensor = F.to_tensor(image)
        img_tensor = img_tensor.unsqueeze(0)
        
        B, C, H, W = img_tensor.shape
        if H % 2 != 0:
            img_tensor = img_tensor[:, :, :-1, :]
        if W % 2 != 0:
            img_tensor = img_tensor[:, :, :, :-1]
        
        ll, lh, hl, hh = self._haar_dwt_2d(img_tensor)
        
        high_freq_bands = [lh, hl, hh]
        
        processed_bands = []
        for band in high_freq_bands:
            band = band.squeeze(0)
            processed_bands.append(band)
        
        return processed_bands
