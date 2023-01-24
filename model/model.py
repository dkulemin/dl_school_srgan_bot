"""
Архитектура и веса модели взяты отсюда:
https://github.com/Lornatang/SRGAN-PyTorch
"""
from pathlib import Path
from torch import Tensor
import math
import numpy as np
import torch
import torch.nn as nn

from utils import GENERATOR_PATH


class _ResidualConvBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super(_ResidualConvBlock, self).__init__()
        self.rcb = nn.Sequential(
            nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
            nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.rcb(x)

        out = torch.add(out, identity)

        return out
    

class _UpsampleBlock(nn.Module):
    def __init__(self, channels: int, upscale_factor: int) -> None:
        super(_UpsampleBlock, self).__init__()
        self.upsample_block = nn.Sequential(
            nn.Conv2d(channels, channels * upscale_factor * upscale_factor, (3, 3), (1, 1), (1, 1)),
            nn.PixelShuffle(2),
            nn.PReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        out = self.upsample_block(x)

        return out
    

class SRResNet(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            channels: int,
            num_rcb: int,
            upscale_factor: int
    ) -> None:
        super(SRResNet, self).__init__()
        # Low frequency information extraction layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, channels, (9, 9), (1, 1), (4, 4)),
            nn.PReLU(),
        )

        # High frequency information extraction block
        trunk = []
        for _ in range(num_rcb):
            trunk.append(_ResidualConvBlock(channels))
        self.trunk = nn.Sequential(*trunk)

        # High-frequency information linear fusion layer
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(channels),
        )

        # zoom block
        upsampling = []
        if upscale_factor == 2 or upscale_factor == 4 or upscale_factor == 8:
            for _ in range(int(math.log(upscale_factor, 2))):
                upsampling.append(_UpsampleBlock(channels, 2))
        elif upscale_factor == 3:
            upsampling.append(_UpsampleBlock(channels, 3))
        self.upsampling = nn.Sequential(*upsampling)

        # reconstruction block
        self.conv3 = nn.Conv2d(channels, out_channels, (9, 9), (1, 1), (4, 4))

        # Initialize neural network weights
        self._initialize_weights()

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

    # Support torch.script function
    def _forward_impl(self, x: Tensor) -> Tensor:
        out1 = self.conv1(x)
        out = self.trunk(out1)
        out2 = self.conv2(out)
        out = torch.add(out1, out2)
        out = self.upsampling(out)
        out = self.conv3(out)

        out = torch.clamp_(out, 0.0, 1.0)

        return out

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)


class SuperResolution():
    def __init__(self, upscale_factor: int = 4):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.generator = SRResNet(
            in_channels=3,
            out_channels=3,
            channels=64,
            num_rcb=16,
            upscale_factor=upscale_factor,
        ).to(self.device)

        if upscale_factor == 4:
            checkpoint = torch.load(GENERATOR_PATH, map_location=lambda storage, loc: storage)
            self.generator.load_state_dict(checkpoint["state_dict"])
        else:
            raise NotImplementedError('ERROR: not implemented for upscale_factor = {}'.format(upscale_factor))
        
    def __call__(self, image: np.array) -> np.array:
        self.generator.eval()
        with torch.no_grad():
            preprocessed_image = self._preprocess_image(image)
            high_res_image =  self.generator(preprocessed_image.unsqueeze(0).to(self.device))
            high_res_image = high_res_image.squeeze() * 255
        return high_res_image.detach().cpu().byte().permute(1, -1, 0).numpy()
    
    def _preprocess_image(self, image: np.array) -> torch.FloatTensor:
        return torch.FloatTensor(np.flip(image, axis=2).copy()).permute(-1, 0, 1) / 255
