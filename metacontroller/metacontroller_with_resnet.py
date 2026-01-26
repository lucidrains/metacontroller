from typing import Any, List, Type, Union, Optional

import torch
from torch import Tensor
from torch import nn
from einops import rearrange
from metacontroller.metacontroller import Transformer

class TransformerWithResnetEncoder(Transformer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.resnet_dim = kwargs["state_embed_readout"]["num_continuous"]
        self.visual_encoder = resnet18(out_dim=self.resnet_dim)

    def visual_encode(self, x: torch.Tensor) -> torch.Tensor:
        b, t = x.shape[:2]
        x = rearrange(x, 'b t h w c -> (b t) c h w')
        h = self.visual_encoder(x)
        h = rearrange(h, '(b t) d -> b t d', b=b, t=t, d = self.resnet_dim)
        return h

# resnet components taken from https://github.com/Lornatang/ResNet-PyTorch

class _BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: int,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_channels: int = 64,
    ) -> None:
        super(_BasicBlock, self).__init__()
        self.stride = stride
        self.downsample = downsample
        self.groups = groups
        self.base_channels = base_channels

        self.conv1 = nn.Conv2d(in_channels, out_channels, (3, 3), (stride, stride), (1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, (3, 3), (1, 1), (1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = torch.add(out, identity)
        out = self.relu(out)

        return out


class _Bottleneck(nn.Module):
    expansion: int = 4

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: int,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_channels: int = 64,
    ) -> None:
        super(_Bottleneck, self).__init__()
        self.stride = stride
        self.downsample = downsample
        self.groups = groups
        self.base_channels = base_channels

        channels = int(out_channels * (base_channels / 64.0)) * groups

        self.conv1 = nn.Conv2d(in_channels, channels, (1, 1), (1, 1), (0, 0), bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, (3, 3), (stride, stride), (1, 1), groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv3 = nn.Conv2d(channels, int(out_channels * self.expansion), (1, 1), (1, 1), (0, 0), bias=False)
        self.bn3 = nn.BatchNorm2d(int(out_channels * self.expansion))
        self.relu = nn.ReLU(True)

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = torch.add(out, identity)
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(
            self,
            arch_cfg: List[int],
            block: Type[Union[_BasicBlock, _Bottleneck]],
            groups: int = 1,
            channels_per_group: int = 64,
            out_dim: int = 1000,
    ) -> None:
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.dilation = 1
        self.groups = groups
        self.base_channels = channels_per_group

        self.conv1 = nn.Conv2d(3, self.in_channels, (7, 7), (2, 2), (3, 3), bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(True)
        self.maxpool = nn.MaxPool2d((3, 3), (2, 2), (1, 1))

        self.layer1 = self._make_layer(arch_cfg[0], block, 64, 1)
        self.layer2 = self._make_layer(arch_cfg[1], block, 128, 2)
        self.layer3 = self._make_layer(arch_cfg[2], block, 256, 2)
        self.layer4 = self._make_layer(arch_cfg[3], block, 512, 2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(512 * block.expansion, out_dim)

        # Initialize neural network weights
        self._initialize_weights()

    def _make_layer(
            self,
            repeat_times: int,
            block: Type[Union[_BasicBlock, _Bottleneck]],
            channels: int,
            stride: int = 1,
    ) -> nn.Sequential:
        downsample = None

        if stride != 1 or self.in_channels != channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, channels * block.expansion, (1, 1), (stride, stride), (0, 0), bias=False),
                nn.BatchNorm2d(channels * block.expansion),
            )

        layers = [
            block(
                self.in_channels,
                channels,
                stride,
                downsample,
                self.groups,
                self.base_channels
            )
        ]
        self.in_channels = channels * block.expansion
        for _ in range(1, repeat_times):
            layers.append(
                block(
                    self.in_channels,
                    channels,
                    1,
                    None,
                    self.groups,
                    self.base_channels,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        out = self._forward_impl(x)

        return out

    # Support torch.script function
    def _forward_impl(self, x: Tensor) -> Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)


def resnet18(**kwargs: Any) -> ResNet:
    model = ResNet([2, 2, 2, 2], _BasicBlock, **kwargs)

    return model


def resnet34(**kwargs: Any) -> ResNet:
    model = ResNet([3, 4, 6, 3], _BasicBlock, **kwargs)

    return model


def resnet50(**kwargs: Any) -> ResNet:
    model = ResNet([3, 4, 6, 3], _Bottleneck, **kwargs)

    return model


def resnet101(**kwargs: Any) -> ResNet:
    model = ResNet([3, 4, 23, 3], _Bottleneck, **kwargs)

    return model


def resnet152(**kwargs: Any) -> ResNet:
    model = ResNet([3, 8, 36, 3], _Bottleneck, **kwargs)

    return model