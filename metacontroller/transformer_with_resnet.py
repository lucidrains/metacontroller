from __future__ import annotations

import torch
from torch import nn, Tensor
from torch.nn import Module, ModuleList
from einops import rearrange, reduce
from einops.layers.torch import Rearrange, Reduce

from metacontroller.metacontroller import Transformer, Encoder

from torch_einops_utils import pack_with_inverse
from torch_einops_utils.save_load import save_load

# resnet components

def exists(v):
    return v is not None

class BasicBlock(Module):
    expansion = 1

    def __init__(
        self,
        dim,
        dim_out,
        stride = 1,
        downsample: Module | None = None
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim_out, 3, stride = stride, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(dim_out)
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = nn.Conv2d(dim_out, dim_out, 3, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(dim_out)
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if exists(self.downsample):
            identity = self.downsample(x)

        out += identity
        return self.relu(out)

class Bottleneck(Module):
    expansion = 4

    def __init__(
        self,
        dim,
        dim_out,
        stride = 1,
        downsample: Module | None = None
    ):
        super().__init__()
        width = dim_out # simple resnet shortcut
        self.conv1 = nn.Conv2d(dim, width, 1, bias = False)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, 3, stride = stride, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, dim_out * self.expansion, 1, bias = False)
        self.bn3 = nn.BatchNorm2d(dim_out * self.expansion)
        self.relu = nn.ReLU(inplace = True)
        self.downsample = downsample

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

        if exists(self.downsample):
            identity = self.downsample(x)

        out += identity
        return self.relu(out)

class ResNet(Module):
    def __init__(
        self,
        block: type[BasicBlock | Bottleneck],
        layers: list[int],
        num_classes = 1000,
        channels = 3
    ):
        super().__init__()
        self.inplanes = 64

        self.conv1 = nn.Conv2d(channels, 64, kernel_size = 7, stride = 2, padding = 3, bias = False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace = True)
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride = 2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride = 2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride = 2)

        self.feature_dim = 512 * block.expansion

        self.fc = nn.Linear(self.feature_dim, num_classes)

    def _make_layer(
        self,
        block: type[BasicBlock | Bottleneck],
        planes: int,
        blocks: int,
        stride: int = 1
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, 1, stride = stride, bias = False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, attn: Encoder | None = None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if exists(attn):
            x = rearrange(x, 'b c h w -> b (h w) c')
            x = attn(x)
            x = reduce(x, 'b ... c ->  b c', 'mean')
        else:
            x = reduce(x, 'b c ... -> b c', 'mean')

        x = self.fc(x)
        return x

# resnet factory

def resnet18(num_classes = 1000):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)

def resnet34(num_classes = 1000):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes)

def resnet50(num_classes = 1000):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes)

# transformer with resnet

@save_load()
class TransformerWithResnet(Transformer):
    def __init__(
        self,
        *,
        resnet_type = 'resnet18',
        is_channel_last = True,
        encoder_kwargs: dict | None = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.is_channel_last = is_channel_last

        resnet_klass = resnet18
        if resnet_type == 'resnet34':
            resnet_klass = resnet34
        elif resnet_type == 'resnet50':
            resnet_klass = resnet50

        self.resnet_dim = kwargs['state_embed_readout']['num_continuous']
        self.visual_encoder = resnet_klass(num_classes = self.resnet_dim)

        self.attn = None
        if exists(encoder_kwargs):
            assert 'dim' not in encoder_kwargs
            encoder_kwargs['dim'] = self.visual_encoder.feature_dim

            self.attn = Encoder(**encoder_kwargs)

    def visual_encode(self, x):
        if self.is_channel_last:
            x = rearrange(x, '... h w c -> ... c h w')

        x, inverse = pack_with_inverse(x, '* c h w')

        h = self.visual_encoder(x, attn = self.attn)

        return inverse(h, '* d')
