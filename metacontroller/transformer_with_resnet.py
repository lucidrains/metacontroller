from __future__ import annotations

import torch
from torch import nn, Tensor
from torch.nn import Module, ModuleList
from einops import rearrange, reduce
from einops.layers.torch import Rearrange, Reduce

from metacontroller.metacontroller import Transformer, Encoder

from torch_einops_utils import pack_with_inverse
from torch_einops_utils.save_load import save_load

# normalization

LOG_TWO_PI = 1.8378770664093453

class LayerNorm2d(Module):
    def __init__(self, dim, bias = False):
        super().__init__()
        self.layernorm = nn.LayerNorm(dim, bias = bias)

    def forward(self, x):
        x = rearrange(x, 'b c h w -> b h w c')
        x = self.layernorm(x)
        return rearrange(x, 'b h w c -> b c h w')

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
        downsample: Module | None = None,
        use_layernorm = False
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim_out, 3, stride = stride, padding = 1, bias = False)
        self.bn1 = LayerNorm2d(dim_out) if use_layernorm else nn.BatchNorm2d(dim_out)
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = nn.Conv2d(dim_out, dim_out, 3, padding = 1, bias = False)
        self.bn2 = LayerNorm2d(dim_out) if use_layernorm else nn.BatchNorm2d(dim_out)
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
        downsample: Module | None = None,
        use_layernorm = False
    ):
        super().__init__()
        width = dim_out # simple resnet shortcut
        self.conv1 = nn.Conv2d(dim, width, 1, bias = False)
        self.bn1 = LayerNorm2d(width) if use_layernorm else nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, 3, stride = stride, padding = 1, bias = False)
        self.bn2 = LayerNorm2d(width) if use_layernorm else nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, dim_out * self.expansion, 1, bias = False)
        self.bn3 = LayerNorm2d(dim_out * self.expansion) if use_layernorm else nn.BatchNorm2d(dim_out * self.expansion)
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

class BasicBlockDec(Module):
    expansion = 1

    def __init__(
        self,
        dim,
        dim_out,
        stride = 1,
        output_padding = 1,
        use_layernorm = False
    ):
        super().__init__()
        norm_layer = LayerNorm2d if use_layernorm else nn.BatchNorm2d

        self.conv2 = nn.Conv2d(dim, dim, 3, padding = 1, bias = False)
        self.bn2 = norm_layer(dim)
        self.relu = nn.ReLU(inplace = True)

        if stride > 1:
            self.conv1 = nn.ConvTranspose2d(dim, dim_out, 3, stride = stride, padding = 1, output_padding = output_padding, bias = False)
        else:
            self.conv1 = nn.Conv2d(dim, dim_out, 3, padding = 1, bias = False)

        self.bn1 = norm_layer(dim_out)

        self.upsample = None
        if stride > 1 or dim != dim_out:
            self.upsample = nn.Sequential(
                nn.ConvTranspose2d(dim, dim_out, 1, stride = stride, output_padding = output_padding if stride > 1 else 0, bias = False),
                norm_layer(dim_out)
            )

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv2(x)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv1(out)
        out = self.bn1(out)

        if exists(self.upsample):
            identity = self.upsample(x)

        out += identity
        return self.relu(out)

class ResNet(Module):
    def __init__(
        self,
        block: type[BasicBlock | Bottleneck],
        layers: list[int],
        num_classes = 1000,
        channels = 3,
        use_layernorm = False
    ):
        super().__init__()
        self.inplanes = 64
        self.use_layernorm = use_layernorm

        self.conv1 = nn.Conv2d(channels, 64, kernel_size = 7, stride = 2, padding = 3, bias = False)
        self.bn1 = LayerNorm2d(64) if use_layernorm else nn.BatchNorm2d(64)
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
                LayerNorm2d(planes * block.expansion) if self.use_layernorm else nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, use_layernorm = self.use_layernorm))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, use_layernorm = self.use_layernorm))

        return nn.Sequential(*layers)

    def forward(self, x, attn: Encoder | None = None, return_feature_map = False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        feature_map = x

        if exists(attn):
            x = rearrange(x, 'b c h w -> b (h w) c')
            x = attn(x)
            x = reduce(x, 'b ... c ->  b c', 'mean')
        else:
            x = reduce(x, 'b c ... -> b c', 'mean')

        x = self.fc(x)
        if return_feature_map:
            return x, feature_map
        return x

# resnet factory

def resnet18(num_classes = 1000, use_layernorm = False):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes, use_layernorm = use_layernorm)

def resnet34(num_classes = 1000, use_layernorm = False):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes, use_layernorm = use_layernorm)

def resnet50(num_classes = 1000, use_layernorm = False):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes, use_layernorm = use_layernorm)

class ResNetDecoder(Module):
    def __init__(
        self,
        block: type[BasicBlockDec],
        layers: list[int],
        output_channels = 6,
        use_layernorm = False,
        output_paddings = [0, 1, 0, 1]
    ):
        super().__init__()
        self.use_layernorm = use_layernorm
        self.inplanes = 512 * block.expansion

        self.layer4 = self._make_layer(block, 256, layers[3], stride = 2, output_padding = output_paddings[3])
        self.layer3 = self._make_layer(block, 128, layers[2], stride = 2, output_padding = output_paddings[2])
        self.layer2 = self._make_layer(block, 64, layers[1], stride = 2, output_padding = output_paddings[1])
        self.layer1 = self._make_layer(block, 64, layers[0], stride = 1, output_padding = output_paddings[0])

        self.upsample_conv1 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 3, stride = 2, padding = 1, output_padding = 1, bias = False),
            LayerNorm2d(64) if use_layernorm else nn.BatchNorm2d(64),
            nn.ReLU(inplace = True),
            nn.ConvTranspose2d(64, output_channels, 7, stride = 2, padding = 3, output_padding = 1, bias = False)
        )

    def _make_layer(
        self,
        block: type[BasicBlockDec],
        planes: int,
        blocks: int,
        stride: int = 1,
        output_padding: int = 1
    ) -> nn.Sequential:
        layers = []
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, self.inplanes, use_layernorm = self.use_layernorm))

        layers.append(block(self.inplanes, planes, stride, output_padding = output_padding, use_layernorm = self.use_layernorm))
        self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer4(x)
        x = self.layer3(x)
        x = self.layer2(x)
        x = self.layer1(x)
        x = self.upsample_conv1(x)
        return x

def resnet18_decoder(output_channels = 6, use_layernorm = False, output_paddings = [0, 1, 0, 1]):
    return ResNetDecoder(BasicBlockDec, [2, 2, 2, 2], output_channels, use_layernorm = use_layernorm, output_paddings = output_paddings)

# transformer with resnet

@save_load()
class TransformerWithResnet(Transformer):
    def __init__(
        self,
        *args,
        resnet_type = 'resnet18',
        is_channel_last = True,
        use_layernorm = False,
        norm_final_encoding = False,
        encoder_kwargs: dict | None = None,
        **kwargs
    ):
        kwargs["state_loss_detach_target_state"] = True
        super().__init__(*args, **kwargs)
        self.is_channel_last = is_channel_last

        # vis encoder
        resnet_klass = resnet18
        if resnet_type == 'resnet34':
            resnet_klass = resnet34
        elif resnet_type == 'resnet50':
            resnet_klass = resnet50
        self.resnet_dim = kwargs['state_embed_readout']['num_continuous']
        self.visual_encoder = resnet_klass(num_classes = self.resnet_dim, use_layernorm = use_layernorm)

        # vis decoder
        if resnet_type == 'resnet18':
            self.visual_decoder = resnet18_decoder(output_channels = 6, use_layernorm = use_layernorm)
        else: raise NotImplementedError()

        self.final_norm = nn.LayerNorm(self.resnet_dim) if norm_final_encoding else nn.Identity()

        # transformer
        self.attn = None
        if exists(encoder_kwargs):
            assert 'dim' not in encoder_kwargs
            encoder_kwargs['dim'] = self.visual_encoder.feature_dim
            self.attn = Encoder(**encoder_kwargs)

    def forward(self, state, actions = None, return_visual_autoencoder_loss = False, **kwargs):
        encoded_state, feature_map, normalized_pixels = self.visual_encode(state, return_feature_map = return_visual_autoencoder_loss)
        model_out = super().forward(encoded_state, actions = actions, **kwargs)

        if not return_visual_autoencoder_loss:
            return model_out

        visual_autoencoder_loss = self.visual_reconstruction_loss(feature_map, normalized_pixels)
        return model_out, visual_autoencoder_loss

    def visual_encode(self, x, return_feature_map = False):
        if self.is_channel_last:
            x = rearrange(x, '... h w c -> ... c h w')

        x, inverse = pack_with_inverse(x, '* c h w')

        if return_feature_map:
            h, feature_map = self.visual_encoder(x, attn = self.attn, return_feature_map = True)
        else:
            h = self.visual_encoder(x, attn = self.attn, return_feature_map = False)
            feature_map = None

        h = self.final_norm(h)

        encoded = inverse(h, '* d')
        if not return_feature_map:
            return encoded, None, None

        normalized_pixels = inverse(x, '* c h w')
        feature_map = inverse(feature_map, '* c h w')
        return encoded, feature_map, normalized_pixels

    def visual_reconstruction_loss(self, feature_map, normalized_pixels):
        _, _, target_h, target_w = normalized_pixels.shape[-4:]
        feature_map, _ = pack_with_inverse(feature_map, '* c h w')
        target = rearrange(normalized_pixels, '... c h w -> (...) c h w')

        recon_dist_params = self.visual_decoder(feature_map)

        recon_dist_params = rearrange(recon_dist_params, 'b (two c) h w -> b c h w two', two = 2)

        mean, log_var = recon_dist_params.unbind(dim = -1)
        log_var = log_var.clamp(min = -7., max = 5.)

        loss = 0.5 * (log_var + ((target - mean).square() * torch.exp(-log_var)) + LOG_TWO_PI)
        loss = loss.mean()
        return loss
