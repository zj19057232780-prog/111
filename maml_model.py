import torch.nn.modules as nn
import torch.nn.functional as F
import torch


class SEAttention(nn.Module):

    def __init__(self, channel=64, reduction=2):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)




def maml_init_(module):
    torch.nn.init.xavier_uniform_(module.weight.data, gain=1.0)
    if module.bias is not None:
        torch.nn.init.constant_(module.bias.data, 0.0)
    return module


def norm_init_(module):
    torch.nn.init.constant_(module.weight.data, 1.0)
    torch.nn.init.constant_(module.bias.data, 0.0)
    return module


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, max_pool_factor=1.0):
        super().__init__()
        stride = (int(2 * max_pool_factor))
        self.max_pool = nn.MaxPool2d(kernel_size=stride, stride=stride, ceil_mode=False)
        self.normalize = nn.BatchNorm2d(out_channels, affine=True)
        torch.nn.init.uniform_(self.normalize.weight)
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=1, bias=True)

        
        maml_init_(self.conv)

    def forward(self, x):
        x = self.conv(x)
        x = self.normalize(x)
        x = self.relu(x)
        x = self.max_pool(x)
        return x


class ConvBase(nn.Sequential):
    def __init__(self, hidden=64, channels=3, layers=4, max_pool_factor=1.0):
        core = [ConvBlock(channels, hidden, 3, max_pool_factor)]
        for _ in range(layers - 1):
            core.append(ConvBlock(hidden, hidden, 3, max_pool_factor))
        super(ConvBase, self).__init__(*core)


class CNN4Backbone(ConvBase):
    def forward(self, x):
        x = super(CNN4Backbone, self).forward(x)
        x = x.reshape(x.size(0), -1)
        return x


class ConvBNAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, groups=1, act=True):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=True,
        )
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU() if act else nn.Identity()
        maml_init_(self.conv)
        norm_init_(self.norm)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class LSKSelectiveKernel(nn.Module):
    """
    Lightweight adaptation of LSKNet's large selective kernel block.

    Reference:
    C:/Users/JieZhang/Desktop/new-ifmaml/模块1LSK/LSKNet-main/LSKNet-main/
    mmrotate/models/backbones/lsknet.py
    """

    def __init__(self, channels):
        super().__init__()
        reduced = max(channels // 2, 1)
        self.conv5 = nn.Conv2d(channels, channels, 5, padding=2, groups=channels, bias=True)
        self.conv7_dilated = nn.Conv2d(
            channels,
            channels,
            7,
            padding=9,
            dilation=3,
            groups=channels,
            bias=True,
        )
        self.reduce5 = nn.Conv2d(channels, reduced, 1, bias=True)
        self.reduce7 = nn.Conv2d(channels, reduced, 1, bias=True)
        self.squeeze = nn.Conv2d(2, 2, 7, padding=3, bias=True)
        self.expand = nn.Conv2d(reduced, channels, 1, bias=True)

        for m in (self.conv5, self.conv7_dilated, self.reduce5, self.reduce7, self.squeeze, self.expand):
            maml_init_(m)

    def forward(self, x):
        attn5 = self.conv5(x)
        attn7 = self.conv7_dilated(attn5)

        attn5 = self.reduce5(attn5)
        attn7 = self.reduce7(attn7)
        attn = torch.cat([attn5, attn7], dim=1)

        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        gate = torch.cat([avg_attn, max_attn], dim=1)
        gate = self.squeeze(gate).sigmoid()

        selected = attn5 * gate[:, 0:1, :, :] + attn7 * gate[:, 1:2, :, :]
        selected = self.expand(selected)
        return x * selected


class LSKLiteBlock(nn.Module):
    def __init__(self, channels, mlp_ratio=2):
        super().__init__()
        hidden = int(channels * mlp_ratio)
        self.norm1 = nn.BatchNorm2d(channels)
        self.proj1 = nn.Conv2d(channels, channels, 1, bias=True)
        self.act = nn.GELU()
        self.lsk = LSKSelectiveKernel(channels)
        self.proj2 = nn.Conv2d(channels, channels, 1, bias=True)

        self.norm2 = nn.BatchNorm2d(channels)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden, channels, 1, bias=True),
        )
        self.layer_scale_1 = torch.nn.Parameter(1e-2 * torch.ones(channels))
        self.layer_scale_2 = torch.nn.Parameter(1e-2 * torch.ones(channels))

        norm_init_(self.norm1)
        norm_init_(self.norm2)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) and m not in self.lsk.modules():
                maml_init_(m)

    def forward(self, x):
        residual = x
        y = self.norm1(x)
        y = self.proj1(y)
        y = self.act(y)
        y = self.lsk(y)
        y = self.proj2(y)
        x = residual + self.layer_scale_1.view(1, -1, 1, 1) * y

        y = self.mlp(self.norm2(x))
        x = x + self.layer_scale_2.view(1, -1, 1, 1) * y
        return x


class GFNetLiteFilter(nn.Module):
    """
    Lightweight global frequency filter adapted from GFNet.

    Reference:
    C:/Users/JieZhang/Desktop/new-ifmaml/模块2gfnet/GFNet-master/GFNet-master/gfnet.py
    """

    def __init__(self, channels, spatial_size=8, weight_scale=0.02):
        super().__init__()
        if isinstance(spatial_size, int):
            height = width = spatial_size
        else:
            height, width = spatial_size
        self.height = int(height)
        self.width = int(width)
        self.complex_weight = torch.nn.Parameter(
            torch.randn(self.height, self.width // 2 + 1, channels, 2, dtype=torch.float32) * weight_scale
        )

    def _resize_weight(self, height, width):
        freq_width = width // 2 + 1
        if height == self.height and freq_width == self.complex_weight.shape[1]:
            return self.complex_weight

        channels = self.complex_weight.shape[2]
        weight = self.complex_weight.permute(2, 3, 0, 1).reshape(
            1,
            channels * 2,
            self.height,
            self.complex_weight.shape[1],
        )
        weight = F.interpolate(weight, size=(height, freq_width), mode='bilinear', align_corners=False)
        weight = weight.reshape(channels, 2, height, freq_width).permute(2, 3, 0, 1)
        return weight.contiguous()

    def forward(self, x):
        b, c, h, w = x.shape
        dtype = x.dtype

        x_freq = x.permute(0, 2, 3, 1).contiguous().float()
        x_freq = torch.fft.rfft2(x_freq, dim=(1, 2), norm='ortho')
        weight = torch.view_as_complex(self._resize_weight(h, w))
        x_freq = x_freq * weight.unsqueeze(0)
        x = torch.fft.irfft2(x_freq, s=(h, w), dim=(1, 2), norm='ortho')
        x = x.permute(0, 3, 1, 2).contiguous()
        return x.to(dtype=dtype)


class GFNetLiteBlock(nn.Module):
    def __init__(
        self,
        channels,
        spatial_size=8,
        mlp_ratio=2,
        weight_scale=0.02,
        layer_scale_init=1e-2,
    ):
        super().__init__()
        hidden = int(channels * mlp_ratio)
        self.norm1 = nn.BatchNorm2d(channels)
        self.filter = GFNetLiteFilter(channels, spatial_size=spatial_size, weight_scale=weight_scale)
        self.norm2 = nn.BatchNorm2d(channels)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden, channels, 1, bias=True),
        )
        self.layer_scale = torch.nn.Parameter(layer_scale_init * torch.ones(channels))

        norm_init_(self.norm1)
        norm_init_(self.norm2)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                maml_init_(m)

    def forward(self, x):
        y = self.filter(self.norm1(x))
        y = self.mlp(self.norm2(y))
        return x + self.layer_scale.view(1, -1, 1, 1) * y


class LSKLiteBackbone(nn.Module):
    """
    LSK-lite feature extractor for 64x64 STFT images.

    It keeps the MAML interface identical to CNN4Backbone but replaces
    fixed 3x3 convolution stacking with large-kernel selective convolution.
    """

    def __init__(
        self,
        channels=1,
        stage_channels=(32, 64, 96),
        stage_depths=(1, 1, 1),
        mlp_ratio=2,
        img_size=64,
        frequency_module='none',
        gfnet_depth=1,
        gfnet_mlp_ratio=2,
        gfnet_weight_scale=0.02,
        gfnet_layer_scale_init=1e-2,
    ):
        super().__init__()
        frequency_module = (frequency_module or 'none').lower()
        self.stem = ConvBNAct(channels, stage_channels[0], kernel_size=3, stride=2)

        stages = []
        in_channels = stage_channels[0]
        for stage_idx, out_channels in enumerate(stage_channels):
            if stage_idx > 0:
                stages.append(ConvBNAct(in_channels, out_channels, kernel_size=3, stride=2))
                in_channels = out_channels
            blocks = [LSKLiteBlock(out_channels, mlp_ratio=mlp_ratio) for _ in range(stage_depths[stage_idx])]
            stages.extend(blocks)
        self.stages = nn.Sequential(*stages)
        if frequency_module in ('none', 'identity', 'off', 'false'):
            self.freq_enhance = nn.Identity()
            self.frequency_module = 'none'
        elif frequency_module in ('gfnet', 'gfnet_lite', 'gfnetlite'):
            final_size = max(1, int(img_size) // (2 ** len(stage_channels)))
            self.freq_enhance = nn.Sequential(*[
                GFNetLiteBlock(
                    stage_channels[-1],
                    spatial_size=final_size,
                    mlp_ratio=gfnet_mlp_ratio,
                    weight_scale=gfnet_weight_scale,
                    layer_scale_init=gfnet_layer_scale_init,
                )
                for _ in range(gfnet_depth)
            ])
            self.frequency_module = 'gfnet_lite'
        else:
            raise ValueError(f'Unknown frequency_module: {frequency_module}')
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.embedding_size = stage_channels[-1]

    def forward(self, x):
        x = self.stem(x)
        x = self.stages(x)
        x = self.freq_enhance(x)
        x = self.pool(x).flatten(1)
        return x


class Net4CNN(torch.nn.Module):
    def __init__(self, output_size, hidden_size, layers, channels, embedding_size):
        super().__init__()
        self.features = CNN4Backbone(hidden_size, channels, layers, max_pool_factor=4 // layers)
        self.classifier = torch.nn.Linear(embedding_size, output_size, bias=True)
        
        maml_init_(self.classifier)
        self.hidden_size = hidden_size

    def forward(self, x):
        x1 = self.features(x)
        x2 = self.classifier(x1)
        return x1,x2
        # return x


class Net4LSK(torch.nn.Module):
    def __init__(
        self,
        output_size,
        channels=1,
        stage_channels=(32, 64, 96),
        stage_depths=(1, 1, 1),
        mlp_ratio=2,
        img_size=64,
        frequency_module='none',
        gfnet_depth=1,
        gfnet_mlp_ratio=2,
        gfnet_weight_scale=0.02,
        gfnet_layer_scale_init=1e-2,
    ):
        super().__init__()
        self.features = LSKLiteBackbone(
            channels=channels,
            stage_channels=tuple(stage_channels),
            stage_depths=tuple(stage_depths),
            mlp_ratio=mlp_ratio,
            img_size=img_size,
            frequency_module=frequency_module,
            gfnet_depth=gfnet_depth,
            gfnet_mlp_ratio=gfnet_mlp_ratio,
            gfnet_weight_scale=gfnet_weight_scale,
            gfnet_layer_scale_init=gfnet_layer_scale_init,
        )
        self.classifier = torch.nn.Linear(self.features.embedding_size, output_size, bias=True)
        maml_init_(self.classifier)
        self.embedding_size = self.features.embedding_size
        self.frequency_module = self.features.frequency_module

    def forward(self, x):
        x1 = self.features(x)
        x2 = self.classifier(x1)
        return x1, x2
