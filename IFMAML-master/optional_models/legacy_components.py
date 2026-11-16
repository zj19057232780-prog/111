"""Historical GFNet/ARSM/EMA implementations, preserved for checkpoint compatibility."""
import torch
import torch.nn.modules as nn
import torch.nn.functional as F

def maml_init_(module):
    torch.nn.init.xavier_uniform_(module.weight.data, gain=1.0)
    if module.bias is not None:
        torch.nn.init.constant_(module.bias.data, 0.0)
    return module


def norm_init_(module):
    torch.nn.init.constant_(module.weight.data, 1.0)
    torch.nn.init.constant_(module.bias.data, 0.0)
    return module


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


class AdaptiveResidualShrinkage(nn.Module):
    """
    Channel-wise adaptive soft-threshold denoising with identity-safe blending.

    The threshold follows residual shrinkage networks used in noisy bearing
    diagnosis. A small learnable blend keeps the initial mapping close to the
    existing backbone, which is important for stable second-order MAML updates.
    """

    def __init__(self, channels, reduction=4, blend_init=1e-3):
        super().__init__()
        hidden = max(channels // int(reduction), 4)
        blend_init = float(blend_init)
        if not 0.0 < blend_init < 1.0:
            raise ValueError(f'arsm blend_init must be in (0, 1), got {blend_init}')

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.threshold_net = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        blend_logit = torch.logit(torch.full((channels,), blend_init))
        self.blend_logit = torch.nn.Parameter(blend_logit)

        for module in self.threshold_net.modules():
            if isinstance(module, nn.Conv2d):
                maml_init_(module)

    def forward(self, x):
        magnitude = x.abs()
        channel_scale = self.pool(magnitude)
        threshold = self.threshold_net(channel_scale) * channel_scale
        shrunk = torch.sign(x) * F.relu(magnitude - threshold)
        blend = torch.sigmoid(self.blend_logit).view(1, -1, 1, 1)
        return x + blend * (shrunk - x)


def _valid_group_factor(channels, factor):
    factor = int(factor)
    factor = max(1, min(factor, channels))
    while channels % factor != 0 and factor > 1:
        factor -= 1
    return factor


class EMALiteAttention(nn.Module):
    """
    Lightweight adaptation of EMA cross-spatial attention.

    Reference:
    C:/Users/JieZhang/Desktop/new-ifmaml/模块3EMA/EMA-attention-module-main/
    EMA-attention-module-main/EMA_attention_module
    """

    def __init__(self, channels, factor=8):
        super().__init__()
        self.groups = _valid_group_factor(channels, factor)
        group_channels = channels // self.groups
        self.softmax = nn.Softmax(dim=-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(group_channels, group_channels)
        self.conv1x1 = nn.Conv2d(group_channels, group_channels, kernel_size=1, bias=True)
        self.conv3x3 = nn.Conv2d(group_channels, group_channels, kernel_size=3, padding=1, bias=True)

        norm_init_(self.gn)
        maml_init_(self.conv1x1)
        maml_init_(self.conv3x3)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, c // self.groups, h, w)
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)

        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)

        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(
            b * self.groups,
            1,
            h,
            w,
        )
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)


class EMALiteBlock(nn.Module):
    def __init__(self, channels, factor=8, layer_scale_init=1e-3):
        super().__init__()
        self.norm = nn.BatchNorm2d(channels)
        self.attn = EMALiteAttention(channels, factor=factor)
        self.layer_scale = torch.nn.Parameter(layer_scale_init * torch.ones(channels))

        norm_init_(self.norm)

    def forward(self, x):
        y = self.attn(self.norm(x))
        return x + self.layer_scale.view(1, -1, 1, 1) * y


