import torch
import torch.nn as nn
from my_modules import MyModule
from collections import OrderedDict

import os
import time
import math
import copy
from functools import partial
from typing import Optional, Callable, Any
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, trunc_normal_
from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count, parameter_count

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"
# train speed is slower after enabling this opts.
# torch.backends.cudnn.enabled = True
# torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.deterministic = True

try:
    from models_mamba.csm_triton import cross_scan_fn, cross_merge_fn
except:
    from models_mamba.csm_triton import cross_scan_fn, cross_merge_fn

try:
    from models_mamba.csms6s import selective_scan_fn, selective_scan_flop_jit
except:
    from models_mamba.csms6s import selective_scan_fn, selective_scan_flop_jit

# FLOPs counter not prepared fro mamba2
try:
    from models_mamba.mamba2.ssd_minimal import selective_scan_chunk_fn
except:
    from models_mamba.mamba2.ssd_minimal import selective_scan_chunk_fn



OPS = {
    # 'skip_connect': lambda C, stride: Identity() if stride == 1 else FactorizedReduce(C, C, affine=False),
    'skip_connect': lambda C, stride, dropout_rate: Identity(),
    # 'VSSBlock': lambda dim, stride: VSSBlock(hidden_dim=dim, stride=stride, ssm_d_state=16, scan_mode="tw_scanning_paths"),
    'Block_16_2x2x2_12': lambda dim, stride, dropout_rate: MyBlock(hidden_dim=dim, stride=stride, ssm_d_state=16, scan_mode="tw_scanning_paths", dropout_rate=dropout_rate),
    'Block_16_4x4x4_12': lambda dim, stride, dropout_rate: MyBlock_Sample(hidden_dim=dim, stride=stride, ssm_d_state=16, sample=2, scan_mode="tw_scanning_paths", dropout_rate=dropout_rate),
    'Block_16_8x8x8_12': lambda dim, stride, dropout_rate: MyBlock_Sample(hidden_dim=dim, stride=stride, ssm_d_state=16, sample=3, scan_mode="tw_scanning_paths", dropout_rate=dropout_rate),
    'Block_32_2x2x2_12': lambda dim, stride, dropout_rate: MyBlock(hidden_dim=dim, stride=stride, ssm_d_state=32, scan_mode="tw_scanning_paths", dropout_rate=dropout_rate),
    'Block_32_4x4x4_12': lambda dim, stride, dropout_rate: MyBlock_Sample(hidden_dim=dim, stride=stride, ssm_d_state=32, sample=2, scan_mode="tw_scanning_paths", dropout_rate=dropout_rate),
    'Block_32_8x8x8_12': lambda dim, stride, dropout_rate: MyBlock_Sample(hidden_dim=dim, stride=stride, ssm_d_state=32, sample=3, scan_mode="tw_scanning_paths", dropout_rate=dropout_rate),

    'Block_16_2x2x2_4': lambda dim, stride, dropout_rate: MyBlock(hidden_dim=dim, stride=stride, ssm_d_state=16, scan_mode="fo_scanning_paths", dropout_rate=dropout_rate),
    'Block_16_4x4x4_4': lambda dim, stride, dropout_rate: MyBlock_Sample(hidden_dim=dim, stride=stride, ssm_d_state=16, sample=2, scan_mode="fo_scanning_paths", dropout_rate=dropout_rate),
    'Block_16_8x8x8_4': lambda dim, stride, dropout_rate: MyBlock_Sample(hidden_dim=dim, stride=stride, ssm_d_state=16, sample=3, scan_mode="fo_scanning_paths", dropout_rate=dropout_rate),
    'Block_32_2x2x2_4': lambda dim, stride, dropout_rate: MyBlock(hidden_dim=dim, stride=stride, ssm_d_state=32, scan_mode="fo_scanning_paths", dropout_rate=dropout_rate),
    'Block_32_4x4x4_4': lambda dim, stride, dropout_rate: MyBlock_Sample(hidden_dim=dim, stride=stride, ssm_d_state=32, sample=2, scan_mode="fo_scanning_paths", dropout_rate=dropout_rate),
    'Block_32_8x8x8_4': lambda dim, stride, dropout_rate: MyBlock_Sample(hidden_dim=dim, stride=stride, ssm_d_state=32, sample=3, scan_mode="fo_scanning_paths", dropout_rate=dropout_rate),

    # 'avg_pool_3x3x3': lambda dim, stride: nn.AvgPool3d(3, stride=stride, padding=1, count_include_pad=False),
    # 'max_pool_3x3x3': lambda dim, stride: nn.MaxPool3d(3, stride=stride, padding=1),
    'Avg_pool_3x3x3': lambda dim, stride, dropout_rate: nn.AvgPool3d(3, stride=1, padding=1, count_include_pad=False),
    'Max_pool_3x3x3': lambda dim, stride, dropout_rate: nn.MaxPool3d(3, stride=1, padding=1),
    'SiLU': lambda dim, stride, dropout_rate: Act(hidden_dim=dim, stride=stride,  act_layer=nn.SiLU),
    'ReLU': lambda dim, stride, dropout_rate: Act(hidden_dim=dim, stride=stride,  act_layer=nn.ReLU),
    'ReLU6': lambda dim, stride, dropout_rate: Act(hidden_dim=dim, stride=stride,  act_layer=nn.ReLU6),
}


class ReLUConvBN(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(ReLUConvBN, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv3d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm3d(C_out, affine=affine)
        )

    def forward(self, x):
        return self.op(x)


class DilConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
        super(DilConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv3d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                      groups=C_in, bias=False),
            nn.Conv3d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm3d(C_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)


class SepConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv3d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
            nn.Conv3d(C_in, C_in, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm3d(C_in, affine=affine),
            nn.ReLU(inplace=False),
            nn.Conv3d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False),
            nn.Conv3d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm3d(C_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Zero(nn.Module):

    def __init__(self, stride):
        super(Zero, self).__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.)
        return x[:, :, ::self.stride, ::self.stride, ::self.stride].mul(0.)    # wms + ::self.stride


class ZeroLayer(nn.Module):

    def __init__(self, stride):
        super(ZeroLayer, self).__init__()
        self.stride = stride

    def forward(self, x):
        n, c, d, h, w = x.size()
        d //= self.stride
        h //= self.stride
        w //= self.stride
        device = x.get_device() if x.is_cuda else torch.device('cpu')
        # noinspection PyUnresolvedReferences
        padding = torch.zeros(n, c, d, h, w, device=device, requires_grad=False)
        # padding = torch.ones(n, c, d, h, w, device=device, requires_grad=False)
        return padding


class FactorizedReduce(nn.Module):

    def __init__(self, C_in, C_out, affine=True):
        super(FactorizedReduce, self).__init__()
        assert C_out % 2 == 0
        self.relu = nn.ReLU(inplace=False)
        self.conv_1 = nn.Conv3d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv_2 = nn.Conv3d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm3d(C_out, affine=affine)

    def forward(self, x):
        x = self.relu(x)
        out = torch.cat([self.conv_1(x), self.conv_2(x)], dim=1)  # wms - [:, :, 1:, 1:]
        out = self.bn(out)
        return out


# def get_same_padding(kernel_size):
#     if isinstance(kernel_size, tuple):
#         assert len(kernel_size) == 2, 'invalid kernel size: %s' % kernel_size
#         p1 = get_same_padding(kernel_size[0])
#         p2 = get_same_padding(kernel_size[1])
#         return p1, p2
#     assert isinstance(kernel_size, int), 'kernel size should be either `int` or `tuple`'
#     assert kernel_size % 2 > 0, 'kernel size should be odd number'
#     return kernel_size // 2


def count_conv_flop(layer, x):
    out_h = int(x.size()[2] / layer.stride[0])
    out_w = int(x.size()[3] / layer.stride[1])
    delta_ops = layer.in_channels * layer.out_channels * layer.kernel_size[0] * layer.kernel_size[1] * \
                out_h * out_w / layer.groups
    return delta_ops


class MBInvertedConvLayer(MyModule):

    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, expand_ratio=6, padding=0, mid_channels=None):
        super(MBInvertedConvLayer, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.kernel_size = kernel_size
        self.stride = stride
        self.expand_ratio = expand_ratio
        self.mid_channels = mid_channels

        if self.mid_channels is None:
            feature_dim = round(self.in_channels * self.expand_ratio)
        else:
            feature_dim = self.mid_channels

        if self.expand_ratio == 1:
            self.inverted_bottleneck = None
        else:
            # self.inverted_bottleneck = nn.Sequential(OrderedDict([
            #     ('conv', nn.Conv2d(self.in_channels, feature_dim, 1, 1, 0, bias=False)),
            #     ('bn', nn.BatchNorm2d(feature_dim)),
            #     ('act', nn.ReLU6(inplace=True)),
            # ]))
            self.inverted_bottleneck = nn.Sequential(OrderedDict([
                ('conv', nn.Conv3d(self.in_channels, feature_dim, 1, 1, 0, bias=False)),
                ('bn', nn.BatchNorm3d(feature_dim)),
                ('act', nn.ReLU6(inplace=True)),
            ]))

        # pad = get_same_padding(self.kernel_size)
        # self.depth_conv = nn.Sequential(OrderedDict([
        #     ('conv', nn.Conv2d(feature_dim, feature_dim, kernel_size, stride, padding, groups=feature_dim, bias=False)),
        #     ('bn', nn.BatchNorm2d(feature_dim)),
        #     ('act', nn.ReLU6(inplace=True)),
        # ]))
        self.depth_conv = nn.Sequential(OrderedDict([
            ('conv', nn.Conv3d(feature_dim, feature_dim, kernel_size, stride, padding, groups=feature_dim, bias=False)),
            ('bn', nn.BatchNorm3d(feature_dim)),
            ('act', nn.ReLU6(inplace=True)),
        ]))

        # self.point_linear = nn.Sequential(OrderedDict([
        #     ('conv', nn.Conv2d(feature_dim, out_channels, 1, 1, 0, bias=False)),
        #     ('bn', nn.BatchNorm2d(out_channels)),
        # ]))
        self.point_linear = nn.Sequential(OrderedDict([
            ('conv', nn.Conv3d(feature_dim, out_channels, 1, 1, 0, bias=False)),
            ('bn', nn.BatchNorm3d(out_channels)),
        ]))

    def forward(self, x):
        if self.inverted_bottleneck:
            x = self.inverted_bottleneck(x)
        x = self.depth_conv(x)
        x = self.point_linear(x)
        return x

    @property
    def module_str(self):
        return '%dx%d_MBConv%d' % (self.kernel_size, self.kernel_size, self.expand_ratio)

    @property
    def config(self):
        return {
            'name': MBInvertedConvLayer.__name__,
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'expand_ratio': self.expand_ratio,
            'mid_channels': self.mid_channels,
        }

    @staticmethod
    def build_from_config(config):
        return MBInvertedConvLayer(**config)

    def get_flops(self, x):
        if self.inverted_bottleneck:
            flop1 = count_conv_flop(self.inverted_bottleneck.conv, x)
            x = self.inverted_bottleneck(x)
        else:
            flop1 = 0

        flop2 = count_conv_flop(self.depth_conv.conv, x)
        x = self.depth_conv(x)

        flop3 = count_conv_flop(self.point_linear.conv, x)
        x = self.point_linear(x)

        return flop1 + flop2 + flop3, x

    @staticmethod
    def is_zero_layer():
        return False


# mamba
# =====================================================
# we have this class as linear and conv init differ from each other
# this function enable loading from both conv2d or linear
class Linear3d(nn.Linear):
    def forward(self, x: torch.Tensor):
        # B, C, H, W = x.shape
        # return F.conv2d(x, self.weight[:, :, None, None], self.bias)
        return F.conv3d(x, self.weight[:, :, None, None, None], self.bias).contiguous()


    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                              error_msgs):
        state_dict[prefix + "weight"] = state_dict[prefix + "weight"].view(self.weight.shape)
        return super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                                             error_msgs)

# class Linear3d(nn.Linear):
#     def forward(self, x: torch.Tensor):
#         x = x.permute(0, 2, 3, 4, 1)
#         x = F.linear(x, self.weight, self.bias)
#         x = x.permute(0, 4, 1, 2, 3)
#         return x


class LayerNorm2d(nn.LayerNorm):
    # @torch.compile
    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 3, 4, 1).contiguous()
        x = nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x




class PatchMerging3D(nn.Module):
    def __init__(self, dim, out_dim=-1, norm_layer=LayerNorm2d, channel_first=True):
        super().__init__()
        # Linear = Linear3d if channel_first else nn.Linear
        # self._patch_merging_pad = self._patch_merging_pad_channel_first if channel_first else self._patch_merging_pad_channel_last
        self._patch_merging_pad = self._make_downsample(dim, out_dim)

        # self.reduction = Linear(4 * dim, (2 * dim) if out_dim < 0 else out_dim, bias=False)
        # self.norm = norm_layer(4 * dim)
        # self.reduction = Linear(8 * dim, (2 * dim) if out_dim < 0 else out_dim, bias=False)

        # self.norm = norm_layer(8 * dim)
        # self.reduction = Linear(8 * dim, out_dim, bias=False)

    @staticmethod
    def _patch_merging_pad_channel_last(x: torch.Tensor):
        # H, W, _ = x.shape[-3:]
        # if (W % 2 != 0) or (H % 2 != 0):
        #     x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        # x0 = x[..., 0::2, 0::2, :]  # ... H/2 W/2 C
        # x1 = x[..., 1::2, 0::2, :]  # ... H/2 W/2 C
        # x2 = x[..., 0::2, 1::2, :]  # ... H/2 W/2 C
        # x3 = x[..., 1::2, 1::2, :]  # ... H/2 W/2 C
        # x = torch.cat([x0, x1, x2, x3], -1)  # ... H/2 W/2 4*C

        Length, H, W, _ = x.shape[-4:]
        if (Length % 2 != 0) or (W % 2 != 0) or (H % 2 != 0):
            x = F.pad(x, (0, 0, 0, W % 2, Length % 2, H % 2))
        x0 = x[..., 0::2, 0::2, 0::2, :]  # ... H/2 W/2 C
        x1 = x[..., 1::2, 0::2, 0::2, :]  # ... H/2 W/2 C
        x2 = x[..., 0::2, 1::2, 0::2, :]  # ... H/2 W/2 C
        x3 = x[..., 0::2, 0::2, 1::2, :]
        x4 = x[..., 1::2, 1::2, 0::2, :]  # ... H/2 W/2 C
        x5 = x[..., 0::2, 1::2, 1::2, :]
        x6 = x[..., 1::2, 0::2, 1::2, :]
        x7 = x[..., 1::2, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], -1)  # ... H/2 W/2 4*C
        return x

    @staticmethod
    # @torch.compile
    def _patch_merging_pad_channel_first(x: torch.Tensor):
        # H, W = x.shape[-2:]
        # if (W % 2 != 0) or (H % 2 != 0):
        #     x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        # x0 = x[..., 0::2, 0::2]  # ... H/2 W/2
        # x1 = x[..., 1::2, 0::2]  # ... H/2 W/2
        # x2 = x[..., 0::2, 1::2]  # ... H/2 W/2
        # x3 = x[..., 1::2, 1::2]  # ... H/2 W/2
        # x = torch.cat([x0, x1, x2, x3], 1)  # ... H/2 W/2 4*C

        # Length, H, W = x.shape[-3:]
        B, C, Length, H, W = x.shape
        if (Length % 2 != 0) or (W % 2 != 0) or (H % 2 != 0):
            x = F.pad(x, (0, 0, 0, W % 2, Length % 2, H % 2))
        x0 = x[..., 0::2, 0::2, 0::2]
        x1 = x[..., 1::2, 0::2, 0::2]
        x2 = x[..., 0::2, 1::2, 0::2]
        x3 = x[..., 0::2, 0::2, 1::2]
        x4 = x[..., 1::2, 1::2, 0::2]
        x5 = x[..., 0::2, 1::2, 1::2]
        x6 = x[..., 1::2, 0::2, 1::2]
        x7 = x[..., 1::2, 1::2, 1::2]
        x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], 1).contiguous()

        # 这个不是采样
        # # Reshaping the tensor to group patches
        # x = x.view(B, C, Length // 2, 2, H // 2, 2, W // 2, 2)
        # # Rearranging dimensions to get the desired patch combinations
        # x = x.permute(0, 1, 2, 4, 6, 3, 5, 7).contiguous()  # Rearrange to bring patches together
        # x = x.contiguous().view(B, C * 8, Length // 2, H // 2, W // 2).contiguous()  # Combining channels
        return x

    @staticmethod
    def _make_downsample(dim=96, out_dim=192, norm_layer=LayerNorm2d, channel_first=False):
        # if channel first, then Norm and Output are both channel_first
        # return nn.Sequential(
        #     # (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
        #     nn.Conv3d(dim, out_dim, kernel_size=2, stride=2),
        #     # (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
        #     nn.SiLU(),
        #     norm_layer(out_dim),
        # )
        return nn.Sequential(
            # nn.ReLU(inplace=False),
            # nn.Conv3d(dim, dim, kernel_size=2, stride=2, groups=dim, bias=False),
            # nn.Conv3d(dim, dim, kernel_size=1, padding=0, bias=False),
            # nn.BatchNorm3d(dim, affine=False),
            # nn.ReLU(inplace=False),
            nn.Conv3d(dim, dim, kernel_size=2, stride=2, groups=dim, bias=False),
            nn.Conv3d(dim, out_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=False),
            nn.BatchNorm3d(out_dim, affine=False),
        )

    @staticmethod
    def _make_downsample_v3(dim=96, out_dim=192, norm_layer=nn.LayerNorm, channel_first=False):
        # if channel first, then Norm and Output are both channel_first
        return nn.Sequential(
            # (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
            nn.Conv3d(dim, out_dim, kernel_size=3, stride=2, padding=1),
            # (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            norm_layer(out_dim),
        )

    def forward(self, x):
        x = self._patch_merging_pad(x)
        # x = self.norm(x)
        # x = self.reduction(x)

        return x


class Upsample3D(nn.Module):
    def __init__(self, dim, out_dim=-1, norm_layer=LayerNorm2d, channel_first=True):
        super().__init__()
        # self.dim = dim
        # Linear = Linear3d if channel_first else nn.Linear
        # self.upsample = self.upsample3d_channel_first if channel_first else self.upsample3d_channel_last
        self.upsample = self._make_upsample(dim, out_dim)
        # self.norm = norm_layer(dim)
        # self.reduction = Linear(dim, out_dim, bias=False)

    @staticmethod
    def upsample3d_channel_last(x: torch.Tensor):
        B, Length, H, W, _ = x.shape

        new_Length = Length * 2
        new_H = H * 2
        new_W = W * 2

        upsampled_x = torch.zeros((B, new_Length, new_H, new_W, _), device=x.device)

        upsampled_x[..., 0::2, 0::2, 0::2, :] = x
        upsampled_x[..., 1::2, 0::2, 0::2, :] = x
        upsampled_x[..., 0::2, 1::2, 0::2, :] = x
        upsampled_x[..., 1::2, 1::2, 0::2, :] = x
        upsampled_x[..., 0::2, 0::2, 1::2, :] = x
        upsampled_x[..., 1::2, 0::2, 1::2, :] = x
        upsampled_x[..., 0::2, 1::2, 1::2, :] = x
        upsampled_x[..., 1::2, 1::2, 1::2, :] = x

        return upsampled_x

    @staticmethod
    # @torch.compile
    def upsample3d_channel_first(x: torch.Tensor):
        B, C, Length, H, W = x.shape

        new_Length = Length * 2
        new_H = H * 2
        new_W = W * 2

        upsampled_x = torch.zeros((B, C, new_Length, new_H, new_W), device=x.device)

        # upsampled_x[..., 0::2, 0::2, 0::2] = x
        # upsampled_x[..., 1::2, 0::2, 0::2] = x
        # upsampled_x[..., 0::2, 1::2, 0::2] = x
        # upsampled_x[..., 1::2, 1::2, 0::2] = x
        # upsampled_x[..., 0::2, 0::2, 1::2] = x
        # upsampled_x[..., 1::2, 0::2, 1::2] = x
        # upsampled_x[..., 0::2, 1::2, 1::2] = x
        # upsampled_x[..., 1::2, 1::2, 1::2] = x
        upsampled_x = F.interpolate(x, size=(new_Length, new_H, new_W), mode='trilinear', align_corners=False)

        return upsampled_x.contiguous()

    @staticmethod
    def _make_upsample(dim=96, out_dim=192, norm_layer=LayerNorm2d, channel_first=False):
        # if channel first, then Norm and Output are both channel_first
        # return nn.Sequential(
        #     nn.ConvTranspose3d(dim, out_dim, kernel_size=2, stride=2),  # 使用转置卷积进行上采样
        #     nn.SiLU(),
        #     norm_layer(out_dim),
        # )

        return nn.Sequential(
            nn.ConvTranspose3d(dim, dim, kernel_size=2, stride=2, groups=dim),
            nn.Conv3d(dim, out_dim, kernel_size=1),
            nn.ReLU(inplace=False),
            nn.BatchNorm3d(out_dim, affine=False),
        )

    def forward(self, x):
        x = self.upsample(x)
        # x = self.norm(x)
        # x = self.reduction(x)

        return x


class Permute(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x: torch.Tensor):
        return x.permute(*self.args)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,
                 channels_first=True):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        Linear = Linear3d if channels_first else nn.Linear
        self.fc1 = Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class gMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,
                 channels_first=True):
        super().__init__()
        self.channel_first = channels_first
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        Linear = Linear3d if channels_first else nn.Linear
        self.fc1 = Linear(in_features, 2 * hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor):
        x = self.fc1(x)
        x, z = x.chunk(2, dim=(1 if self.channel_first else -1))
        x = self.fc2(x * self.act(z))
        x = self.drop(x)
        return x


# 继承softmax为了适应不同维度（c位置不同）
class SoftmaxSpatial(nn.Softmax):
    def forward(self, x: torch.Tensor):
        if self.dim == -1:
            B, C, H, W = x.shape
            return super().forward(x.view(B, C, -1)).view(B, C, H, W)
        elif self.dim == 1:
            B, H, W, C = x.shape
            return super().forward(x.view(B, -1, C)).view(B, H, W, C)
        else:
            raise NotImplementedError


# =====================================================
class mamba_init:
    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # S4D real initialization
        A = torch.arange(1, d_state + 1, dtype=torch.float32, device=device).view(1, -1).repeat(d_inner, 1).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = A_log[None].repeat(copies, 1, 1).contiguous()
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = D[None].repeat(copies, 1).contiguous()
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    @classmethod
    def init_dt_A_D(cls, d_state, dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, k_group=4):
        # dt proj ============================
        dt_projs = [
            cls.dt_init(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor)
            for _ in range(k_group)
        ]
        dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in dt_projs], dim=0))  # (K, inner, rank)
        dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in dt_projs], dim=0))  # (K, inner)
        del dt_projs

        # A, D =======================================
        A_logs = cls.A_log_init(d_state, d_inner, copies=k_group, merge=True)  # (K * D, N)
        Ds = cls.D_init(d_inner, copies=k_group, merge=True)  # (K * D)
        return A_logs, Ds, dt_projs_weight, dt_projs_bias


# support: v01-v05; v051d,v052d,v052dc;
# postfix: _onsigmoid,_onsoftmax,_ondwconv3,_onnone;_nozact,_noz;_oact;_no32;
# history support: v2,v3;v31d,v32d,v32dc;
class SS2Dv2:
    def __initv2__(
            self,
            # basic dims ===========
            d_model=96,
            d_state=16,
            ssm_ratio=2.0,
            dt_rank="auto",
            act_layer=nn.SiLU,
            # dwconv ===============
            d_conv=3,  # < 2 means no conv
            conv_bias=True,
            # ======================
            dropout=0.0,
            bias=False,
            # dt init ==============
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            initialize="v0",
            # ======================
            forward_type="v2",
            channel_first=True,
            scan_mode="tw_scanning_paths",
            # ======================
            **kwargs,
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        self.scan_mode = scan_mode
        # self.k_group = 4
        if self.scan_mode == "tw_scanning_paths":
            self.k_group = 12
        else:
            self.k_group = 4
        self.d_model = int(d_model)
        self.d_state = int(d_state)
        self.d_inner = int(ssm_ratio * d_model)
        self.dt_rank = int(math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank)
        self.channel_first = channel_first
        self.with_dconv = d_conv > 1
        self.Linear = Linear3d if channel_first else nn.Linear
        # 改前向函数
        self.forward = self.forwardv2

        # tags for forward_type ==============================
        checkpostfix = self.checkpostfix
        self.disable_force32, forward_type = checkpostfix("_no32", forward_type)
        self.oact, forward_type = checkpostfix("_oact", forward_type)
        self.disable_z, forward_type = checkpostfix("_noz", forward_type)
        self.disable_z_act, forward_type = checkpostfix("_nozact", forward_type)
        self.out_norm, forward_type = self.get_outnorm(forward_type, self.d_inner, channel_first)

        # # forward_type debug =======================================
        # FORWARD_TYPES = dict(
        #     v01=partial(self.forward_corev2, force_fp32=(not self.disable_force32), selective_scan_backend="mamba",
        #                 scan_force_torch=True),
        #     v02=partial(self.forward_corev2, force_fp32=(not self.disable_force32), selective_scan_backend="mamba"),
        #     v03=partial(self.forward_corev2, force_fp32=(not self.disable_force32), selective_scan_backend="oflex"),
        #     v04=partial(self.forward_corev2, force_fp32=False),  # selective_scan_backend="oflex", scan_mode="cross2d"
        #     v05=partial(self.forward_corev2, force_fp32=False, no_einsum=True),
        #     # selective_scan_backend="oflex", scan_mode="cross2d"
        #     # ===============================
        #     v051d=partial(self.forward_corev2, force_fp32=False, no_einsum=True, scan_mode="unidi"),
        #     v052d=partial(self.forward_corev2, force_fp32=False, no_einsum=True, scan_mode="bidi"),
        #     v052dc=partial(self.forward_corev2, force_fp32=False, no_einsum=True, scan_mode="cascade2d"),
        #     # ===============================
        #     v2=partial(self.forward_corev2, force_fp32=(not self.disable_force32), selective_scan_backend="core"),
        #     v3=partial(self.forward_corev2, force_fp32=False, selective_scan_backend="oflex"),
        # )
        # self.forward_core = FORWARD_TYPES.get(forward_type, None)
        self.forward_core = partial(self.forward_corev2, force_fp32=(not self.disable_force32), selective_scan_backend="mamba")
        # in proj =======================================
        d_proj = self.d_inner if self.disable_z else (self.d_inner * 2)
        # self.in_proj = self.Linear(self.d_model, d_proj, bias=bias)
        # self.act: nn.Module = act_layer()

        # conv =======================================
        # if self.with_dconv:
        #     # self.conv2d = nn.Conv2d(
        #     #     in_channels=self.d_inner,
        #     #     out_channels=self.d_inner,
        #     #     groups=self.d_inner,
        #     #     bias=conv_bias,
        #     #     kernel_size=d_conv,
        #     #     padding=(d_conv - 1) // 2,
        #     #     **factory_kwargs,
        #     # )
        #     self.conv3d = nn.Conv3d(
        #         in_channels=self.d_inner,
        #         out_channels=self.d_inner,
        #         groups=self.d_inner,
        #         bias=conv_bias,
        #         kernel_size=d_conv,
        #         padding=(d_conv - 1) // 2,
        #         **factory_kwargs,
        #     )

        # x proj ============================
        self.x_proj = [
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False)
            for _ in range(self.k_group)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K, N, inner)
        del self.x_proj

        # out proj =======================================
        self.out_act = nn.GELU() if self.oact else nn.Identity()
        # self.out_proj = self.Linear(self.d_inner, self.d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        # if initialize in ["v0"]:
        #     self.A_logs, self.Ds, self.dt_projs_weight, self.dt_projs_bias = mamba_init.init_dt_A_D(
        #         self.d_state, self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
        #         k_group=self.k_group,
        #     )
        # elif initialize in ["v1"]:
        #     # simple init dt_projs, A_logs, Ds
        #     self.Ds = nn.Parameter(torch.ones((self.k_group * self.d_inner)))
        #     self.A_logs = nn.Parameter(torch.randn(
        #         (self.k_group * self.d_inner, self.d_state)))  # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
        #     self.dt_projs_weight = nn.Parameter(
        #         0.1 * torch.randn((self.k_group, self.d_inner, self.dt_rank)))  # 0.1 is added in 0430
        #     self.dt_projs_bias = nn.Parameter(0.1 * torch.randn((self.k_group, self.d_inner)))  # 0.1 is added in 0430
        # elif initialize in ["v2"]:
        #     # simple init dt_projs, A_logs, Ds
        #     self.Ds = nn.Parameter(torch.ones((self.k_group * self.d_inner)))
        #     self.A_logs = nn.Parameter(torch.zeros(
        #         (self.k_group * self.d_inner, self.d_state)))  # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
        #     self.dt_projs_weight = nn.Parameter(0.1 * torch.rand((self.k_group, self.d_inner, self.dt_rank)))
        #     self.dt_projs_bias = nn.Parameter(0.1 * torch.rand((self.k_group, self.d_inner)))

        self.A_logs, self.Ds, self.dt_projs_weight, self.dt_projs_bias = mamba_init.init_dt_A_D(
            self.d_state, self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
            k_group=self.k_group,
        )


    def forward_corev2(
            self,
            x: torch.Tensor = None,
            # ==============================
            force_fp32=False,  # True: input fp32
            # ==============================
            ssoflex=True,  # True: input 16 or 32 output 32 False: output dtype as input
            no_einsum=False,  # replace einsum with linear or conv1d to raise throughput
            # ==============================
            selective_scan_backend=None,
            # ==============================
            # scan_mode=scan_mode,
            scan_force_torch=False,
            # ==============================
            **kwargs,
    ):

        assert self.scan_mode in ["unidi", "bidi", "tw_scanning_paths", "fo_scanning_paths"]
        assert selective_scan_backend in [None, "oflex", "core", "mamba", "torch"]
        delta_softplus = True
        out_norm = self.out_norm
        channel_first = self.channel_first
        to_fp32 = lambda *args: (_a.to(torch.float32) for _a in args)

        # B, D, H, W = x.shape
        B, D, Length, H, W = x.shape
        N = self.d_state
        K, D, R = self.k_group, self.d_inner, self.dt_rank
        # L = H * W
        L = Length * H * W
        _scan_mode = dict(tw_scanning_paths=0, unidi=1, bidi=2, fo_scanning_paths=3)[self.scan_mode]   # 0

        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True):
            return selective_scan_fn(u, delta, A, B, C, D, delta_bias, delta_softplus, ssoflex,
                                     backend=selective_scan_backend)

        if _scan_mode == 100:
            pass
        else:
            x_proj_bias = getattr(self, "x_proj_bias", None)
            xs = cross_scan_fn(x, in_channel_first=True, out_channel_first=True, scans=_scan_mode,
                               force_torch=scan_force_torch)
            if no_einsum:
                pass
                # x_dbl = F.conv1d(xs.view(B, -1, L), self.x_proj_weight.view(-1, D, 1),
                #                  bias=(x_proj_bias.view(-1) if x_proj_bias is not None else None), groups=K)
                # dts, Bs, Cs = torch.split(x_dbl.view(B, K, -1, L), [R, N, N], dim=2)
                # dts = F.conv1d(dts.contiguous().view(B, -1, L), self.dt_projs_weight.view(K * D, -1, 1), groups=K)
            else:
                x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
                if x_proj_bias is not None:
                    x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
                dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
                dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)

            xs = xs.view(B, -1, L)
            dts = dts.contiguous().view(B, -1, L)
            As = -self.A_logs.to(torch.float).exp()  # (k * c, d_state)
            Ds = self.Ds.to(torch.float)  # (K * c)
            Bs = Bs.contiguous().view(B, K, N, L)
            Cs = Cs.contiguous().view(B, K, N, L)
            delta_bias = self.dt_projs_bias.view(-1).to(torch.float)

            if force_fp32:   #meijing
                xs, dts, Bs, Cs = to_fp32(xs, dts, Bs, Cs)

            # ys: torch.Tensor = selective_scan(
            #     xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus
            # ).view(B, K, -1, H, W)
            ys: torch.Tensor = selective_scan(
                xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus
            ).view(B, K, -1, Length, H, W)

            y: torch.Tensor = cross_merge_fn(ys, in_channel_first=True, out_channel_first=True, scans=_scan_mode,
                                             force_torch=scan_force_torch)

            if getattr(self, "__DEBUG__", False):#meijing
                setattr(self, "__data__", dict(
                    A_logs=self.A_logs, Bs=Bs, Cs=Cs, Ds=Ds,
                    us=xs, dts=dts, delta_bias=delta_bias,
                    ys=ys, y=y, H=H, W=W,
                ))

        # y = y.view(B, -1, H, W)
        y = y.view(B, -1, Length, H, W)
        if not channel_first:
            # y = y.view(B, -1, H * W).transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1)  # (B, L, C)
            y = y.view(B, -1, Length * H * W).transpose(dim0=1, dim1=2).contiguous().view(B, Length, H, W,
                                                                                          -1)  # (B, L, C)
        y = out_norm(y)

        return y.to(x.dtype)

    def forwardv2(self, x: torch.Tensor, **kwargs):
        # x = self.in_proj(x)
        # if not self.disable_z:
        #     x, z = x.chunk(2, dim=(1 if self.channel_first else -1))  # (b, h, w, d)
        #     if not self.disable_z_act:
        #         z = self.act(z)
        # # if not self.channel_first:
        # #     # x = x.permute(0, 3, 1, 2).contiguous()
        # #     x = x.permute(0, 4, 1, 2, 3).contiguous()
        # if self.with_dconv:
        #     x = self.conv3d(x)  # (b, d, h, w)
        # x = self.act(x)
        y = self.forward_core(x)
        # y = self.out_act(y)
        out = self.dropout(self.out_act(y))
        # if not self.disable_z:
        #     y = y * z
        # out = self.dropout(self.out_proj(y))
        return out

    @staticmethod
    def get_outnorm(forward_type="", d_inner=192, channel_first=True):
        def checkpostfix(tag, value):
            ret = value[-len(tag):] == tag
            if ret:
                value = value[:-len(tag)]
            return ret, value

        LayerNorm = LayerNorm2d if channel_first else nn.LayerNorm

        out_norm_none, forward_type = checkpostfix("_onnone", forward_type)
        out_norm_dwconv3, forward_type = checkpostfix("_ondwconv3", forward_type)
        out_norm_cnorm, forward_type = checkpostfix("_oncnorm", forward_type)
        out_norm_softmax, forward_type = checkpostfix("_onsoftmax", forward_type)
        out_norm_sigmoid, forward_type = checkpostfix("_onsigmoid", forward_type)

        out_norm = nn.Identity()
        if out_norm_none:
            out_norm = nn.Identity()
        elif out_norm_cnorm:
            out_norm = nn.Sequential(
                LayerNorm(d_inner),
                (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
                nn.Conv2d(d_inner, d_inner, kernel_size=3, padding=1, groups=d_inner, bias=False),
                (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            )
        elif out_norm_dwconv3:
            out_norm = nn.Sequential(
                (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
                nn.Conv2d(d_inner, d_inner, kernel_size=3, padding=1, groups=d_inner, bias=False),
                (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            )
        elif out_norm_softmax:
            out_norm = SoftmaxSpatial(dim=(-1 if channel_first else 1))
        elif out_norm_sigmoid:
            out_norm = nn.Sigmoid()
        else:
            out_norm = LayerNorm(d_inner)

        return out_norm, forward_type

    @staticmethod
    def checkpostfix(tag, value):
        ret = value[-len(tag):] == tag
        if ret:
            value = value[:-len(tag)]
        return ret, value


class SS2D(nn.Module, SS2Dv2):
    def __init__(
            self,
            # basic dims ===========
            d_model=96,
            d_state=16,
            ssm_ratio=2.0,
            dt_rank="auto",
            act_layer=nn.SiLU,
            # dwconv ===============
            d_conv=3,  # < 2 means no conv
            conv_bias=True,
            # ======================
            dropout=0.0,
            bias=False,
            # dt init ==============
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            initialize="v0",
            # ======================
            forward_type="v2",
            channel_first=True,
            scan_mode="tw_scanning_paths",
            # ======================
            **kwargs,
    ):
        nn.Module.__init__(self)
        kwargs.update(
            d_model=d_model, d_state=d_state, ssm_ratio=ssm_ratio, dt_rank=dt_rank,
            act_layer=act_layer, d_conv=d_conv, conv_bias=conv_bias, dropout=dropout, bias=bias,
            dt_min=dt_min, dt_max=dt_max, dt_init=dt_init, dt_scale=dt_scale, dt_init_floor=dt_init_floor,
            initialize=initialize, forward_type=forward_type, channel_first=channel_first, scan_mode=scan_mode,
        )
        # if forward_type in ["v0", "v0seq"]:
        #     self.__initv0__(seq=("seq" in forward_type), **kwargs)
        # elif forward_type.startswith("xv"):
        #     self.__initxv__(**kwargs)
        # elif forward_type.startswith("m"):
        #     self.__initm0__(**kwargs)
        # else:
        #     self.__initv2__(**kwargs)
        self.__initv2__(**kwargs)


# =====================================================
class MyBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 96,
            stride=1,
            ssm_d_state: int = 16,
            scan_mode="tw_scanning_paths",
            **kwargs,
    ):
        super().__init__()
        self.scan_mode = scan_mode
        # self.mlp_ratio = 4.0
        self.mlp_ratio = 0
        # self.ssm_ratio = 2.0
        self.ssm_ratio = 1.0
        self.ssm_branch = self.ssm_ratio > 0
        self.mlp_branch = self.mlp_ratio > 0
        self.use_checkpoint = False
        self.post_norm = False
        self.gmlp = False
        self.mlp_drop_rate = 0.0
        self.mlp_act_layer = nn.GELU
        self.ssm_act_layer = nn.SiLU
        self.norm_layer = LayerNorm2d
        self.ssm_conv_bias = True
        self.ssm_init = "v0"
        self.ssm_dt_rank = "auto"
        self.ssm_drop_rate = 0.0
        self.ssm_conv = 3
        self.channel_first = True
        self.drop_path = 0.0
        # self.drop_path = 0.5
        # self.drop_path = 0.2
        # self.drop_path = 0.7
        self.stride = stride
        # self.C_prev = hidden_dim // 2
        self.C_prev = hidden_dim
        # if self.stride == 2:
        #     self.Downsample = PatchMerging3D(
        #         self.C_prev,
        #         hidden_dim,
        #         channel_first=True)

        if self.ssm_branch:
            # self.norm = self.norm_layer(hidden_dim)
            self.norm = LayerNorm2d(hidden_dim)
            self.op = SS2D(
                d_model=hidden_dim,
                d_state=ssm_d_state,
                ssm_ratio=self.ssm_ratio,
                dt_rank=self.ssm_dt_rank,
                act_layer=self.ssm_act_layer,
                # ==========================
                d_conv=self.ssm_conv,
                conv_bias=self.ssm_conv_bias,
                # ==========================
                dropout=self.ssm_drop_rate,
                # bias=False,
                # ==========================
                # dt_min=0.001,
                # dt_max=0.1,
                # dt_init="random",
                # dt_scale="random",
                # dt_init_floor=1e-4,
                initialize=self.ssm_init,
                # ==========================
                forward_type='v02',
                channel_first=self.channel_first,
                scan_mode=self.scan_mode,
            )

        self.drop_path = DropPath(self.drop_path)

        # if self.mlp_branch:
        #     _MLP = Mlp if not self.gmlp else gMlp
        #     # self.norm2 = self.norm_layer(hidden_dim)
        #     self.norm2 = LayerNorm2d(hidden_dim)
        #     mlp_hidden_dim = int(hidden_dim * self.mlp_ratio)
        #     self.mlp = _MLP(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=self.mlp_act_layer,
        #                     drop=self.mlp_drop_rate, channels_first=True)

    def forward(self, x: torch.Tensor):
        # x = input
        # if self.stride == 2:
        #     x = self.Downsample(x)
        # if self.ssm_branch:
        #     if self.post_norm:
        #         x = x + self.drop_path(self.norm(self.op(x)))
        #     else:
        #         x = x + self.drop_path(self.op(self.norm(x)))
        x = x + self.drop_path(self.op(self.norm(x)))
        # if self.mlp_branch:
        #     if self.post_norm:
        #         x = x + self.drop_path(self.norm2(self.mlp(x)))  # FFN
        #     else:
        #         x = x + self.drop_path(self.mlp(self.norm2(x)))  # FFN
        return x

    # def forward(self, input: torch.Tensor):
    #     # if self.use_checkpoint:
    #     #     return checkpoint.checkpoint(self._forward, input)
    #     # else:
    #     #     return self._forward(input)
    #     return self._forward(input)


class MyBlock_Sample(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 96,
            stride=1,
            ssm_d_state: int = 16,
            sample=2,
            act_layer=nn.SiLU,
            scan_mode="tw_scanning_paths",
            **kwargs,
    ):
        super().__init__()
        self.scan_mode = scan_mode
        self.sample = sample
        self.hidden_dim = hidden_dim
        self.m = 2 * self.hidden_dim
        self.stride = stride
        self.C_prev = hidden_dim
        # self.act: nn.Module = act_layer()
        # self.Downsample2 = PatchMerging3D(self.hidden_dim, self.hidden_dim, channel_first=True)
        # if self.stride == 2:
        #     # self.Downsample = PatchMerging3D(
        #     #     self.hidden_dim,
        #     #     self.hidden_dim,
        #     #     channel_first=True)
        #     self.Downsample2 = nn.Sequential(
        #         PatchMerging3D(self.hidden_dim, self.hidden_dim, channel_first=True),
        #         PatchMerging3D(self.hidden_dim, self.hidden_dim, channel_first=True),
        #     )
        # else:
        #     self.Downsample2 = PatchMerging3D(self.hidden_dim, self.hidden_dim, channel_first=True)

        self.Downsample2 = PatchMerging3D(self.hidden_dim, self.hidden_dim, channel_first=True)
        # self.Downsample2 = PatchMerging3D(self.hidden_dim, self.m, channel_first=True)
        # if self.sample == 3:
        #     self.Downsample3 = PatchMerging3D(self.hidden_dim, self.hidden_dim, channel_first=True)

        self.Upsample2 = Upsample3D(self.hidden_dim, self.hidden_dim, channel_first=True)
        # self.Upsample2 = Upsample3D(self.m, self.hidden_dim, channel_first=True)
        # if self.sample == 3:
        #     self.Upsample3 = Upsample3D(self.hidden_dim, self.hidden_dim, channel_first=True)

        self.MyBlock = MyBlock(hidden_dim=self.hidden_dim, stride=1, ssm_d_state=ssm_d_state, scan_mode=self.scan_mode)
        # self.MyBlock = MyBlock(hidden_dim=self.m, stride=1, ssm_d_state=ssm_d_state, scan_mode=self.scan_mode)

    def forward(self, x: torch.Tensor):
        # if self.stride == 2:
        #     input = self.Downsample(x)

        x = self.Downsample2(x)
        # if self.sample == 3:
        #     x = self.Downsample3(x)
        # x = self.act(x)
        x = self.MyBlock(x)

        x = self.Upsample2(x)
        # if self.sample == 3:
        #     x = self.Upsample3(x)
        return x


class Act(nn.Module):
    def __init__(
            self,
            hidden_dim,
            stride,
            act_layer,
            **kwargs,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.stride = stride
        self.C_prev = hidden_dim
        self.act: nn.Module = act_layer()
        # self.norm_layer = LayerNorm2d(hidden_dim)
        # if self.stride == 2:
        #     self.Downsample = PatchMerging3D(
        #         self.C_prev,
        #         hidden_dim,
        #         channel_first=True)

    def forward(self, x: torch.Tensor, **kwargs):
        # if self.stride == 2:
        #     x = self.Downsample(x)
        x = self.act(x)
        # x = self.norm_layer(x)
        return x



