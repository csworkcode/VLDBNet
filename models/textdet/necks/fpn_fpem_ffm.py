# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule, ModuleList, Sequential

from mmocr.registry import MODELS


class FPEM(BaseModule):
    """FPN-like feature fusion module in PANet.

    Args:
        in_channels (int): Number of input channels.
        init_cfg (dict or list[dict], optional): Initialization configs.
    """

    def __init__(self,
                 in_channels: int = 256,
                 init_cfg: Optional[Union[Dict, List[Dict]]] = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.up_add1 = SeparableConv2d(in_channels, in_channels, 1)
        self.up_add2 = SeparableConv2d(in_channels, in_channels, 1)
        self.up_add3 = SeparableConv2d(in_channels, in_channels, 1)
        self.down_add1 = SeparableConv2d(in_channels, in_channels, 2)
        self.down_add2 = SeparableConv2d(in_channels, in_channels, 2)
        self.down_add3 = SeparableConv2d(in_channels, in_channels, 2)

    def forward(self, c2: torch.Tensor, c3: torch.Tensor, c4: torch.Tensor,
                c5: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            c2, c3, c4, c5 (Tensor): Each has the shape of
                :math:`(N, C_i, H_i, W_i)`.

        Returns:
            list[Tensor]: A list of 4 tensors of the same shape as input.
        """
        # upsample
        c4 = self.up_add1(self._upsample_add(c5, c4))  # c4 shape
        c3 = self.up_add2(self._upsample_add(c4, c3))
        c2 = self.up_add3(self._upsample_add(c3, c2))

        # downsample
        c3 = self.down_add1(self._upsample_add(c3, c2))
        c4 = self.down_add2(self._upsample_add(c4, c3))
        c5 = self.down_add3(self._upsample_add(c5, c4))  # c4 / 2
        return c2, c3, c4, c5

    def _upsample_add(self, x, y):
        return F.interpolate(x, size=y.size()[2:]) + y


class SeparableConv2d(BaseModule):
    """Implementation of separable convolution, which is consisted of depthwise
    convolution and pointwise convolution.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        stride (int): Stride of the depthwise convolution.
        init_cfg (dict or list[dict], optional): Initialization configs.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride: int = 1,
                 init_cfg: Optional[Union[Dict, List[Dict]]] = None) -> None:
        super().__init__(init_cfg=init_cfg)

        self.depthwise_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            padding=1,
            stride=stride,
            groups=in_channels)
        self.pointwise_conv = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
        """
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

@MODELS.register_module()
class FPNFF(BaseModule):
    """FPN-like fusion module in Real-time Scene Text Detection with
    Differentiable Binarization.

    This was partially adapted from https://github.com/MhLiao/DB and
    https://github.com/WenmuZhou/DBNet.pytorch.

    Args:
        in_channels (list[int]): A list of numbers of input channels.
        lateral_channels (int): Number of channels for lateral layers.
        out_channels (int): Number of output channels.
        bias_on_lateral (bool): Whether to use bias on lateral convolutional
            layers.
        bn_re_on_lateral (bool): Whether to use BatchNorm and ReLU
            on lateral convolutional layers.
        bias_on_smooth (bool): Whether to use bias on smoothing layer.
        bn_re_on_smooth (bool): Whether to use BatchNorm and ReLU on smoothing
            layer.
        asf_cfg (dict, optional): Adaptive Scale Fusion module configs. The
            attention_type can be 'ScaleChannelSpatial'.
        conv_after_concat (bool): Whether to add a convolution layer after
            the concatenation of predictions.
        init_cfg (dict or list[dict], optional): Initialization configs.
    """

    def __init__(
        self,
        in_channels: List[int],
        lateral_channels: int = 256,
        out_channels: int = 64,
        fpem_repeat: int = 2,
        align_corners: bool = False,
        bias_on_lateral: bool = False,
        bn_re_on_lateral: bool = False,
        bias_on_smooth: bool = False,
        bn_re_on_smooth: bool = False,
        asf_cfg: Optional[Dict] = None,
        conv_after_concat: bool = False,
        init_cfg: Optional[Union[Dict, List[Dict]]] = [
            dict(type='Kaiming', layer='Conv'),
            dict(type='Constant', layer='BatchNorm', val=1., bias=1e-4)
        ]
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.lateral_channels = lateral_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.bn_re_on_lateral = bn_re_on_lateral
        self.bn_re_on_smooth = bn_re_on_smooth
        self.asf_cfg = asf_cfg
        self.conv_after_concat = conv_after_concat
        self.lateral_convs = ModuleList()
        self.smooth_convs = ModuleList()
        self.num_outs = self.num_ins

        for i in range(self.num_ins):
            norm_cfg = None
            act_cfg = None
            if self.bn_re_on_lateral:
                norm_cfg = dict(type='BN')
                act_cfg = dict(type='ReLU')
            l_conv = ConvModule(
                in_channels[i],
                lateral_channels,
                1,
                bias=bias_on_lateral,
                conv_cfg=None,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)
            norm_cfg = None
            act_cfg = None
            if self.bn_re_on_smooth:
                norm_cfg = dict(type='BN')
                act_cfg = dict(type='ReLU')

            smooth_conv = ConvModule(
                lateral_channels,
                out_channels,
                3,
                bias=bias_on_smooth,
                padding=1,
                conv_cfg=None,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)

            self.lateral_convs.append(l_conv)
            self.smooth_convs.append(smooth_conv)

        if self.asf_cfg is not None:
            self.asf_conv = ConvModule(
                out_channels * self.num_outs,
                out_channels * self.num_outs,
                3,
                padding=1,
                conv_cfg=None,
                norm_cfg=None,
                act_cfg=None,
                inplace=False)
            if self.asf_cfg['attention_type'] == 'ScaleChannelSpatial':
                self.asf_attn = ScaleChannelSpatialAttention(
                    self.out_channels * self.num_outs,
                    (self.out_channels * self.num_outs) // 4, self.num_outs)
            else:
                raise NotImplementedError

        if self.conv_after_concat:
            norm_cfg = dict(type='BN')
            act_cfg = dict(type='ReLU')
            self.out_conv = ConvModule(
                out_channels * self.num_outs,
                out_channels * self.num_outs,
                3,
                padding=1,
                conv_cfg=None,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)

        self.fpems = ModuleList()
        for _ in range(fpem_repeat):
            self.fpems.append(FPEM(lateral_channels))

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            inputs (list[Tensor]): Each tensor has the shape of
                :math:`(N, C_i, H_i, W_i)`. It usually expects 4 tensors
                (C2-C5 features) from ResNet.

        Returns:
            Tensor: A tensor of shape :math:`(N, C_{out}, H_0, W_0)` where
            :math:`C_{out}` is ``out_channels``.
        """
        assert len(inputs) == len(self.in_channels)
        # build laterals  调整通道为256
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        used_backbone_levels = len(laterals)

        c2, c3, c4, c5 = laterals
        # FPEM
        for i, fpem in enumerate(self.fpems):
            c2, c3, c4, c5 = fpem(c2, c3, c4, c5)
            if i == 0:
                c2_ffm = c2
                c3_ffm = c3
                c4_ffm = c4
                c5_ffm = c5
            else:
                c2_ffm = c2_ffm + c2
                c3_ffm = c3_ffm + c3
                c4_ffm = c4_ffm + c4
                c5_ffm = c5_ffm + c5

        # # FFM
        # c5 = F.interpolate(
        #     c5_ffm,
        #     c2_ffm.size()[-2:],
        #     mode='bilinear',
        #     align_corners=self.align_corners)
        # c4 = F.interpolate(
        #     c4_ffm,
        #     c2_ffm.size()[-2:],
        #     mode='bilinear',
        #     align_corners=self.align_corners)
        # c3 = F.interpolate(
        #     c3_ffm,
        #     c2_ffm.size()[-2:],
        #     mode='bilinear',
        #     align_corners=self.align_corners)

        laterals = [c2_ffm, c3_ffm, c4_ffm, c5_ffm]
        #
        # # build top-down path 上采样并累加
        # for i in range(used_backbone_levels - 1, 0, -1):
        #     prev_shape = laterals[i - 1].shape[2:]
        #     laterals[i - 1] = laterals[i - 1] + F.interpolate(
        #         laterals[i], size=prev_shape, mode='nearest')

        # build outputs
        # part 1: from original levels 调整通道为64
        outs = [
            self.smooth_convs[i](laterals[i])
            for i in range(used_backbone_levels)
        ]
        # 调整尺寸为 256/4=160
        for i, out in enumerate(outs):
            outs[i] = F.interpolate(
                outs[i], size=outs[0].shape[2:], mode='nearest')
        # cat
        out = torch.cat(outs, dim=1)
        if self.asf_cfg is not None:
            asf_feature = self.asf_conv(out)
            attention = self.asf_attn(asf_feature)
            enhanced_feature = []
            for i, out in enumerate(outs):
                enhanced_feature.append(attention[:, i:i + 1] * outs[i])
            out = torch.cat(enhanced_feature, dim=1)

        if self.conv_after_concat:
            out = self.out_conv(out)

        return out


class ScaleChannelSpatialAttention(BaseModule):
    """Spatial Attention module in Real-Time Scene Text Detection with
    Differentiable Binarization and Adaptive Scale Fusion.

    This was partially adapted from https://github.com/MhLiao/DB

    Args:
        in_channels (int): A numbers of input channels.
        c_wise_channels (int): Number of channel-wise attention channels.
        out_channels (int): Number of output channels.
        init_cfg (dict or list[dict], optional): Initialization configs.
    """

    def __init__(
        self,
        in_channels: int,
        c_wise_channels: int,
        out_channels: int,
        init_cfg: Optional[Union[Dict, List[Dict]]] = [
            dict(type='Kaiming', layer='Conv', bias=0)
        ]
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # Channel Wise
        self.channel_wise = Sequential(
            ConvModule(
                in_channels,
                c_wise_channels,
                1,
                bias=False,
                conv_cfg=None,
                norm_cfg=None,
                act_cfg=dict(type='ReLU'),
                inplace=False),
            ConvModule(
                c_wise_channels,
                in_channels,
                1,
                bias=False,
                conv_cfg=None,
                norm_cfg=None,
                act_cfg=dict(type='Sigmoid'),
                inplace=False))
        # Spatial Wise
        self.spatial_wise = Sequential(
            ConvModule(
                1,
                1,
                3,
                padding=1,
                bias=False,
                conv_cfg=None,
                norm_cfg=None,
                act_cfg=dict(type='ReLU'),
                inplace=False),
            ConvModule(
                1,
                1,
                1,
                bias=False,
                conv_cfg=None,
                norm_cfg=None,
                act_cfg=dict(type='Sigmoid'),
                inplace=False))
        # Attention Wise
        self.attention_wise = ConvModule(
            in_channels,
            out_channels,
            1,
            bias=False,
            conv_cfg=None,
            norm_cfg=None,
            act_cfg=dict(type='Sigmoid'),
            inplace=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs (Tensor): A concat FPN feature tensor that has the shape of
                :math:`(N, C, H, W)`.

        Returns:
            Tensor: An attention map of shape :math:`(N, C_{out}, H, W)`
            where :math:`C_{out}` is ``out_channels``.
        """
        out = self.avg_pool(inputs)
        out = self.channel_wise(out)
        out = out + inputs
        inputs = torch.mean(out, dim=1, keepdim=True)
        out = self.spatial_wise(inputs) + out
        out = self.attention_wise(out)

        return out
