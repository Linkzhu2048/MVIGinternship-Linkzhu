# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""构建 ResNet 图像骨干与冻结批归一化，返回特征和位置编码。"""
import torch
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import List
from torch import Tensor as NestedTensor
from .position_encoding import build_position_encoding

class FrozenBatchNorm2d(torch.nn.Module):
    """使用固定均值、方差和仿射参数的批归一化层。"""

    def __init__(self, n):
        """初始化使用固定均值、方差和仿射参数的批归一化层。

        输入：n。
        输出：配置好的模块实例。
        """
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer('weight', torch.ones(n))
        self.register_buffer('bias', torch.zeros(n))
        self.register_buffer('running_mean', torch.zeros(n))
        self.register_buffer('running_var', torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        """兼容不含批次计数器的冻结归一化参数加载。

        输入：state_dict、prefix、local_metadata、strict、missing_keys、unexpected_keys、error_msgs。
        输出：无；更新模块参数及加载错误列表。
        """
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]
        super(FrozenBatchNorm2d, self)._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        """用固定统计量归一化输入特征。

        输入：x。
        输出：与输入形状相同的特征。
        """
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-05
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias

class BackboneBase(nn.Module):
    """从图像骨干提取指定层的特征。"""

    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool):
        """初始化从图像骨干提取指定层的特征。

        输入：backbone、train_backbone、num_channels、return_interm_layers。
        输出：配置好的模块实例。
        """
        super().__init__()
        if return_interm_layers:
            return_layers = {'layer1': '0', 'layer2': '1', 'layer3': '2', 'layer4': '3'}
        else:
            return_layers = {'layer4': '0'}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor):
        """提取图像骨干的指定层输出。

        输入：tensor。
        输出：按层名称组织的特征字典。
        """
        xs = self.body(tensor)
        return xs

class Backbone(BackboneBase):
    """使用冻结批归一化的 ResNet 骨干。"""

    def __init__(self, name: str, train_backbone: bool, return_interm_layers: bool, dilation: bool, pretrained: bool=True):
        """初始化使用冻结批归一化的 ResNet 骨干。

        输入：name、train_backbone、return_interm_layers、dilation、pretrained。
        输出：配置好的模块实例。
        """
        backbone = getattr(torchvision.models, name)(replace_stride_with_dilation=[False, False, dilation], weights=torchvision.models.get_model_weights(name).DEFAULT if pretrained else None, norm_layer=FrozenBatchNorm2d)
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)

class Joiner(nn.Sequential):
    """组合图像骨干与位置编码。"""

    def __init__(self, backbone, position_embedding):
        """初始化组合图像骨干与位置编码。

        输入：backbone、position_embedding。
        输出：配置好的模块实例。
        """
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        """提取图像特征并计算对应位置编码。

        输入：tensor_list。
        输出：特征列表和位置编码列表。
        """
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            pos.append(self[1](x).to(x.dtype))
        return (out, pos)

def build_backbone(args):
    """构建配置指定的 ResNet 和正弦位置编码。

    输入：args。
    输出：带通道数信息的图像特征模块。
    """
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    return_interm_layers = args.masks
    backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation, args.pretrained_backbone)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model
