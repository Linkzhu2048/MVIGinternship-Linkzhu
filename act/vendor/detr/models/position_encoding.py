# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""为图像特征生成 ACT 使用的二维正弦位置编码。"""
import math
import torch
from torch import nn

class PositionEmbeddingSine(nn.Module):
    """二维图像特征的正弦和余弦位置编码。"""

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        """初始化二维图像特征的正弦和余弦位置编码。

        输入：num_pos_feats、temperature、normalize、scale。
        输出：配置好的模块实例。
        """
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError('normalize should be True if scale is passed')
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor):
        """按特征图高宽计算二维正弦位置编码。

        输入：tensor。
        输出：与图像空间尺寸对应的位置编码张量。
        """
        x = tensor
        not_mask = torch.ones_like(x[0, [0]])
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-06
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

def build_position_encoding(args):
    """构建二维正弦位置编码，仅支持 ACT 使用的分支。

    输入：args。
    输出：正弦位置编码模块。
    """
    if args.position_embedding not in ('v2', 'sine'):
        raise ValueError('仅支持正弦位置编码')
    return PositionEmbeddingSine(args.hidden_dim // 2, normalize=True)
