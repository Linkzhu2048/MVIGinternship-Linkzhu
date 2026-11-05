"""封装 ACT 网络、图像归一化、原版 L1 与 KL 损失及 AdamW 优化器。"""
from types import SimpleNamespace
import torch
from torch import nn
from i2rt_act_vendor.detr.models.detr_vae import build

def masked_l1(prediction, actions, is_pad):
    """对非填充动作计算原版 L1，分母仍为完整动作张量元素数。

    输入：prediction、actions、is_pad。
    输出：平均损失、有效误差总和和有效元素数。
    """
    valid = (~is_pad).unsqueeze(-1).expand_as(actions)
    count = valid.sum()
    if count.item() == 0:
        raise ValueError('Action chunk contains no valid targets')
    total = (prediction - actions).abs().masked_fill(~valid, 0).sum()
    return (total / prediction.numel(), total, count)

class ACTPolicy(nn.Module):
    """对官方 ACT 网络添加图像归一化、损失计算和推理接口。"""

    def __init__(self, config, pretrained=None):
        """构建 14 维 ACT 并注册图像归一化缓冲区。

        输入：config、pretrained。
        输出：策略网络实例。
        """
        super().__init__()
        if config['state_dim'] != 14 or config['action_dim'] != 14:
            raise ValueError('Expected 14D I2RT')
        args = SimpleNamespace(
            **{k: v for k, v in config.items()
               if k not in ('state_dim', 'action_dim', 'chunk_size', 'kl_weight')},
            num_queries=config['chunk_size'], lr_backbone=1e-5,
            position_embedding='sine', masks=False, dilation=False, pre_norm=False,
        )
        if pretrained is not None:
            args.pretrained_backbone = pretrained
        self.model = build(args)
        self.kl_weight = config['kl_weight']
        self.register_buffer('image_mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1))
        self.register_buffer('image_std', torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1))

    def forward(self, qpos, image, actions=None, is_pad=None):
        """归一化图像并预测动作；给出示范动作时计算 L1 与 KL。

        输入：qpos、image、actions、is_pad。
        输出：推理时动作块，训练时损失与统计字典。
        """
        image = (image - self.image_mean) / self.image_std
        pred, _, (mu, logvar) = self.model(qpos, image, None, actions, is_pad)
        if actions is None:
            return pred
        l1, total, count = masked_l1(pred, actions, is_pad)
        kl_by_sample = -0.5 * (1 + logvar - mu.square() - logvar.exp()).sum(-1)
        kl = kl_by_sample.mean()
        return dict(
            loss=l1 + self.kl_weight * kl, l1=l1, kl=kl,
            l1_sum=total.detach(), valid_elements=count.detach(),
            kl_sum=kl_by_sample.detach().sum(), batch_size=len(qpos),
        )

def make_optimizer(policy, config):
    """分别组织图像骨干与其他参数的学习率。

    输入：policy、config。
    输出：AdamW 优化器。
    """
    groups = [
        dict(params=[p for n, p in policy.named_parameters()
                     if p.requires_grad and 'backbones' not in n], lr=config['lr']),
        dict(params=[p for n, p in policy.named_parameters()
                     if p.requires_grad and 'backbones' in n], lr=config['lr_backbone']),
    ]
    return torch.optim.AdamW(groups, lr=config['lr'], weight_decay=config['weight_decay'])
