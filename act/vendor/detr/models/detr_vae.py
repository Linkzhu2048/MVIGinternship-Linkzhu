# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""构建 ACT 的条件变分自编码器，利用双腕图像和关节状态预测动作块。"""
import torch
from torch import nn
from torch.autograd import Variable
from .backbone import build_backbone
from .transformer import build_transformer, TransformerEncoder, TransformerEncoderLayer
import numpy as np

def reparametrize(mu, logvar):
    """按均值和对数方差重参数化采样潜变量。

    输入：mu、logvar。
    输出：与均值形状相同的潜变量张量。
    """
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std * eps

def get_sinusoid_encoding_table(n_position, d_hid):
    """生成序列使用的固定正弦位置编码表。

    输入：n_position、d_hid。
    输出：形状为位置数乘隐藏维度的张量。
    """

    def get_position_angle_vec(position):
        """计算指定位置各维度的位置编码角度。

        输入：position。
        输出：各隐藏维度的角度列表。
        """
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]
    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
    return torch.FloatTensor(sinusoid_table).unsqueeze(0)

class DETRVAE(nn.Module):
    """ACT 的条件变分自编码器与动作序列预测网络。"""

    def __init__(self, backbones, transformer, encoder, state_dim, num_queries, camera_names):
        """初始化ACT 的条件变分自编码器与动作序列预测网络。

        输入：backbones、transformer、encoder、state_dim、num_queries、camera_names。
        输出：配置好的模块实例。
        """
        super().__init__()
        self.num_queries = num_queries
        self.camera_names = camera_names
        self.transformer = transformer
        self.encoder = encoder
        hidden_dim = transformer.d_model
        self.action_head = nn.Linear(hidden_dim, state_dim)
        self.is_pad_head = nn.Linear(hidden_dim, 1)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        if backbones is not None:
            self.input_proj = nn.Conv2d(backbones[0].num_channels, hidden_dim, kernel_size=1)
            self.backbones = nn.ModuleList(backbones)
            self.input_proj_robot_state = nn.Linear(14, hidden_dim)
        else:
            self.input_proj_robot_state = nn.Linear(14, hidden_dim)
            self.input_proj_env_state = nn.Linear(7, hidden_dim)
            self.pos = torch.nn.Embedding(2, hidden_dim)
            self.backbones = None
        self.latent_dim = 32
        self.cls_embed = nn.Embedding(1, hidden_dim)
        self.encoder_action_proj = nn.Linear(14, hidden_dim)
        self.encoder_joint_proj = nn.Linear(14, hidden_dim)
        self.latent_proj = nn.Linear(hidden_dim, self.latent_dim * 2)
        self.register_buffer('pos_table', get_sinusoid_encoding_table(1 + 1 + num_queries, hidden_dim))
        self.latent_out_proj = nn.Linear(self.latent_dim, hidden_dim)
        self.additional_pos_embed = nn.Embedding(2, hidden_dim)

    def forward(self, qpos, image, env_state, actions=None, is_pad=None):
        """训练时编码示范动作得到潜变量，推理时使用零潜变量，并预测动作序列。

        输入：qpos、image、env_state、actions、is_pad。
        输出：动作预测、填充预测以及潜变量均值和对数方差。
        """
        is_training = actions is not None
        bs, _ = qpos.shape
        if is_training:
            action_embed = self.encoder_action_proj(actions)
            qpos_embed = self.encoder_joint_proj(qpos)
            qpos_embed = torch.unsqueeze(qpos_embed, axis=1)
            cls_embed = self.cls_embed.weight
            cls_embed = torch.unsqueeze(cls_embed, axis=0).repeat(bs, 1, 1)
            encoder_input = torch.cat([cls_embed, qpos_embed, action_embed], axis=1)
            encoder_input = encoder_input.permute(1, 0, 2)
            cls_joint_is_pad = torch.full((bs, 2), False).to(qpos.device)
            is_pad = torch.cat([cls_joint_is_pad, is_pad], axis=1)
            pos_embed = self.pos_table.clone().detach()
            pos_embed = pos_embed.permute(1, 0, 2)
            encoder_output = self.encoder(encoder_input, pos=pos_embed, src_key_padding_mask=is_pad)
            encoder_output = encoder_output[0]
            latent_info = self.latent_proj(encoder_output)
            mu = latent_info[:, :self.latent_dim]
            logvar = latent_info[:, self.latent_dim:]
            latent_sample = reparametrize(mu, logvar)
            latent_input = self.latent_out_proj(latent_sample)
        else:
            mu = logvar = None
            latent_sample = torch.zeros([bs, self.latent_dim], dtype=torch.float32).to(qpos.device)
            latent_input = self.latent_out_proj(latent_sample)
        if self.backbones is not None:
            all_cam_features = []
            all_cam_pos = []
            for cam_id, cam_name in enumerate(self.camera_names):
                features, pos = self.backbones[0](image[:, cam_id])
                features = features[0]
                pos = pos[0]
                all_cam_features.append(self.input_proj(features))
                all_cam_pos.append(pos)
            proprio_input = self.input_proj_robot_state(qpos)
            src = torch.cat(all_cam_features, axis=3)
            pos = torch.cat(all_cam_pos, axis=3)
            hs = self.transformer(src, None, self.query_embed.weight, pos, latent_input, proprio_input, self.additional_pos_embed.weight)[0]
        else:
            qpos = self.input_proj_robot_state(qpos)
            env_state = self.input_proj_env_state(env_state)
            transformer_input = torch.cat([qpos, env_state], axis=1)
            hs = self.transformer(transformer_input, None, self.query_embed.weight, self.pos.weight)[0]
        a_hat = self.action_head(hs)
        is_pad_hat = self.is_pad_head(hs)
        return (a_hat, is_pad_hat, [mu, logvar])

def build_encoder(args):
    """按配置构建用于编码示范动作的 Transformer 编码器。

    输入：args。
    输出：编码器模块。
    """
    d_model = args.hidden_dim
    dropout = args.dropout
    nhead = args.nheads
    dim_feedforward = args.dim_feedforward
    num_encoder_layers = args.enc_layers
    normalize_before = args.pre_norm
    activation = 'relu'
    encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, normalize_before)
    encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
    encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
    return encoder

def build(args):
    """组装图像骨干、Transformer 和潜变量编码器。

    输入：args。
    输出：ACT 的 DETRVAE 模型。
    """
    state_dim = 14
    backbones = []
    backbone = build_backbone(args)
    backbones.append(backbone)
    transformer = build_transformer(args)
    encoder = build_encoder(args)
    model = DETRVAE(backbones, transformer, encoder, state_dim=state_dim, num_queries=args.num_queries, camera_names=args.camera_names)
    n_parameters = sum((p.numel() for p in model.parameters() if p.requires_grad))
    print('number of parameters: %.2fM' % (n_parameters / 1000000.0,))
    return model
