# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from util.misc import NestedTensor, is_main_process

from .position_encoding import build_position_encoding

import IPython
e = IPython.embed

class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other policy_models than torchvision.policy_models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool):
        super().__init__()
        # for name, parameter in backbone.named_parameters(): # only train later layers # TODO do we want this?
        #     if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
        #         parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {'layer4': "0"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor):
        xs = self.body(tensor)
        return xs
        # out: Dict[str, NestedTensor] = {}
        # for name, x in xs.items():
        #     m = tensor_list.mask
        #     assert m is not None
        #     mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
        #     out[name] = NestedTensor(x, mask)
        # return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process(), norm_layer=FrozenBatchNorm2d) # pretrained # TODO do we want frozen batch_norm??
        if name in ('resnet18', 'resnet34'):
            layer_channels = [128, 256, 512]
        else:
            layer_channels = [512, 1024, 2048]
        num_channels = layer_channels if return_interm_layers else [layer_channels[-1]]
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.dtype))

        return out, pos

class HSFPN(nn.Module):
    """高级筛选特征金字塔（参考MFDS-DETR[2,3](@ref)）"""
    def __init__(self, in_channels_list, out_channels=256):
        super().__init__()
        #通道注意力筛选器（配置不同层特征权重）
        self.ca = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(ch, out_channels, 1),
                nn.Sigmoid()
            ) for ch in in_channels_list
        ])
        # self.ca = nn.ModuleList([
        #     nn.Sequential(
        #         nn.AdaptiveAvgPool2d(1),
        #         nn.Conv2d(ch, out_channels//16, 1),  # 压缩通道
        #         nn.ReLU(),
        #         nn.Conv2d(out_channels//16, out_channels, 1),  # 激励通道
        #         nn.Sigmoid()
        # ) for ch in in_channels_list
        # ])
        # 横向连接卷积
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(ch, out_channels, 1) for ch in in_channels_list
        ])
        # 融合卷积
        self.fusion_conv = nn.Conv2d(out_channels, out_channels, 3, padding=1)

    def forward(self, features):
        c3, c4, c5 = features
        
        # 自上而下融合路径
        p5 = self.lateral_convs[2](c5)# [B,512,7,7] → [B,256,7,7]
        p4 = self.ca[2](c5) * F.interpolate(p5, scale_factor=2, mode='nearest') + self.lateral_convs[1](c4)#上采样p5->c3，→ [B,256,14,14]  # 使用更高效的上采样模式
        del c5
        p3 = self.ca[1](c4) * F.interpolate(p4, scale_factor=2, mode='nearest') + self.lateral_convs[0](c3)#→ [B,256,28,28]
        p2 = self.fusion_conv(p3 + F.interpolate(p4, scale_factor=2, mode='nearest') + F.interpolate(p5, scale_factor=4))
        p2= F.avg_pool2d(p2, kernel_size=4) #这样操作不知道能不能行 #→ [B,256,7,7] 其实是[B,512,15,20]
        # 特征增强
        return p2

class JoinerWithFPN(nn.Sequential):
        def forward(self, tensor_list: NestedTensor):
            xs = self[0](tensor_list)  # Backbone输出
            fused_feat = self[1](list(xs.values()))  # FPN融合
            pos = self[2](fused_feat)  # 位置编码
            return fused_feat, pos

def build_backbone(args):
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    return_interm_layers = args.masks
    backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model


def build_Joiner(args):
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone < 0 #反着写，改掉了backbone中的注释，让backbone fixed
    # return_interm_layers = args.masks
    return_interm_layers = True
    backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation)
    fpn = HSFPN(backbone.num_channels, args.hidden_dim)
    model = JoinerWithFPN(backbone, fpn, position_embedding)
    model.num_channels = args.hidden_dim
    return model