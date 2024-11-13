from __future__ import annotations
import re
import time
import math
import numpy as np
from functools import partial
from typing import Optional, Union, Type, List, Tuple, Callable, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"

from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrUpBlock
import timm
from mamba_ssm import Mamba

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]

class Att_MambaLayer(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2, num_slices=None, num_heads=2, drop=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.proj1 = nn.Conv2d(dim, dim, 3, 1, 1)
        self.proj2 = nn.Conv2d(dim, dim, 1, 1, 0)
        self.norm = nn.LayerNorm(dim)
        self.nonliner = nn.ReLU()

        assert dim % 2 == 0, "dim must be divisible by 2 for height and width splitting"

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)

        self.softmax = nn.Softmax(dim=-1)

        self.out_proj = nn.Linear(dim, dim)
        
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        self.fc1 = nn.Linear(dim, dim) 
        self.dwconv = DWConv(dim)         
        # Mamba Block
        self.mamba = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            bimamba_type="v3",
            nslices=num_slices,
        )
       
    def forward(self, x):
        B, C, H, W = x.shape
        x_skip = x
        assert C == self.dim
        n_tokens = x.shape[2:].numel()
        x_flat = x.transpose(1, 2).reshape(B, C, H, W)
        x_conv = self.proj1(x_flat)
        x_conv = x_conv.reshape(B, C, n_tokens).transpose(-1, -2)
        x_conv = self.norm(x_conv)

        # QKV attention mechanism
        Q = self.q_proj(x_conv) 
        K = self.k_proj(x_conv)
        V = self.v_proj(x_conv)

        Q = Q.reshape(B, -1, self.num_heads, C // self.num_heads).transpose(1, 2)
        K = K.reshape(B, -1, self.num_heads, C // self.num_heads).transpose(1, 2)
        V = V.reshape(B, -1, self.num_heads, C // self.num_heads).transpose(1, 2)

        attention_scores = torch.matmul(Q, K.transpose(-1, -2)) / (C // self.num_heads) ** 0.5
        attention_probs = self.softmax(attention_scores)
        attention_output = torch.matmul(attention_probs, V)

        attention_output = attention_output.transpose(1, 2).reshape(B, n_tokens, C)
        attention_output = self.out_proj(attention_output)
        
        x_attention_output = attention_output.view(B, C, H * W).transpose(1, 2)  # (B, H * W, C)
    
        x_mamba = self.mamba(x_attention_output)
        
        x_mamba = x_mamba.transpose(1, 2).reshape(B, C, H, W)
        x_mamba = self.proj1(x_mamba)
        x_mamba = self.nonliner(x_mamba)
        
        x_mamba = self.proj2(x_mamba)
        x = self.nonliner(x_mamba)
        
        x_s = x.reshape(B, C, H * W).contiguous()
        x_shift_r = x_s.transpose(1, 2)
        x = self.fc1(x_shift_r)
        x = self.dwconv(x, H, W) 
        
        x = x.view(B, H, W, C).permute(0, 3, 1, 2)
        out = x + x_skip
        return out

class MLC(nn.Module):
    def __init__(self, in_channles) -> None:
        super().__init__()
       
        self.proj = nn.Conv2d(in_channles, in_channles, 3, 1, 1)
        self.norm = nn.LayerNorm(in_channles)
        self.nonliner = nn.ReLU()

        self.proj2 = nn.Conv2d(in_channles, in_channles, 3, 1, 1)
        self.norm2 = nn.LayerNorm(in_channles)
        self.nonliner2 = nn.ReLU()

        self.proj3 = nn.Conv2d(in_channles, in_channles, 1, 1, 0)
        self.norm3 = nn.LayerNorm(in_channles)
        self.nonliner3 = nn.ReLU()

        self.proj4 = nn.Conv2d(in_channles, in_channles, 1, 1, 0)
        self.norm4 = nn.LayerNorm(in_channles)
        self.nonliner4 = nn.ReLU() 

    def forward(self, x):

        x_skip = x 
        B, C, H, W = x.shape

        x1 = self.proj(x)
        x1 = self.norm(x1)
        x1 = self.nonliner(x1)

        x1 = self.proj2(x1)
        x1 = self.norm2(x1)
        x1 = self.nonliner2(x1)
     
        x2 = self.proj3(x)
        x2 = self.norm3(x2)
        x2 = self.nonliner3(x2)

        x2 = self.proj3(x2)
        x2 = self.norm3(x2)
        x2 = self.nonliner3(x2)
        
        x = x1 + x2
        
        x = self.proj(x)
        x = self.nonliner(x)
        
        x = self.proj4(x)
        x = self.norm4(x)
        x = self.nonliner4(x)
        
        return x + x_skip

class DWConv(nn.Module):
    def __init__(self, dim):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class shiftmlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., shift_size=5):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.dim = in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.shift_size = shift_size
        self.pad = shift_size // 2
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
      #  print(f"shiftmlp forward: B={B}, N={N}, C={C}, H={H}, W={W}")
        
        # Initial reshaping
        xn = x.transpose(1, 2).view(B, C, H, W).contiguous()
      #  print(f"after transpose and view: {xn.shape}")
        
        # Padding
        xn = F.pad(xn, (self.pad, self.pad, self.pad, self.pad), "constant", 0)
       # print(f"after padding: {xn.shape}")
        
        # Shift and concatenate along width
        xs = torch.chunk(xn, self.shift_size, 1)
        x_shift = [torch.roll(x_c, shift, 2) for x_c, shift in zip(xs, range(-self.pad, self.pad + 1))]
        x_cat = torch.cat(x_shift, 1)
       # print(f"after chunking and rolling: {x_cat.shape}")
        
        # Narrow and reshape
        x_cat = torch.narrow(x_cat, 2, self.pad, H)
        x_s = torch.narrow(x_cat, 3, self.pad, W)
        x_s = x_s.reshape(B, C, H * W).contiguous()
       # print(f"after narrowing and reshaping: {x_s.shape}")
        
        # Transpose and apply fc1
        x_shift_r = x_s.transpose(1, 2)
        x = self.fc1(x_shift_r)
      #  print(f"after fc1: {x.shape}")
        
        # Apply DWConv
        x = self.dwconv(x, H, W)
      #  print(f"after dwconv: {x.shape}")
        
        # Activation and dropout
        x = self.act(x)
        x = self.drop(x)
        
        # Ensure the output shape is correct for the second shift and concatenate
        B, N, C = x.shape
        H_out = int(math.sqrt(N))
        W_out = H_out
      #  print(f"adjusted H_out: {H_out}, W_out: {W_out}")
        
        # Reshape for second shift and concatenation along height
        xn = x.transpose(1, 2).view(B, C, H_out, W_out).contiguous()
        xn = F.pad(xn, (self.pad, self.pad, self.pad, self.pad), "constant", 0)
        xs = torch.chunk(xn, self.shift_size, 1)
        x_shift = [torch.roll(x_c, shift, 3) for x_c, shift in zip(xs, range(-self.pad, self.pad + 1))]
        x_cat = torch.cat(x_shift, 1)
        x_cat = torch.narrow(x_cat, 2, self.pad, H_out)
        x_s = torch.narrow(x_cat, 3, self.pad, W_out)
        x_s = x_s.reshape(B, C, H_out * W_out).contiguous()
      #  print(f"after second narrowing and reshaping: {x_s.shape}")
        
        # Transpose and apply fc2
        x_shift_c = x_s.transpose(1, 2)
        x = self.fc2(x_shift_c)
       # print(f"after fc2: {x.shape}")
        
        # Final dropout and return
        x = self.drop(x)
        return x


class MAMEncoder(nn.Module):
    def __init__(self, in_chans=1, depths=[2, 2, 2, 2], dims=[48, 96, 192, 384],
                 drop_path_rate=0., layer_scale_init_value=1e-6, out_indices=[0, 1, 2, 3]):
        super().__init__()

        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=7, stride=2, padding=3),
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                nn.InstanceNorm2d(dims[i]),
                nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        self.mlcs = nn.ModuleList()
        num_slices_list = [64, 32, 16, 8]
        cur = 0
        for i in range(4):
            mlc = MLC(dims[i])

            stage = nn.Sequential(
                *[Att_MambaLayer(dim=dims[i], num_slices=num_slices_list[i]) for j in range(depths[i])]
            )

            self.stages.append(stage)
            self.mlcs.append(mlc)
            cur += depths[i]

        self.out_indices = out_indices
        
        self.mlps = nn.ModuleList()
       
        for i_layer in range(4):
            layer = nn.InstanceNorm2d(dims[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)
            self.mlps.append(shiftmlp(dims[i_layer], 2 * dims[i_layer]))
   

    def forward_features(self, x):
        outs = []
        for i in range(4):
            x = self.downsample_layers[i](x)
           # print(f"downsample layer {i} output: {x.shape}")
            x = self.mlcs[i](x)
            x = self.stages[i](x)
            
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x)
                B, C, H, W = x.shape
               # print(f"norm layer {i} output: {x_out.shape}")
                x_out = x_out.permute(0, 2, 3, 1).reshape(B, H*W, C) 
               # print(f"reshaped x_out before shiftmlp: {x_out.shape}")
                x_out = self.mlps[i](x_out, H, W)  
               
                x_out = x_out.view(B, H, W, C).permute(0, 3, 1, 2) 
               # print(f"reshaped x_out after shiftmlp: {x_out.shape}")
                outs.append(x_out)

        return tuple(outs)

    def forward(self, x):
        x = self.forward_features(x)
        return x

class UNetUpBlockWithInterpolation(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels):
        super(UNetUpBlockWithInterpolation, self).__init__()
        # Use 1x1 convolution to adjust the number of channels.
        self.reduce_channels = nn.Conv2d(in_channels + skip_channels, in_channels, kernel_size=1)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat((x, skip), dim=1)
        x = self.reduce_channels(x)  
        x = self.conv(x)
        return x
              
class AttmNet(nn.Module):
    def __init__(self, 
                 in_chans=3,
                 out_chans=1,
                 depths=[2, 2, 2, 2],
                 feat_size=[48, 96, 192, 384],
                 hidden_size=768,
                 norm_name="instance",
                 res_block=True,
                 spatial_dims=2) -> None:
        super().__init__()

        self.hidden_size = hidden_size

        self.vit = MAMEncoder(
            in_chans=in_chans, 
            depths=depths,
            dims=feat_size,
            drop_path_rate=0,
            layer_scale_init_value=1e-6
        )
        
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_chans,
            out_channels=feat_size[0],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder2 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=feat_size[0],
            out_channels=feat_size[1],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder3 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=feat_size[1],
            out_channels=feat_size[2],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder4 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=feat_size[2],
            out_channels=feat_size[3],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder5 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=feat_size[3],
            out_channels=hidden_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.decoder5 = UNetUpBlockWithInterpolation(hidden_size, feat_size[3], feat_size[3])
        self.decoder4 = UNetUpBlockWithInterpolation(feat_size[3], feat_size[2], feat_size[2])
        self.decoder3 = UNetUpBlockWithInterpolation(feat_size[2], feat_size[1], feat_size[1])
        self.decoder2 = UNetUpBlockWithInterpolation(feat_size[1], feat_size[0], feat_size[0])

        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feat_size[0], out_channels=out_chans)

    def forward(self, x_in):
        outs = self.vit(x_in)
        enc1 = self.encoder1(x_in)
        x2 = outs[0]
        enc2 = self.encoder2(x2)
        x3 = outs[1]
        enc3 = self.encoder3(x3)
        x4 = outs[2]
        enc4 = self.encoder4(x4)
        enc_hidden = self.encoder5(outs[3])

        dec3 = self.decoder5(enc_hidden, enc4)
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder2(dec1, enc1)

        out = self.out(dec0)
        return out

 
