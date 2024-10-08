##############################################################
# % Author: Castle
# % Date:01/12/2022
###############################################################

import torch
import torch.nn as nn
from functools import partial, reduce
from timm.models.layers import DropPath, trunc_normal_
from extensions.chamfer_dist import ChamferDistanceL1
from .build import MODELS, build_model_from_cfg
from models.Transformer_utils import *
from utils import misc
from utils.utils_pose import geodesic_rotation_error
from utils import convert_rotation

from utils.commons import categories_with_labels

import torch
import numpy as np
from PIL import Image
import os
import torch.nn.functional as F



class SelfAttnBlockApi(nn.Module):
    r'''
        1. Norm Encoder Block 
            block_style = 'attn'
        2. Concatenation Fused Encoder Block
            block_style = 'attn-deform'  
            combine_style = 'concat'
        3. Three-layer Fused Encoder Block
            block_style = 'attn-deform'  
            combine_style = 'onebyone'        
    '''
    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., init_values=None,
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, block_style='attn-deform', combine_style='concat',
            k=10, n_group=2
        ):

        super().__init__()
        self.combine_style = combine_style
        assert combine_style in ['concat', 'onebyone'], f'got unexpect combine_style {combine_style} for local and global attn'
        self.norm1 = norm_layer(dim)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()        

        # Api desigin
        block_tokens = block_style.split('-')
        assert len(block_tokens) > 0 and len(block_tokens) <= 2, f'invalid block_style {block_style}'
        self.block_length = len(block_tokens)
        self.attn = None
        self.local_attn = None
        for block_token in block_tokens:
            assert block_token in ['attn', 'rw_deform', 'deform', 'graph', 'deform_graph'], f'got unexpect block_token {block_token} for Block component'
            if block_token == 'attn':
                self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
            elif block_token == 'rw_deform':
                self.local_attn = DeformableLocalAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, k=k, n_group=n_group)
            elif block_token == 'deform':
                self.local_attn = DeformableLocalCrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, k=k, n_group=n_group)
            elif block_token == 'graph':
                self.local_attn = DynamicGraphAttention(dim, k=k)
            elif block_token == 'deform_graph':
                self.local_attn = improvedDeformableLocalGraphAttention(dim, k=k)
        if self.attn is not None and self.local_attn is not None:
            if combine_style == 'concat':
                self.merge_map = nn.Linear(dim*2, dim)
            else:
                self.norm3 = norm_layer(dim)
                self.ls3 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
                self.drop_path3 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, pos, idx=None):
        feature_list = []
        if self.block_length == 2:
            if self.combine_style == 'concat':
                norm_x = self.norm1(x)
                if self.attn is not None:
                    global_attn_feat = self.attn(norm_x)
                    feature_list.append(global_attn_feat)
                if self.local_attn is not None:
                    local_attn_feat = self.local_attn(norm_x, pos, idx=idx)
                    feature_list.append(local_attn_feat)
                # combine
                if len(feature_list) == 2:
                    f = torch.cat(feature_list, dim=-1)
                    f = self.merge_map(f)
                    x = x + self.drop_path1(self.ls1(f))
                else:
                    raise RuntimeError()
            else: # onebyone
                x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
                x = x + self.drop_path3(self.ls3(self.local_attn(self.norm3(x), pos, idx=idx)))

        elif self.block_length == 1:
            norm_x = self.norm1(x)
            if self.attn is not None:
                global_attn_feat = self.attn(norm_x)
                feature_list.append(global_attn_feat)
            if self.local_attn is not None:
                local_attn_feat = self.local_attn(norm_x, pos, idx=idx)
                feature_list.append(local_attn_feat)
            # combine
            if len(feature_list) == 1:
                f = feature_list[0]
                x = x + self.drop_path1(self.ls1(f))
            else:
                raise RuntimeError()

        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x
   
class CrossAttnBlockApi(nn.Module):
    r'''
        1. Norm Decoder Block 
            self_attn_block_style = 'attn'
            cross_attn_block_style = 'attn'
        2. Concatenation Fused Decoder Block
            self_attn_block_style = 'attn-deform'  
            self_attn_combine_style = 'concat'
            cross_attn_block_style = 'attn-deform'  
            cross_attn_combine_style = 'concat'
        3. Three-layer Fused Decoder Block
            self_attn_block_style = 'attn-deform'  
            self_attn_combine_style = 'onebyone'
            cross_attn_block_style = 'attn-deform'  
            cross_attn_combine_style = 'onebyone'    
        4. Design by yourself
            #  only deform the cross attn
            self_attn_block_style = 'attn'  
            cross_attn_block_style = 'attn-deform'  
            cross_attn_combine_style = 'concat'    
            #  perform graph conv on self attn
            self_attn_block_style = 'attn-graph'  
            self_attn_combine_style = 'concat'    
            cross_attn_block_style = 'attn-deform'  
            cross_attn_combine_style = 'concat'    
    '''
    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., init_values=None,
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, 
            self_attn_block_style='attn-deform', self_attn_combine_style='concat',
            cross_attn_block_style='attn-deform', cross_attn_combine_style='concat',
            k=10, n_group=2
        ):
        super().__init__()        
        self.norm2 = norm_layer(dim)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()      

        # Api desigin
        # first we deal with self-attn
        self.norm1 = norm_layer(dim)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.self_attn_combine_style = self_attn_combine_style
        assert self_attn_combine_style in ['concat', 'onebyone'], f'got unexpect self_attn_combine_style {self_attn_combine_style} for local and global attn'
  
        self_attn_block_tokens = self_attn_block_style.split('-')
        assert len(self_attn_block_tokens) > 0 and len(self_attn_block_tokens) <= 2, f'invalid self_attn_block_style {self_attn_block_style}'
        self.self_attn_block_length = len(self_attn_block_tokens)
        self.self_attn = None
        self.local_self_attn = None
        for self_attn_block_token in self_attn_block_tokens:
            assert self_attn_block_token in ['attn', 'rw_deform', 'deform', 'graph', 'deform_graph'], f'got unexpect self_attn_block_token {self_attn_block_token} for Block component'
            if self_attn_block_token == 'attn':
                self.self_attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
            elif self_attn_block_token == 'rw_deform':
                self.local_self_attn = DeformableLocalAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, k=k, n_group=n_group)
            elif self_attn_block_token == 'deform':
                self.local_self_attn = DeformableLocalCrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, k=k, n_group=n_group)
            elif self_attn_block_token == 'graph':
                self.local_self_attn = DynamicGraphAttention(dim, k=k)
            elif self_attn_block_token == 'deform_graph':
                self.local_self_attn = improvedDeformableLocalGraphAttention(dim, k=k)
        if self.self_attn is not None and self.local_self_attn is not None:
            if self_attn_combine_style == 'concat':
                self.self_attn_merge_map = nn.Linear(dim*2, dim)
            else:
                self.norm3 = norm_layer(dim)
                self.ls3 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
                self.drop_path3 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # Then we deal with cross-attn
        self.norm_q = norm_layer(dim)
        self.norm_v = norm_layer(dim)
        self.ls4 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path4 = DropPath(drop_path) if drop_path > 0. else nn.Identity()  

        self.cross_attn_combine_style = cross_attn_combine_style
        assert cross_attn_combine_style in ['concat', 'onebyone'], f'got unexpect cross_attn_combine_style {cross_attn_combine_style} for local and global attn'
        
        # Api desigin
        cross_attn_block_tokens = cross_attn_block_style.split('-')
        assert len(cross_attn_block_tokens) > 0 and len(cross_attn_block_tokens) <= 2, f'invalid cross_attn_block_style {cross_attn_block_style}'
        self.cross_attn_block_length = len(cross_attn_block_tokens)
        self.cross_attn = None
        self.local_cross_attn = None
        for cross_attn_block_token in cross_attn_block_tokens:
            assert cross_attn_block_token in ['attn', 'deform', 'graph', 'deform_graph'], f'got unexpect cross_attn_block_token {cross_attn_block_token} for Block component'
            if cross_attn_block_token == 'attn':
                self.cross_attn = CrossAttention(dim, dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
            elif cross_attn_block_token == 'deform':
                self.local_cross_attn = DeformableLocalCrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, k=k, n_group=n_group)
            elif cross_attn_block_token == 'graph':
                self.local_cross_attn = DynamicGraphAttention(dim, k=k)
            elif cross_attn_block_token == 'deform_graph':
                self.local_cross_attn = improvedDeformableLocalGraphAttention(dim, k=k)
        if self.cross_attn is not None and self.local_cross_attn is not None:
            if cross_attn_combine_style == 'concat':
                self.cross_attn_merge_map = nn.Linear(dim*2, dim)
            else:
                self.norm_q_2 = norm_layer(dim)
                self.norm_v_2 = norm_layer(dim)
                self.ls5 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
                self.drop_path5 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, q, v, q_pos, v_pos, self_attn_idx=None, cross_attn_idx=None, denoise_length=None):
        # q = q + self.drop_path(self.self_attn(self.norm1(q)))

        # calculate mask, shape N,N
        # 1 for mask, 0 for not mask
        # mask shape N, N
        # q: [ true_query; denoise_token ]
        if denoise_length is None:
            mask = None
        else:
            query_len = q.size(1)
            mask = torch.zeros(query_len, query_len).to(q.device)
            mask[:-denoise_length, -denoise_length:] = 1.

        # Self attn
        feature_list = []
        if self.self_attn_block_length == 2:
            if self.self_attn_combine_style == 'concat':
                norm_q = self.norm1(q)
                if self.self_attn is not None:
                    global_attn_feat = self.self_attn(norm_q, mask=mask)
                    feature_list.append(global_attn_feat)
                if self.local_self_attn is not None:
                    local_attn_feat = self.local_self_attn(norm_q, q_pos, idx=self_attn_idx, denoise_length=denoise_length)
                    feature_list.append(local_attn_feat)
                # combine
                if len(feature_list) == 2:
                    f = torch.cat(feature_list, dim=-1)
                    f = self.self_attn_merge_map(f)
                    q = q + self.drop_path1(self.ls1(f))
                else:
                    raise RuntimeError()
            else: # onebyone
                q = q + self.drop_path1(self.ls1(self.self_attn(self.norm1(q), mask=mask)))
                q = q + self.drop_path3(self.ls3(self.local_self_attn(self.norm3(q), q_pos, idx=self_attn_idx, denoise_length=denoise_length)))

        elif self.self_attn_block_length == 1:
            norm_q = self.norm1(q)
            if self.self_attn is not None:
                global_attn_feat = self.self_attn(norm_q, mask=mask)
                feature_list.append(global_attn_feat)
            if self.local_self_attn is not None:
                local_attn_feat = self.local_self_attn(norm_q, q_pos, idx=self_attn_idx, denoise_length=denoise_length)
                feature_list.append(local_attn_feat)
            # combine
            if len(feature_list) == 1:
                f = feature_list[0]
                q = q + self.drop_path1(self.ls1(f))
            else:
                raise RuntimeError()

        # q = q + self.drop_path(self.attn(self.norm_q(q), self.norm_v(v)))
        # Cross attn
        feature_list = []
        if self.cross_attn_block_length == 2:
            if self.cross_attn_combine_style == 'concat':
                norm_q = self.norm_q(q)
                norm_v = self.norm_v(v)
                if self.cross_attn is not None:
                    global_attn_feat = self.cross_attn(norm_q, norm_v)
                    feature_list.append(global_attn_feat)
                if self.local_cross_attn is not None:
                    local_attn_feat = self.local_cross_attn(q=norm_q, v=norm_v, q_pos=q_pos, v_pos=v_pos, idx=cross_attn_idx)
                    feature_list.append(local_attn_feat)
                # combine
                if len(feature_list) == 2:
                    f = torch.cat(feature_list, dim=-1)
                    f = self.cross_attn_merge_map(f)
                    q = q + self.drop_path4(self.ls4(f))
                else:
                    raise RuntimeError()
            else: # onebyone
                q = q + self.drop_path4(self.ls4(self.cross_attn(self.norm_q(q), self.norm_v(v))))
                q = q + self.drop_path5(self.ls5(self.local_cross_attn(q=self.norm_q_2(q), v=self.norm_v_2(v), q_pos=q_pos, v_pos=v_pos, idx=cross_attn_idx)))

        elif self.cross_attn_block_length == 1:
            norm_q = self.norm_q(q)
            norm_v = self.norm_v(v)
            if self.cross_attn is not None:
                global_attn_feat = self.cross_attn(norm_q, norm_v)
                feature_list.append(global_attn_feat)
            if self.local_cross_attn is not None:
                local_attn_feat = self.local_cross_attn(q=norm_q, v=norm_v, q_pos=q_pos, v_pos=v_pos, idx=cross_attn_idx)
                feature_list.append(local_attn_feat)
            # combine
            if len(feature_list) == 1:
                f = feature_list[0]
                q = q + self.drop_path4(self.ls4(f))
            else:
                raise RuntimeError()

        q = q + self.drop_path2(self.ls2(self.mlp(self.norm2(q))))
        return q
######################################## Entry ########################################  

class TransformerEncoder(nn.Module):
    """ Transformer Encoder without hierarchical structure
    """
    def __init__(self, embed_dim=256, depth=4, num_heads=4, mlp_ratio=4., qkv_bias=False, init_values=None,
        drop_rate=0., attn_drop_rate=0., drop_path_rate=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
        block_style_list=['attn-deform'], combine_style='concat', k=10, n_group=2):
        super().__init__()
        self.k = k
        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(SelfAttnBlockApi(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, init_values=init_values,
                drop=drop_rate, attn_drop=attn_drop_rate, 
                drop_path = drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate,
                act_layer=act_layer, norm_layer=norm_layer,
                block_style=block_style_list[i], combine_style=combine_style, k=k, n_group=n_group
            ))

    def forward(self, x, pos):
        idx = idx = knn_point(self.k, pos, pos)
        for _, block in enumerate(self.blocks):
            x = block(x, pos, idx=idx) 
        return x

class TransformerDecoder(nn.Module):
    """ Transformer Decoder without hierarchical structure
    """
    def __init__(self, embed_dim=256, depth=4, num_heads=4, mlp_ratio=4., qkv_bias=False, init_values=None,
        drop_rate=0., attn_drop_rate=0., drop_path_rate=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
        self_attn_block_style_list=['attn-deform'], self_attn_combine_style='concat',
        cross_attn_block_style_list=['attn-deform'], cross_attn_combine_style='concat',
        k=10, n_group=2):
        super().__init__()
        self.k = k
        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(CrossAttnBlockApi(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, init_values=init_values,
                drop=drop_rate, attn_drop=attn_drop_rate, 
                drop_path = drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate,
                act_layer=act_layer, norm_layer=norm_layer,
                self_attn_block_style=self_attn_block_style_list[i], self_attn_combine_style=self_attn_combine_style,
                cross_attn_block_style=cross_attn_block_style_list[i], cross_attn_combine_style=cross_attn_combine_style,
                k=k, n_group=n_group
            ))

    def forward(self, q, v, q_pos, v_pos, denoise_length=None):
        if denoise_length is None:
            self_attn_idx = knn_point(self.k, q_pos, q_pos)
        else:
            self_attn_idx = None
        cross_attn_idx = knn_point(self.k, v_pos, q_pos)
        for _, block in enumerate(self.blocks):
            q = block(q, v, q_pos, v_pos, self_attn_idx=self_attn_idx, cross_attn_idx=cross_attn_idx, denoise_length=denoise_length)
        return q

class PointTransformerEncoder(nn.Module):
    """ Vision Transformer for point cloud encoder/decoder
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    Args:
        embed_dim (int): embedding dimension
        depth (int): depth of transformer
        num_heads (int): number of attention heads
        mlp_ratio (int): ratio of mlp hidden dim to embedding dim
        qkv_bias (bool): enable bias for qkv if True
        init_values: (float): layer-scale init values
        drop_rate (float): dropout rate
        attn_drop_rate (float): attention dropout rate
        drop_path_rate (float): stochastic depth rate
        norm_layer: (nn.Module): normalization layer
        act_layer: (nn.Module): MLP activation layer
    """
    def __init__(
            self, embed_dim=256, depth=12, num_heads=4, mlp_ratio=4., qkv_bias=True, init_values=None,
            drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
            norm_layer=None, act_layer=None,
            block_style_list=['attn-deform'], combine_style='concat',
            k=10, n_group=2
        ):
        super().__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        assert len(block_style_list) == depth
        self.blocks = TransformerEncoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            depth = depth,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            init_values=init_values,
            drop_rate=drop_rate, 
            attn_drop_rate=attn_drop_rate,
            drop_path_rate = dpr,
            norm_layer=norm_layer, 
            act_layer=act_layer,
            block_style_list=block_style_list,
            combine_style=combine_style,
            k=k,
            n_group=n_group)
        self.norm = norm_layer(embed_dim) 
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, pos):
        x = self.blocks(x, pos)
        return x

class PointTransformerDecoder(nn.Module):
    """ Vision Transformer for point cloud encoder/decoder
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    """
    def __init__(
            self, embed_dim=256, depth=12, num_heads=4, mlp_ratio=4., qkv_bias=True, init_values=None,
            drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
            norm_layer=None, act_layer=None,
            self_attn_block_style_list=['attn-deform'], self_attn_combine_style='concat',
            cross_attn_block_style_list=['attn-deform'], cross_attn_combine_style='concat',
            k=10, n_group=2
        ):
        """
        Args:
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            init_values: (float): layer-scale init values
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer: (nn.Module): normalization layer
            act_layer: (nn.Module): MLP activation layer
        """
        super().__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        assert len(self_attn_block_style_list) == len(cross_attn_block_style_list) == depth
        self.blocks = TransformerDecoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            depth = depth,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            init_values=init_values,
            drop_rate=drop_rate, 
            attn_drop_rate=attn_drop_rate,
            drop_path_rate = dpr,
            norm_layer=norm_layer, 
            act_layer=act_layer,
            self_attn_block_style_list=self_attn_block_style_list, 
            self_attn_combine_style=self_attn_combine_style,
            cross_attn_block_style_list=cross_attn_block_style_list, 
            cross_attn_combine_style=cross_attn_combine_style,
            k=k, 
            n_group=n_group
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, q, v, q_pos, v_pos, denoise_length=None):
        q = self.blocks(q, v, q_pos, v_pos, denoise_length=denoise_length)
        return q

class PointTransformerEncoderEntry(PointTransformerEncoder):
    def __init__(self, config, **kwargs):
        super().__init__(**dict(config))

class PointTransformerDecoderEntry(PointTransformerDecoder):
    def __init__(self, config, **kwargs):
        super().__init__(**dict(config))

######################################## Grouper ########################################  
class DGCNN_Grouper(nn.Module):
    def __init__(self, k = 16):
        super().__init__()
        '''
        K has to be 16
        '''
        print('using group version 2')
        self.k = k
        # self.knn = KNN(k=k, transpose_mode=False)
        self.input_trans = nn.Conv1d(3, 8, 1)

        self.layer1 = nn.Sequential(nn.Conv2d(16, 32, kernel_size=1, bias=False),
                                   nn.GroupNorm(4, 32),
                                   nn.LeakyReLU(negative_slope=0.2)
                                   )

        self.layer2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   nn.GroupNorm(4, 64),
                                   nn.LeakyReLU(negative_slope=0.2)
                                   )

        self.layer3 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1, bias=False),
                                   nn.GroupNorm(4, 64),
                                   nn.LeakyReLU(negative_slope=0.2)
                                   )

        self.layer4 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=1, bias=False),
                                   nn.GroupNorm(4, 128),
                                   nn.LeakyReLU(negative_slope=0.2)
                                   )
        self.num_features = 128
    @staticmethod
    def fps_downsample(coor, x, num_group):
        xyz = coor.transpose(1, 2).contiguous() # b, n, 3
        fps_idx = pointnet2_utils.furthest_point_sample(xyz, num_group)

        combined_x = torch.cat([coor, x], dim=1)

        new_combined_x = (
            pointnet2_utils.gather_operation(
                combined_x, fps_idx
            )
        )

        new_coor = new_combined_x[:, :3]
        new_x = new_combined_x[:, 3:]

        return new_coor, new_x

    def get_graph_feature(self, coor_q, x_q, coor_k, x_k):

        # coor: bs, 3, np, x: bs, c, np

        k = self.k
        batch_size = x_k.size(0)
        num_points_k = x_k.size(2)
        num_points_q = x_q.size(2)

        with torch.no_grad():
            # _, idx = self.knn(coor_k, coor_q)  # bs k np
            idx = knn_point(k, coor_k.transpose(-1, -2).contiguous(), coor_q.transpose(-1, -2).contiguous()) # B G M
            idx = idx.transpose(-1, -2).contiguous()
            assert idx.shape[1] == k
            idx_base = torch.arange(0, batch_size, device=x_q.device).view(-1, 1, 1) * num_points_k
            idx = idx + idx_base
            idx = idx.view(-1)
        num_dims = x_k.size(1)
        x_k = x_k.transpose(2, 1).contiguous()
        feature = x_k.view(batch_size * num_points_k, -1)[idx, :]
        feature = feature.view(batch_size, k, num_points_q, num_dims).permute(0, 3, 2, 1).contiguous()
        x_q = x_q.view(batch_size, num_dims, num_points_q, 1).expand(-1, -1, -1, k)
        feature = torch.cat((feature - x_q, x_q), dim=1)
        return feature

    def forward(self, x, num):
        '''
            INPUT:
                x : bs N 3
                num : list e.g.[1024, 512]
            ----------------------
            OUTPUT:

                coor bs N 3
                f    bs N C(128) 
        '''
        x = x.transpose(-1, -2).contiguous()

        coor = x
        f = self.input_trans(x)

        f = self.get_graph_feature(coor, f, coor, f)
        f = self.layer1(f)
        f = f.max(dim=-1, keepdim=False)[0]

        coor_q, f_q = self.fps_downsample(coor, f, num[0])
        f = self.get_graph_feature(coor_q, f_q, coor, f)
        f = self.layer2(f)
        f = f.max(dim=-1, keepdim=False)[0]
        coor = coor_q

        f = self.get_graph_feature(coor, f, coor, f)
        f = self.layer3(f)
        f = f.max(dim=-1, keepdim=False)[0]

        coor_q, f_q = self.fps_downsample(coor, f, num[1])
        f = self.get_graph_feature(coor_q, f_q, coor, f)
        f = self.layer4(f)
        f = f.max(dim=-1, keepdim=False)[0]
        coor = coor_q

        coor = coor.transpose(-1, -2).contiguous()
        f = f.transpose(-1, -2).contiguous()

        return coor, f

class Encoder(nn.Module):
    def __init__(self, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1)
        )
    def forward(self, point_groups):
        '''
            point_groups : B G N 3
            -----------------
            feature_global : B G C
        '''
        bs, g, n , _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, 3)
        # encoder
        feature = self.first_conv(point_groups.transpose(2,1))  # BG 256 n
        feature_global = torch.max(feature,dim=2,keepdim=True)[0]  # BG 256 1
        feature = torch.cat([feature_global.expand(-1,-1,n), feature], dim=1)# BG 512 n
        feature = self.second_conv(feature) # BG 1024 n
        feature_global = torch.max(feature, dim=2, keepdim=False)[0] # BG 1024
        return feature_global.reshape(bs, g, self.encoder_channel)

class SimpleEncoder(nn.Module):
    def __init__(self, k = 32, embed_dims=128):
        super().__init__()
        self.embedding = Encoder(embed_dims)
        self.group_size = k

        self.num_features = embed_dims

    def forward(self, xyz, n_group):
        # 2048 divide into 128 * 32, overlap is needed
        if isinstance(n_group, list):
            n_group = n_group[-1] 

        center = misc.fps(xyz, n_group) # B G 3
            
        assert center.size(1) == n_group, f'expect center to be B {n_group} 3, but got shape {center.shape}'
        
        batch_size, num_points, _ = xyz.shape
        # knn to get the neighborhood
        idx = knn_point(self.group_size, xyz, center)
        assert idx.size(1) == n_group
        assert idx.size(2) == self.group_size
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(batch_size, n_group, self.group_size, 3).contiguous()
            
        assert neighborhood.size(1) == n_group
        assert neighborhood.size(2) == self.group_size
            
        features = self.embedding(neighborhood) # B G C
        
        return center, features

######################################## Fold ########################################    
class Fold(nn.Module):
    def __init__(self, in_channel, step , hidden_dim=512):
        super().__init__()

        self.in_channel = in_channel
        self.step = step

        a = torch.linspace(-1., 1., steps=step, dtype=torch.float).view(1, step).expand(step, step).reshape(1, -1)
        b = torch.linspace(-1., 1., steps=step, dtype=torch.float).view(step, 1).expand(step, step).reshape(1, -1)
        self.folding_seed = torch.cat([a, b], dim=0).cuda()

        self.folding1 = nn.Sequential(
            nn.Conv1d(in_channel + 2, hidden_dim, 1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim//2, 1),
            nn.BatchNorm1d(hidden_dim//2),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim//2, 3, 1),
        )

        self.folding2 = nn.Sequential(
            nn.Conv1d(in_channel + 3, hidden_dim, 1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim//2, 1),
            nn.BatchNorm1d(hidden_dim//2),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim//2, 3, 1),
        )

    def forward(self, x):
        num_sample = self.step * self.step
        bs = x.size(0)
        features = x.view(bs, self.in_channel, 1).expand(bs, self.in_channel, num_sample)
        seed = self.folding_seed.view(1, 2, num_sample).expand(bs, 2, num_sample).to(x.device)

        x = torch.cat([seed, features], dim=1)
        fd1 = self.folding1(x)
        x = torch.cat([fd1, features], dim=1)
        fd2 = self.folding2(x)

        return fd2

class SimpleRebuildFCLayer(nn.Module):
    def __init__(self, input_dims, step, hidden_dim=512):
        super().__init__()
        self.input_dims = input_dims
        self.step = step
        self.layer = Mlp(self.input_dims, hidden_dim, step * 3)

    def forward(self, rec_feature):
        '''
        Input BNC
        '''
        batch_size = rec_feature.size(0)
        g_feature = rec_feature.max(1)[0]
        token_feature = rec_feature
            
        patch_feature = torch.cat([
                g_feature.unsqueeze(1).expand(-1, token_feature.size(1), -1),
                token_feature
            ], dim = -1)
        rebuild_pc = self.layer(patch_feature).reshape(batch_size, -1, self.step , 3)
        assert rebuild_pc.size(1) == rec_feature.size(1)
        return rebuild_pc

######################################## PCTransformer ########################################   
class PCTransformer_dino_encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        ########## NEW: 添加dino feature提取器#########
        self.dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        ##############################################
        
        ## NEW: 提前引出global feature###
        self.pre_global_feature = config.pre_global_feature
        #################################
        encoder_config = config.encoder_config
        decoder_config = config.decoder_config
        self.center_num  = getattr(config, 'center_num', [512, 128])
        self.encoder_type = config.encoder_type
        assert self.encoder_type in ['graph', 'pn'], f'unexpected encoder_type {self.encoder_type}'
        ##################################
        
        in_chans = 3
        self.num_query = query_num = config.num_query
        global_feature_dim = config.global_feature_dim

        print_log(f'Transformer with config {config}', logger='MODEL')
        # base encoder
        if self.encoder_type == 'graph':
            self.grouper = DGCNN_Grouper(k = 16)
        else:
            self.grouper = SimpleEncoder(k = 32, embed_dims=512)
        self.pos_embed = nn.Sequential(
            nn.Linear(in_chans, 128),
            nn.GELU(),
            nn.Linear(128, encoder_config.embed_dim)
        )  
        self.input_proj = nn.Sequential(
            nn.Linear(self.grouper.num_features, 512),
            nn.GELU(),
            nn.Linear(512, encoder_config.embed_dim)
        )
        # Coarse Level 1 : Encoder
        self.encoder = PointTransformerEncoderEntry(encoder_config)

        self.increase_dim = nn.Sequential(
            nn.Linear(encoder_config.embed_dim, 1024),
            nn.GELU(),
            nn.Linear(1024, global_feature_dim))
        # query generator
        self.coarse_pred = nn.Sequential(
            nn.Linear(global_feature_dim, 1024),
            nn.GELU(),
            nn.Linear(1024, 3 * query_num)
        )
        self.mlp_query = nn.Sequential(
            nn.Linear(global_feature_dim + 3, 1024),
            nn.GELU(),
            nn.Linear(1024, 1024),
            nn.GELU(),
            nn.Linear(1024, decoder_config.embed_dim)
        )
        # assert decoder_config.embed_dim == encoder_config.embed_dim
        if decoder_config.embed_dim == encoder_config.embed_dim:
            self.mem_link = nn.Identity()
        else:
            self.mem_link = nn.Linear(encoder_config.embed_dim, decoder_config.embed_dim)
        # Coarse Level 2 : Decoder
        self.decoder = PointTransformerDecoderEntry(decoder_config)
 
        self.query_ranking = nn.Sequential(
            nn.Linear(3, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

        # NEW: 设置attention计算需要的函数
        # NEW: Cross-Attention between x + pe and dino_feature
        self.cross_attn = nn.MultiheadAttention(embed_dim=encoder_config.embed_dim, num_heads=8, batch_first=True)
        self.attn_norm = nn.LayerNorm(encoder_config.embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def batch_crop_and_resize(self, rgb_batch, mask_batch, output_size=(224, 224), padding=10):
        """
        高效批量裁剪和调整大小 RGB 图像，根据掩码。

        参数:
        - rgb_batch: Tensor, 形状 [batch_size, height, width, 3], dtype=torch.uint8
        - mask_batch: Tensor, 形状 [batch_size, height, width], dtype=torch.uint8
        - output_size: tuple, 输出图像的大小 (height, width)
        - padding: int, 边界框的填充像素数

        返回:
        - resized_rgb: Tensor, 形状 [batch_size, 3, output_size[0], output_size[1]]
        """
        # 确保输入的 batch 是正确的
        # 这里假设 rgb_batch 和 mask_batch 已经传入函数，无需重新赋值
        device = rgb_batch.device
        batch_size, height, width, channels = rgb_batch.shape

        # 确保掩码为二值
        mask_batch = (mask_batch == 0).float()  # 假设 1 表示前景

        # 计算掩码在 y 轴和 x 轴的投影
        mask_any_y = mask_batch.any(dim=2)  # [batch_size, height]
        mask_any_x = mask_batch.any(dim=1)  # [batch_size, width]

        # 计算 y_min 和 y_max
        y_min = mask_any_y.float().argmax(dim=1)  # [batch_size]
        y_max = mask_any_y.flip(dims=[1]).float().argmax(dim=1)
        y_max = height - 1 - y_max  # 反转索引以获得正确的 y_max

        # 计算 x_min 和 x_max
        x_min = mask_any_x.float().argmax(dim=1)  # [batch_size]
        x_max = mask_any_x.flip(dims=[1]).float().argmax(dim=1)
        x_max = width - 1 - x_max  # 反转索引以获得正确的 x_max

        # 处理空掩码的情况：如果掩码全为零，则使用中心裁剪
        mask_sum = mask_batch.view(batch_size, -1).sum(dim=1)
        empty_mask = (mask_sum == 0)

        # 定义中心裁剪的边界
        center_crop_h, center_crop_w = output_size
        y_center = height // 2
        x_center = width // 2

        # 将中心裁剪边界转换为张量
        y_center_tensor = torch.full_like(y_min, y_center, dtype=y_min.dtype, device=device)
        x_center_tensor = torch.full_like(x_min, x_center, dtype=x_min.dtype, device=device)

        # 使用 torch.clamp 进行裁剪，确保传递的是张量
        y_min_center = (y_center_tensor - center_crop_h // 2).clamp(0, height - 1)
        y_max_center = (y_center_tensor + center_crop_h // 2).clamp(0, height - 1)
        x_min_center = (x_center_tensor - center_crop_w // 2).clamp(0, width - 1)
        x_max_center = (x_center_tensor + center_crop_w // 2).clamp(0, width - 1)

        # 将中心裁剪边界应用到空掩码样本
        y_min[empty_mask] = y_min_center[empty_mask]
        y_max[empty_mask] = y_max_center[empty_mask]
        x_min[empty_mask] = x_min_center[empty_mask]
        x_max[empty_mask] = x_max_center[empty_mask]

        # 添加填充
        y_min = (y_min - padding).clamp(0, height - 1)
        y_max = (y_max + padding).clamp(0, height - 1)
        x_min = (x_min - padding).clamp(0, width - 1)
        x_max = (x_max + padding).clamp(0, width - 1)

        # 计算裁剪区域的宽度和高度
        box_width = x_max - x_min + 1  # [batch_size]
        box_height = y_max - y_min + 1  # [batch_size]

        # 防止宽度和高度为零
        box_width = torch.clamp(box_width, min=1)
        box_height = torch.clamp(box_height, min=1)

        # 计算裁剪区域的中心点（归一化坐标）
        center_x = (x_min + x_max) / 2 / (width - 1) * 2 - 1  # [batch_size]
        center_y = (y_min + y_max) / 2 / (height - 1) * 2 - 1  # [batch_size]

        # 计算裁剪区域的归一化宽度和高度
        norm_w = box_width / (width - 1) * 2  # [batch_size]
        norm_h = box_height / (height - 1) * 2  # [batch_size]

        # 创建裁剪网格
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, steps=output_size[0], device=device),
            torch.linspace(-1, 1, steps=output_size[1], device=device)
        )
        grid = torch.stack((grid_x, grid_y), dim=2)  # [H, W, 2]
        grid = grid.unsqueeze(0).repeat(batch_size, 1, 1, 1)  # [batch_size, H, W, 2]

        # 缩放网格
        norm_wh = torch.stack((norm_w, norm_h), dim=1).view(batch_size, 1, 1, 2)  # [batch_size, 1, 1, 2]
        grid = grid * norm_wh / 2  # 由于网格范围为 [-1, 1]

        # 平移网格到中心点
        center_xy = torch.stack((center_x, center_y), dim=1).view(batch_size, 1, 1, 2)  # [batch_size, 1, 1, 2]
        grid = grid + center_xy

        # 限制网格范围在 [-1, 1]
        grid = grid.clamp(-1, 1)

        # 转换 rgb_batch 形状为 [batch_size, 3, height, width] 并归一化到 [0, 1]
        rgb_batch = rgb_batch.permute(0, 3, 1, 2).float() / 255.0  # [B, 3, H, W]

        # 使用 grid_sample 进行裁剪和缩放
        resized_rgb = F.grid_sample(rgb_batch, grid, mode='bilinear', padding_mode='zeros', align_corners=True)

        return resized_rgb
    
    def save_resized_images(self, resized_rgb, save_dir='/home/zhangjinyu/code_repo/pointr/tmp'):
        """
        保存裁剪和调整大小后的 RGB 图像。

        参数:
        - resized_rgb: Tensor, 形状 [batch_size, 3, H, W], 归一化到 [0, 1]
        - save_dir: str, 保存图像的目录
        """
        os.makedirs(save_dir, exist_ok=True)
        resized_rgb = resized_rgb.cpu()  # 转移到 CPU
        for i in range(resized_rgb.shape[0]):
            img = resized_rgb[i].permute(1, 2, 0).numpy()  # [H, W, 3]
            img = (img * 255).astype(np.uint8)
            pil_image = Image.fromarray(img)
            pil_image.save(os.path.join(save_dir, f"resized_rgb_{i}.png"))
            print(f"Saved resized_rgb_{i}.png")

    def forward(self, xyz, rgb, mask):
        bs = xyz.size(0)
        coor, f = self.grouper(xyz, self.center_num) # b n c
        pe =  self.pos_embed(coor)
        x = self.input_proj(f) # b 384
        
        resized_rgb = self.batch_crop_and_resize(rgb, mask, output_size=(224, 224), padding=10)
            
        # 利用dino获取rgb feature 与 x feature计算attention
        dino_feature = self.dinov2_vits14(resized_rgb) # b embed_dim

        ### x + pe 与 dino_feature 进行attention 并归一化 ###
        
        ################################## 方法一 ###############
        # x + pe: (B, N, C)
        x_pe = x + pe  # Shape: (B, N, C)

        # dino_feature: (B, C) -> (B, 1, C)
        dino_feat = dino_feature.unsqueeze(1)  # Shape: (B, 1, C)

        # Perform cross-attention: Query=x_pe, Key=dino_feat, Value=dino_feat
        attn_output, attn_weights = self.cross_attn(query=x_pe, key=dino_feat, value=dino_feat)  # attn_output: (B, N, C)

        # Add the attention output to x_pe (residual connection)
        x_pe = x_pe + attn_output  # Shape: (B, N, C)

        # Apply Layer Normalization
        x_pe = self.attn_norm(x_pe)  # Shape: (B, N, C)

        # Update x to be used in the encoder
        x = x_pe  # Shape: (B, N, C)
        
        ###########################################################

        # Continue with the existing encoder
        x = self.encoder(x, coor) # b n c # 这里encoder就是self attention
        # x = self.encoder(x + pe, coor) # b n c # 这里encoder就是self attention
        global_feature = self.increase_dim(x) # B 1024 N # 提升到global_feature_dim
        global_feature = torch.max(global_feature, dim=1)[0] # B 1024 # max后变为1024维

        coarse = self.coarse_pred(global_feature).reshape(bs, -1, 3) # global feature直接输出coarse points 两层mlp（让我想是否可以mlp出pose mat）

        coarse_inp = misc.fps(xyz, self.num_query//2) # B 128 3
        coarse = torch.cat([coarse, coarse_inp], dim=1) # B 224+128 3

        # mem = self.mem_link(x) # mem来自于encoder输出

        # query selection
        query_ranking = self.query_ranking(coarse) # b n 1 # query ranking是什么？ 线性层+sigmoid
        idx = torch.argsort(query_ranking, dim=1, descending=True) # b n 1
        coarse = torch.gather(coarse, 1, idx[:,:self.num_query].expand(-1, -1, coarse.size(-1))) # [48, 512, 3]
        
        return coarse, global_feature
    
    # 用于debug
    def save_masks(self, mask_batch, save_dir='/home/zhangjinyu/code_repo/pointr/tmp', batch_indices=None):
        """
        保存批量掩码为图像文件。

        参数:
        - mask_batch: Tensor, 形状 [batch_size, height, width], dtype=torch.uint8 或 torch.bool
        - save_dir: str, 保存掩码图像的目录
        - batch_indices: list 或 None, 指定保存哪些样本。如果为 None，则保存全部
        """
        # 确保保存目录存在
        os.makedirs(save_dir, exist_ok=True)
        
        # 将掩码转移到 CPU 并转换为 NumPy 数组
        mask_cpu = mask_batch.cpu().numpy()
        
        # 处理每个样本
        for i in range(mask_cpu.shape[0]):
            if batch_indices is not None and i not in batch_indices:
                continue  # 跳过未指定的样本
            
            mask = mask_cpu[i]  # [height, width]
            
            # 确保掩码是二值的
            unique_values = np.unique(mask)
            if not np.array_equal(unique_values, [0, 1]) and not np.array_equal(unique_values, [0]) and not np.array_equal(unique_values, [1]):
                raise ValueError(f"Mask at index {i} contains values other than 0 and 1: {unique_values}")
            
            # 将掩码从 [0, 1] 转换为 [0, 255] 以便可视化
            mask_image = (mask * 255).astype(np.uint8)
            
            # 创建 PIL Image 对象
            pil_image = Image.fromarray(mask_image, mode='L')  # 'L' 表示灰度图
            
            # 定义保存路径
            filename = f"mask_{i}.png"
            save_path = os.path.join(save_dir, filename)
            
            # 保存图像
            pil_image.save(save_path)
            print(f"Saved mask {i} to {save_path}")

######################################## PoinTr ########################################  

@MODELS.register_module()
class AdaPoinTr_Pose_dino_encoder_mlp(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        
        # NEW: 根据cate_num 指定输出维度
        self.cate_num = config.cate_num
        
        # NEW: 得到rotate loss计算方式
        self.rotate_loss_type = config.pose_config.rotate_loss_type
        self.trans_loss_type = config.pose_config.trans_loss_type
        self.size_loss_type = config.pose_config.size_loss_type
        
        # TODO: 设计如何对应mapping
        # self.mapping = {
        #     '02876657': 0,
        #     '02880940': 1,
        #     '02942699': 2,
        #     '02946921': 3,
        #     '03642806': 4,
        #     '03797390': 5
        # }
        self.mapping = categories_with_labels
        
        self.trans_dim = config.decoder_config.embed_dim
        self.num_query = config.num_query
        self.num_points = getattr(config, 'num_points', None)

        self.decoder_type = config.decoder_type
        assert self.decoder_type in ['fold', 'fc'], f'unexpected decoder_type {self.decoder_type}'

        self.base_model = PCTransformer_dino_encoder(config)
        
        # if self.decoder_type == 'fold':
        #     self.factor = self.fold_step**2
        #     self.decode_head = Fold(self.trans_dim, step=self.fold_step, hidden_dim=256)  # rebuild a cluster point
        # else:
        #     if self.num_points is not None:
        #         self.factor = self.num_points // self.num_query
        #         assert self.num_points % self.num_query == 0
        #         self.decode_head = SimpleRebuildFCLayer(self.trans_dim * 2, step=self.num_points // self.num_query)  # rebuild a cluster point
        #     else:
        #         self.factor = self.fold_step**2
        #         self.decode_head = SimpleRebuildFCLayer(self.trans_dim * 2, step=self.fold_step**2)
        # self.increase_dim = nn.Sequential(
        #         nn.Conv1d(self.trans_dim, 1024, 1),
        #         nn.BatchNorm1d(1024),
        #         nn.LeakyReLU(negative_slope=0.2),
        #         nn.Conv1d(1024, 1024, 1)
        #     )
        self.build_loss_func()
        
        ############# 添加分支预测 rotate trans size ############
        if config.pose_config.normalize_head:
            self.rotat_head = nn.Sequential(
                nn.Linear(1024, 512),
                nn.BatchNorm1d(512),
                nn.GELU(),
                nn.Dropout(p=0.1),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.GELU(),
                nn.Dropout(p=0.1),
                nn.Linear(256, 6 * self.cate_num),
            )
            self.trans_head = nn.Sequential(
                nn.Linear(1024, 512),
                nn.BatchNorm1d(512),
                nn.GELU(),
                nn.Dropout(p=0.1),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.GELU(),
                nn.Dropout(p=0.1),
                nn.Linear(256, 3 * self.cate_num),
            )
            self.size_head = nn.Sequential(
                nn.Linear(1024, 512),
                nn.BatchNorm1d(512),
                nn.GELU(),
                nn.Dropout(p=0.1),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.GELU(),
                nn.Dropout(p=0.1),
                nn.Linear(256, 3 * self.cate_num),
            )
        else:
            self.rotat_head = nn.Sequential(
                nn.Linear(1024, 512),
                nn.GELU(),
                nn.Linear(512, 256),
                nn.GELU(),
                nn.Linear(256, 6*self.cate_num),
            )

            
            self.trans_head = nn.Sequential(
                nn.Linear(1024, 512),
                nn.GELU(),
                nn.Linear(512, 256),
                nn.GELU(),
                nn.Linear(256, 3*self.cate_num),
            )
            
            self.size_head = nn.Sequential(
                nn.Linear(1024, 512),
                nn.GELU(),
                nn.Linear(512, 256),
                nn.GELU(),
                nn.Linear(256, 3*self.cate_num),
            )
        #########################################################
        
        ############# 添加mlp 预测完整点云 ##################
        self.fold_step = 2
        self.mlp_decoder_feature = config.mlp_decoder_config.feature
        if config.mlp_decoder_config.feature == "global_feature":
            # self.mlp_decoder = nn.Sequential(
            #     nn.Linear(1024, 1024),
            #     nn.BatchNorm1d(1024),
            #     nn.ReLU(),
            #     nn.Linear(1024, 1024),
            #     nn.BatchNorm1d(1024),
            #     nn.ReLU(),
            #     nn.Linear(1024, 3 * self.fold_step**2 * self.num_points),
            # )
            pass
        elif config.mlp_decoder_config.feature == "rebuild_feature":
            self.decode_head = SimpleRebuildFCLayer(self.trans_dim * 2, step=self.fold_step**2)  # rebuild a cluster point
            self.reduce_map = nn.Linear(1027, self.trans_dim)
        else:
            raise NotImplementedError
        #####################################################
        

    def build_loss_func(self):
        self.loss_func = ChamferDistanceL1()

    def get_loss(self, ret, gt, gt_taxonomys, gt_rotat_mat, gt_trans_mat, gt_size_mat, epoch=1):
        
        # pred_coarse, denoised_coarse, denoised_fine, pred_fine, pred_rotat_mat, pred_trans_mat, pred_size_mat = ret
        pred_coarse, pred_dense, pred_rotat_mat, pred_trans_mat, pred_size_mat = ret

        # recon loss
        loss_dense = self.loss_func(pred_dense, gt)
        loss_coarse = self.loss_func(pred_coarse, gt)
        
        # pose loss
        # TODO: 检查正确性
        gt_cate_ids = torch.tensor([self.mapping[tax] for tax in gt_taxonomys]).cuda()
        index = gt_cate_ids.squeeze() + torch.arange(gt.shape[0], dtype=torch.long).cuda() * self.cate_num
        pred_trans_mat = pred_trans_mat.view(-1, 3).contiguous() # bs, 3*nc -> bs*nc, 3
        pred_trans_mat = torch.index_select(pred_trans_mat, 0, index).contiguous()  # bs x 3
        pred_size_mat = pred_size_mat.view(-1, 3).contiguous() # bs, 3*nc -> bs*nc, 3
        pred_size_mat = torch.index_select(pred_size_mat, 0, index).contiguous()  # bs x 3
        pred_rotat_mat = pred_rotat_mat.view(-1, 6).contiguous() # bs, 6*nc -> bs*nc, 6
        pred_rotat_mat = torch.index_select(pred_rotat_mat, 0, index).contiguous()  # bs x 6
        
        minloss_gt_rotat_mat_list = []
        if self.rotate_loss_type == 'l1':
            loss_fn =nn.SmoothL1Loss()
            for gt_mat_list_6d, pred_mat_6d in zip(gt_rotat_mat, pred_rotat_mat):
                losses = torch.tensor([loss_fn(pred_mat_6d, gt_mat) for gt_mat in gt_mat_list_6d])
                idx = torch.argmin(losses)
                minloss_gt_rotat_mat_list.append(gt_mat_list_6d[idx])

            minloss_gt_rotat_mat_stack = torch.stack(minloss_gt_rotat_mat_list)
        elif self.rotate_loss_type == 'l2':
            loss_fn = nn.MSELoss()
            for gt_mat_list_6d, pred_mat_6d in zip(gt_rotat_mat, pred_rotat_mat):
                losses = torch.tensor([loss_fn(pred_mat_6d, gt_mat) for gt_mat in gt_mat_list_6d])
                idx = torch.argmin(losses)
                minloss_gt_rotat_mat_list.append(gt_mat_list_6d[idx])

            minloss_gt_rotat_mat_stack = torch.stack(minloss_gt_rotat_mat_list)
        elif self.rotate_loss_type == 'geodesic':
            loss_fn = geodesic_rotation_error
            for gt_mat_list_6d, pred_mat_6d in zip(gt_rotat_mat, pred_rotat_mat):
                gt_mat_list = convert_rotation.compute_rotation_matrix_from_ortho6d(gt_mat_list_6d)
                pred_mat = convert_rotation.single_rotation_matrix_from_ortho6d(pred_mat_6d)
                losses = torch.tensor([loss_fn(pred_mat, gt_mat)[1] for gt_mat in gt_mat_list])
                idx = torch.argmin(losses)
                minloss_gt_rotat_mat_list.append(gt_mat_list_6d[idx])

            minloss_gt_rotat_mat_stack = torch.stack(minloss_gt_rotat_mat_list)
            
        if self.rotate_loss_type == 'l1':
            loss_rotat = nn.SmoothL1Loss()(minloss_gt_rotat_mat_stack, pred_rotat_mat)
        elif self.rotate_loss_type == 'l2':
            loss_rotat = nn.MSELoss()(minloss_gt_rotat_mat_stack, pred_rotat_mat)
        elif self.rotate_loss_type == 'geodesic':
            r1, r2 = convert_rotation.compute_rotation_matrix_from_ortho6d(minloss_gt_rotat_mat_stack), convert_rotation.compute_rotation_matrix_from_ortho6d(pred_rotat_mat) 
            loss_rotat, loss_rotat_all = geodesic_rotation_error(r1, r2)
        
        if self.rotate_loss_type == 'l1':
            loss_trans = nn.SmoothL1Loss()(pred_trans_mat, gt_trans_mat)
        elif self.rotate_loss_type == 'l2':
            loss_trans = nn.MSELoss()(pred_trans_mat, gt_trans_mat)

        if self.rotate_loss_type == 'l1':
            loss_size = nn.SmoothL1Loss()(pred_size_mat, gt_size_mat)
        elif self.rotate_loss_type == 'l2':
            loss_size = nn.MSELoss()(pred_size_mat, gt_size_mat)

        return loss_coarse, loss_dense, loss_rotat, loss_trans, loss_size

    def forward(self, xyz, rgb, mask):
        coarse_point_cloud, global_feature = self.base_model(xyz, rgb, mask) # B M C and B M 3
    
        ############## 添加mlp 预测完整点云 ############
        B, M ,_ = coarse_point_cloud.shape
        if self.mlp_decoder_feature == "global_feature":
            pass
        elif self.mlp_decoder_feature == "rebuild_feature":
            rebuild_feature = torch.cat(
                [coarse_point_cloud, global_feature.unsqueeze(-2).expand(-1, M, -1)], dim=-1
            )
            rebuild_feature = self.reduce_map(rebuild_feature) # B M C
            relative_xyz = self.decode_head(rebuild_feature)   # B M S 3
            rebuild_points = (relative_xyz + coarse_point_cloud.unsqueeze(-2))  # B M S 3
        else:
            raise NotImplementedError
        dense_point_cloud = rebuild_points.reshape(B, -1, 3).contiguous()  # B M 3
        ########################################
        
        ############# 添加分支预测 rotate trans size ############
        rotation_mat = self.rotat_head(global_feature)
        trans_mat = self.trans_head(global_feature)
        size_mat = self.size_head(global_feature)
        
        ########################################################

        ret = (coarse_point_cloud, dense_point_cloud, rotation_mat, trans_mat, size_mat)
        return ret