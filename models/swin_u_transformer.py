
"""
# Code from https://github.com/HuCaoFighting/Swin-Unet
# Code from https://github.com/CAMTL/CA-MTL
# Code from VTAGML
"""

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math

from einops import rearrange

#from .swin_transformer import SwinTransformerBlock, BasicLayer, SwinTransformer
#from utils import logger
#logger = logging.get_logger("swin-Unet")


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)

    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)

    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        # pad to multiple of window size
        H, W = self.input_resolution
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        H += pad_b
        W += pad_r

        if pad_r == 0 and pad_b == 0:
            self.pad_size = None
        else:
            self.pad_size = (pad_b, pad_r)


        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            #H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape

        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # pad to the multiple of window size
        OH, OW = H, W
        if self.pad_size:
            import torch.nn.functional as F
            pad_l, pad_t, pad_b, pad_r = 0, 0, self.pad_size[0], self.pad_size[1]
            x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
            H = OH + pad_b
            W = OW + pad_r

        # cyclic shift
        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        # un-pad
        if self.pad_size:
            x = x[:, :OH, :OW, :].contiguous()

        #x = x.view(B, H * W, C)
        x = x.view(B, OH * OW, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)


    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops


class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution

        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, f"input feature has wrong size L : {L}, H : {H}, W : {W}"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C // 4)
        x = x.view(B, -1, C // 4)
        x = self.norm(x)

        return x


class FinalPatchExpand_X4(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, 16 * dim, bias=False)
        self.output_dim = dim
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale,
                      c=C // (self.dim_scale ** 2))
        x = x.view(B, -1, self.output_dim)
        x = self.norm(x)

        return x


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self,  dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 block_module=SwinTransformerBlock,
                 num_prompts=None,
                 cfg=None,):

        super().__init__()
        self.cfg = cfg
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.num_prompts = num_prompts

        self.tasks = cfg.TASKS

        # build blocks
        if num_prompts is not None:
            self.blocks = nn.ModuleList([
                block_module(num_prompts,
                dim=dim, input_resolution=input_resolution,
                                     num_heads=num_heads, window_size=window_size,
                                     shift_size=0 if (i % 2 == 0) else window_size // 2,
                                     mlp_ratio=mlp_ratio,
                                     qkv_bias=qkv_bias, qk_scale=qk_scale,
                                     drop=drop, attn_drop=attn_drop,
                                     drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                     norm_layer=norm_layer)
                for i in range(depth)])
            #self.num_prompts = num_prompts

        else:
            self.blocks = nn.ModuleList([
                block_module(
                    dim=dim, input_resolution=input_resolution,
                    num_heads=num_heads, window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop, attn_drop=attn_drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,  # noqa
                    norm_layer=norm_layer)
                for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            if num_prompts is None:
                self.downsample = downsample(
                    input_resolution, dim=dim, norm_layer=norm_layer
                )
            else:
                self.downsample = downsample(
                    num_prompts,
                    input_resolution, dim=dim, norm_layer=norm_layer
                )
        else:
            self.downsample = None

    def forward(self, x, mtl_prompt_embd=None, spa_prompt = None):
        if mtl_prompt_embd is None:
            for blk in self.blocks:
                if self.use_checkpoint:
                    x = checkpoint.checkpoint(blk, x)
                else:
                    x = blk(x)
            if self.downsample is not None:
                x = self.downsample(x)

            return x

        elif mtl_prompt_embd is not None and spa_prompt is not None:
            B = len(x)
            num_blocks = len(self.blocks)

            for i in range(num_blocks - 1):
                prompt_emb = mtl_prompt_embd.expand(B, -1, -1)
                x = torch.cat((prompt_emb, x), dim=1)
                x, prompt_emb = self.blocks[i](x)

            out = {}
            x_task = {}
            x_ori = x

            x = torch.cat((prompt_emb, x_ori), dim=1)
            x, prompt_emb = self.blocks[-1](x)
            if self.downsample is not None:
                x = torch.cat((prompt_emb, x), dim=1)
                x, prompt_emb = self.downsample(x)

            spa_prompt_emb = {}
            for task in self.tasks:
                spa_prompt_emb[task] = spa_prompt[task].expand(B, -1, -1)

                x_task[task] = torch.cat((spa_prompt_emb[task], x_ori), dim=1)
                out[task], spa_prompt_emb[task] = self.blocks[-1](x_task[task])

                if self.downsample is not None:
                    out[task] = torch.cat((spa_prompt_emb[task], out[task]), dim=1)
                    out[task], spa_prompt_emb[task] = self.downsample(out[task])
                    out[task] = out[task] + x

            return x, out, prompt_emb, spa_prompt_emb

        else:
            B = len(x)
            num_blocks = len(self.blocks)

            if self.cfg.MODEL.MTLPROMPT.PROMPT.SHARED.TYPE == "DEEP":
                for i in range(num_blocks):
                    prompt_emb = mtl_prompt_embd[i].expand(B, -1, -1)
                    x = torch.cat((prompt_emb, x), dim=1)
                    x, _ = self.blocks[i](x)
            elif self.cfg.MODEL.MTLPROMPT.PROMPT.SHARED.TYPE == "SHALLOW":
                for i in range(num_blocks):
                    prompt_emb = mtl_prompt_embd.expand(B, -1, -1)
                    x = torch.cat((prompt_emb, x), dim=1)
                    x, prompt_emb = self.blocks[i](x)

            if self.downsample is not None:
                x = torch.cat((prompt_emb, x), dim=1)
                x, prompt_emb = self.downsample(x)

            return x, prompt_emb


    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class BasicLayer_up(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        upsample (nn.Module | None, optional): upsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, upsample=None, use_checkpoint=False,
                 block_module=SwinTransformerBlock,
                 num_prompts=None,
                 cfg=None,
                 ):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.num_prompts = num_prompts

        # build blocks
        if num_prompts is not None:
            self.blocks = nn.ModuleList([
                block_module(dim=dim, input_resolution=input_resolution,
                             num_heads=num_heads, window_size=window_size,
                             shift_size=0 if (i % 2 == 0) else window_size // 2,
                             mlp_ratio=mlp_ratio,
                             qkv_bias=qkv_bias, qk_scale=qk_scale,
                             drop=drop, attn_drop=attn_drop,
                             drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                             norm_layer=norm_layer,
                             num_prompts=num_prompts,
                             cfg=cfg)
                for i in range(depth)])

        else:
            self.blocks = nn.ModuleList([
                block_module(dim=dim, input_resolution=input_resolution,
                                     num_heads=num_heads, window_size=window_size,
                                     shift_size=0 if (i % 2 == 0) else window_size // 2,
                                     mlp_ratio=mlp_ratio,
                                     qkv_bias=qkv_bias, qk_scale=qk_scale,
                                     drop=drop, attn_drop=attn_drop,
                                     drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                     norm_layer=norm_layer)
                for i in range(depth)])

        # patch merging layer
        if upsample is not None:
            if num_prompts is None:
                self.upsample = upsample(input_resolution, dim=dim, dim_scale=2, norm_layer=norm_layer)
            else:
                self.upsample = upsample(num_prompts, input_resolution, dim=dim, dim_scale=2, norm_layer=norm_layer)
        else:
            self.upsample = None

    def forward(self, x, spa_prompt_embd=None, chan_prompt_embd=None):
        if spa_prompt_embd is None and chan_prompt_embd is None :
            for blk in self.blocks:
                if self.use_checkpoint:
                    x = checkpoint.checkpoint(blk, x)
                else:
                    x = blk(x)
                if self.upsample is not None:
                    x = self.upsample(x)

                return x

        elif spa_prompt_embd is not None:
            B = x.shape[0]
            num_blocks = len(self.blocks)

            for i in range(num_blocks):
                prompt_embd = spa_prompt_embd.expand(B, -1, -1)

                x = torch.cat((prompt_embd, x), dim=1)
                x, prompt_embd = self.blocks[i](x)

            if self.upsample is not None:
                x = self.upsample(x)

            return x

        elif chan_prompt_embd is not None:
            B = x.shape[0]
            num_blocks = len(self.blocks)

            for i in range(num_blocks):
                prompt_embd = chan_prompt_embd.expand(B, -1, -1)

                x, prompt_embd = self.blocks[i](x, prompt_embd)

            if self.upsample is not None:
                x = self.upsample(x)

            return x



class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # proj(x): B C Ph Pw -> flatten(2) B C Ph*Pw -> transpose(1,2) B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x  # B Ph*Pw C

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


# TODO
class Decoder_MTLdSwinUTransformer(nn.Module):
    def __init__(self, cfg, img_size=224, patch_size=4, in_chans=3,
                 embed_dim=96, depths=[2, 2, 18, 2], depths_decoder=[2, 2, 2, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, final_upsample="expand_first", task_classes=[14, 1], tasks=None,
                 decoder_type=None,
                 device='cuda', **kwargs):

        super().__init__()

        self.cfg = cfg

        # TODO : Attribute  정
        self.num_layers = len(depths_decoder)  # len(depth) stage encoder, default = 4
        self.embed_dim = embed_dim  # # of channels in hidden layer in first stage
        self.ape = ape  # default : False, absolute patch embedding
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))  # 전체 stage 후 encoder output channel dimensions
        self.mlp_ratio = mlp_ratio
        self.final_upsample = final_upsample  # setting to expand the last layer
        self.decoder_type = decoder_type
        self.depths_decoder = depths_decoder

        # task info
        self.task_classes = task_classes  # List of number of prediction classes for each tasks
        self.tasks = tasks

        self.device = device

        # assert len(task_classes) == len(tasks), "number of tasks must match the number of number of classes"

        self.norm_up = norm_layer(self.embed_dim)

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)  # Output : B Ph*Pw C
        num_patches = self.patch_embed.num_patches
        self.num_patches = num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution
        # self.patch_grid = self.patch_embed.patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)



        dpr_decoder = [x.item() for x in
                       torch.linspace(0, drop_path_rate, sum(depths_decoder))]  # stochastic depth decay rule
        dpr_encoder = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        """ Decoder Module """
        self.decoder_layers_layers_up = nn.ModuleDict()  # decoder layer 에서 upsample Transformer
        # self.decoder_layers_concat_back_dim = nn.ModuleList()  #
        self.decoder_layers_norm_up = nn.ModuleList()  # decoder layer 에서 upsampling 하고 normalize
        self.decoder_layers_up = nn.ModuleDict()  # Final up sample 만 포함?

        for task in tasks:
            up = FinalPatchExpand_X4(input_resolution=(
                img_size[0] // patch_size, img_size[1] // patch_size), dim_scale=4, dim=embed_dim)

            self.decoder_layers_up[task] = up

            i_layer = 2
            for task in tasks:
                layers_up = BasicLayer_up(dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                                          input_resolution=(
                                              self.patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                                              self.patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer))),
                                          depth=depths_decoder[(self.num_layers - 1 - i_layer)],
                                          num_heads=num_heads[(self.num_layers - 1 - i_layer)],
                                          window_size=window_size,
                                          mlp_ratio=self.mlp_ratio,
                                          qkv_bias=qkv_bias, qk_scale=qk_scale,
                                          drop=drop_rate, attn_drop=attn_drop_rate,
                                          drop_path=dpr_decoder[
                                                    sum(depths_decoder[:(self.num_layers - 1 - i_layer)]):sum(
                                                        depths_decoder[:(self.num_layers - 1 - i_layer) + 1])],
                                          norm_layer=norm_layer,
                                          upsample=PatchExpand if (i_layer < self.num_layers - 1) else None,
                                          use_checkpoint=use_checkpoint)

                self.decoder_layers_layers_up[task] = layers_up

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    # Decoder
    def forward_up_features(self, x, layers_up, concat_back_dim=None,
                            norm_up=None, spa_prompt=None):  # PatchExpand, BasicLayerup

        # TODO : Channel and spatial prompts
        if self.cfg.MODEL.MTLPROMPT.DECODER_TYPE == "firstblocksep":
            decoder_output = {}
            for task in self.tasks:
                if self.cfg.MODEL.MTLPROMPT.PROMPT.SPATIAL.ENABLED:
                    if self.cfg.MODEL.MTLPROMPT.PROMPT.SPATIAL.TYPE == 'SHALLOW':
                        decoder_output[task] = layers_up[task](x, spa_prompt)
                else:
                    decoder_output[task] = layers_up[task](x)

                decoder_output[task] = self.norm_up(decoder_output[task])

            return decoder_output
        else:
            inx = 0
            for layer_up in layers_up:

                if inx == 0:
                    x = layer_up(x)
                    inx = inx + 1
                    mtl_prompt_embd = self.prompt_upsampler(mtl_prompt_embd)

                else:
                    x = torch.cat([x, x_downsample[3 - inx]], -1)
                    x = self.decoder_layers_concat_back_dim[inx](x)

                    if self.cfg.prompts_dictionary.skip_connection:
                        mtl_prompt_embd = torch.cat([mtl_prompt_embd, prompt_downsample[3 - inx]], -1)
                        # print(mtl_prompt_embd.shape)
                        mtl_prompt_embd = self.decoder_layers_concat_back_dim[inx](mtl_prompt_embd)
                        # print(mtl_prompt_embd.shape)

                    mtl_prompt_embd = self.prompt_dropout(mtl_prompt_embd)
                    x, mtl_prompt_embd = layer_up(x, mtl_prompt_embd)
                    inx = inx + 1

            x = x[:, self.all_prompt_len:, :]
            x = self.norm_up(x)  # B L C B*task , 224 /4 *224/4 , 96

            return x

    def forward(self, x, spa_prompt=None):  # x : B, C, H, W // task_id : B * Task

        if self.cfg.MODEL.MTLPROMPT.DECODER_TYPE == "firstblocksep":
            outputs = {}
            layers_up = self.decoder_layers_layers_up  # PatchExpand, BasicLayerup

            concat_back_dim = None
            norm_up = self.decoder_layers_norm_up
            up = self.decoder_layers_up  # Final layer up

            x = self.forward_up_features(x, layers_up, concat_back_dim, norm_up, spa_prompt)

            for task in self.tasks:
                decoder_output = x[task]
                outputs[task] = self.up_x4(decoder_output, up[task])  # ([4, 96, 224, 224])

        return outputs

    def up_x4(self, x, up):
        H, W = self.patches_resolution  # 56, 56

        B, L, C = x.shape  # torch.Size([6, 3146, 128])
        assert L == H * W, "input features has wrong size"

        if self.final_upsample == "expand_first":
            x = up(x)
            x = x.view(B, 4 * H, 4 * W, -1)
            x = x.permute(0, 3, 1, 2)  # B,C,H,W

        return x




class MTLSwinUNet(nn.Module):
    """
    (Shared) Swin Encoder <- Freeze
    (Shared | Seperated) Swin Decoder <- Unfreeze
    (Task-specific) Task heads <- Unfreeze
    """
    def __init__(self,
                 img_size=224, patch_size=4, in_chans=3,
                 embed_dim=96, depths=[2, 2, 18, 2], depths_decoder=[2, 2, 2, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, final_upsample="expand_first", task_classes = [14, 1], tasks=None,
                 decoder_type = None,
                 device='cuda',
                 cfg=None,
                     **kwargs):
        super().__init__()

        self.cfg = cfg

        self.num_layers = len(depths)  # len(depth) stage encoder, default = 4
        self.embed_dim = embed_dim # # of channels in hidden layer in first stage
        self.ape = ape # default : False, absolute patch embedding
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))  # 전체 stage 후 encoder output channel dimensions
        self.mlp_ratio = mlp_ratio
        self.final_upsample = final_upsample # setting to expand the last layer
        self.decoder_type = decoder_type
        self.depths_decoder = depths_decoder

        # task info
        self.task_classes = task_classes # List of number of prediction classes for each tasks
        self.tasks = tasks

        self.device = device

        #assert len(task_classes) == len(tasks), "number of tasks must match the number of number of classes"

        self.norm_up = norm_layer(self.embed_dim)


        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)   # Output : B Ph*Pw C
        num_patches = self.patch_embed.num_patches
        self.num_patches = num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution
        #self.patch_grid = self.patch_embed.patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr_encoder = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        #dpr_decoder = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths_decoder))]  # stochastic depth decay rule

        #self.task_id_2_task_idx = {i: i for i, t in enumerate(tasks)}  # {0:0, 1:1}

        """ Encoder Module """

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):  # self.num_layrs = 4, each stage
            layer = BasicLayer(cfg = cfg,
                                dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr_encoder[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint)
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)  # num_features = 32C

        # """ Decoder Module """
        #
        # if cfg.MODEL.MTLPROMPT.DECODER_TYPE == "share":
        #     self.decoder_layers_layers_up = nn.ModuleList()  # decoder layer 에서 upsample Transformer
        #     self.decoder_layers_concat_back_dim = nn.ModuleList()  #
        #     self.decoder_layers_norm_up = nn.ModuleList()  # decoder layer 에서 upsampling 하고 normalize
        #     self.decoder_layers_up = nn.ModuleList()  # Final up sample 만 포함?
        #     # self.decoder_layers_output = nn.ModuleDict()  # Task specific heads
        #
        #     # Shared decoder, specific task heads <- VTAGML 이랑 다른 부분
        #     for i_layer in range(self.num_layers):
        #         concat_linear = nn.Linear(2 * int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
        #                                   int(embed_dim * 2 ** (self.num_layers - 1 - i_layer))) if i_layer > 0 else nn.Identity()
        #         if i_layer == 0:
        #             layer_up = PatchExpand(
        #                 input_resolution=(patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
        #                                   patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer))),
        #                 dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)), dim_scale=2, norm_layer=norm_layer)
        #         else:
        #
        #             layer_up = BasicLayer_up(dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
        #                                      input_resolution=(
        #                                      patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
        #                                      patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer))),
        #                                      depth=depths_decoder[(self.num_layers - 1 - i_layer)],
        #                                      num_heads=num_heads[(self.num_layers - 1 - i_layer)],
        #                                      window_size=window_size,
        #                                      mlp_ratio=self.mlp_ratio,
        #                                      qkv_bias=qkv_bias, qk_scale=qk_scale,
        #                                      drop=drop_rate, attn_drop=attn_drop_rate,
        #                                      drop_path=dpr_decoder[sum(depths_decoder[:(self.num_layers - 1 - i_layer)]):sum(
        #                                          depths_decoder[:(self.num_layers - 1 - i_layer) + 1])],
        #                                      norm_layer=norm_layer,
        #                                      upsample=PatchExpand if (i_layer < self.num_layers - 1) else None,
        #                                      use_checkpoint=use_checkpoint)
        #         # 이 부분 수정
        #         #self.layers_up.append(layer_up)
        #         #self.concat_back_dim.append(concat_linear)
        #         self.decoder_layers_layers_up.append(layer_up)
        #         self.decoder_layers_concat_back_dim.append(concat_linear)
        #
        #     # self.decoder_layers_layers_up.append(self.layers_up)
        #     # self.decoder_layers_concat_back_dim.append(self.concat_back_dim)
        #     self.decoder_layers_norm_up.append(norm_layer(self.embed_dim))
        #
        #     if self.final_upsample == "expand_first":
        #         up = FinalPatchExpand_X4(input_resolution=(
        #             img_size[0] // patch_size, img_size[1] // patch_size), dim_scale=4, dim=embed_dim)
        #
        #         self.decoder_layers_up.append(up)
        #
        #
        #     # for task in tasks:
        #     #     # Output the correct number of channels
        #     #     dec_output = nn.Conv2d(in_channels=embed_dim, out_channels=self.task_classes[task], kernel_size=1, bias=False)
        #     #
        #     #     #self.decoder_layers_output.append(dec_output)
        #     #     self.decoder_layers_output[task] = dec_output
        #
        # elif cfg.MODEL.MTLPROMPT.DECODER_TYPE == "sep_last":
        #     self.decoder_layers_layers_up = nn.ModuleList()  # decoder layer 에서 upsample Transformer
        #     self.decoder_layers_concat_back_dim = nn.ModuleList()  #
        #     self.decoder_layers_norm_up = nn.ModuleList()  # decoder layer 에서 upsampling 하고 normalize
        #     self.decoder_layers_up = nn.ModuleList()  # Final up sample 만 포함?
        #
        #     decoder_last_layers_up = nn.ModuleDict()
        #     # self.decoder_layers_output = nn.ModuleDict()  # Task specific heads
        #
        #     # Shared decoder, specific task heads <- VTAGML 이랑 다른 부분
        #     for i_layer in range(self.num_layers-1):
        #         concat_linear = nn.Linear(2 * int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
        #                                   int(embed_dim * 2 ** (
        #                                               self.num_layers - 1 - i_layer))) if i_layer > 0 else nn.Identity()
        #         if i_layer == 0:
        #             layer_up = PatchExpand(
        #                 input_resolution=(patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
        #                                   patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer))),
        #                 dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)), dim_scale=2, norm_layer=norm_layer)
        #         else:
        #             layer_up = BasicLayer_up(dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
        #                                      input_resolution=(
        #                                          patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
        #                                          patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer))),
        #                                      depth=depths_decoder[(self.num_layers - 1 - i_layer)],
        #                                      num_heads=num_heads[(self.num_layers - 1 - i_layer)],
        #                                      window_size=window_size,
        #                                      mlp_ratio=self.mlp_ratio,
        #                                      qkv_bias=qkv_bias, qk_scale=qk_scale,
        #                                      drop=drop_rate, attn_drop=attn_drop_rate,
        #                                      drop_path=dpr_decoder[
        #                                                sum(depths_decoder[:(self.num_layers - 1 - i_layer)]):sum(
        #                                                    depths_decoder[:(self.num_layers - 1 - i_layer) + 1])],
        #                                      norm_layer=norm_layer,
        #                                      upsample=PatchExpand if (i_layer < self.num_layers - 1) else None,
        #                                      use_checkpoint=use_checkpoint)
        #         # 이 부분 수정
        #         # self.layers_up.append(layer_up)
        #         # self.concat_back_dim.append(concat_linear)
        #         self.decoder_layers_layers_up.append(layer_up)
        #         self.decoder_layers_concat_back_dim.append(concat_linear)
        #
        #     # self.decoder_layers_layers_up.append(self.layers_up)
        #     # self.decoder_layers_concat_back_dim.append(self.concat_back_dim)
        #     self.decoder_layers_norm_up.append(norm_layer(self.embed_dim))
        #
        #     if self.final_upsample == "expand_first":
        #
        #         up = FinalPatchExpand_X4(input_resolution=(
        #             img_size // patch_size, img_size // patch_size), dim_scale=4, dim=embed_dim)
        #
        #         self.decoder_layers_up.append(up)
        #
        #
        #     i_layer = 3
        #     for task in tasks:
        #         last_layers_up = BasicLayer_up(dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
        #                                      input_resolution=(
        #                                          patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
        #                                          patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer))),
        #                                      depth=depths_decoder[(self.num_layers - 1 - i_layer)],
        #                                      num_heads=num_heads[(self.num_layers - 1 - i_layer)],
        #                                      window_size=window_size,
        #                                      mlp_ratio=self.mlp_ratio,
        #                                      qkv_bias=qkv_bias, qk_scale=qk_scale,
        #                                      drop=drop_rate, attn_drop=attn_drop_rate,
        #                                      drop_path=dpr_decoder[
        #                                                sum(depths_decoder[:(self.num_layers - 1 - i_layer)]):sum(
        #                                                    depths_decoder[:(self.num_layers - 1 - i_layer) + 1])],
        #                                      norm_layer=norm_layer,
        #                                      upsample=PatchExpand if (i_layer < self.num_layers - 1) else None,
        #                                      use_checkpoint=use_checkpoint)
        #
        #         decoder_last_layers_up[task] = last_layers_up
        #     self.decoder_layers_layers_up.append(decoder_last_layers_up)
        #
        #     concat_linear = nn.Linear(2 * int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
        #                               int(embed_dim * 2 ** (
        #                                       self.num_layers - 1 - i_layer))) if i_layer > 0 else nn.Identity()
        #     self.decoder_layers_concat_back_dim.append(concat_linear)
        #
        #
        # else:
        #     self.decoder_layers_layers_up = nn.ModuleDict()  # decoder layer 에서 upsample Transformer
        #     self.decoder_layers_concat_back_dim = nn.ModuleDict()  #
        #     self.decoder_layers_norm_up = nn.ModuleDict()  # decoder layer 에서 upsampling 하고 normalize
        #     self.decoder_layers_up = nn.ModuleDict()  # Final up sample 만 포함?
        #
        #     for task in tasks:
        #
        #         # self.decoder_layers_output = nn.ModuleDict()  # Task specific heads
        #
        #         layers_up = nn.ModuleList()
        #         concat_back_dim = nn.ModuleList()
        #
        #         for i_layer in range(self.num_layers):
        #             concat_linear = nn.Linear(2 * int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
        #                                       int(embed_dim * 2 ** (
        #                                                   self.num_layers - 1 - i_layer))) if i_layer > 0 else nn.Identity()
        #             if i_layer == 0:
        #                 layer_up = PatchExpand(
        #                     input_resolution=(patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
        #                                       patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer))),
        #                     dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)), dim_scale=2,
        #                     norm_layer=norm_layer)
        #             else:
        #                 layer_up = BasicLayer_up(dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
        #                                          input_resolution=(
        #                                              patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
        #                                              patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer))),
        #                                          depth=depths_decoder[(self.num_layers - 1 - i_layer)],
        #                                          num_heads=num_heads[(self.num_layers - 1 - i_layer)],
        #                                          window_size=window_size,
        #                                          mlp_ratio=self.mlp_ratio,
        #                                          qkv_bias=qkv_bias, qk_scale=qk_scale,
        #                                          drop=drop_rate, attn_drop=attn_drop_rate,
        #                                          drop_path=dpr_decoder[sum(depths[:(self.num_layers - 1 - i_layer)]):sum(
        #                                              depths[:(self.num_layers - 1 - i_layer) + 1])],
        #                                          norm_layer=norm_layer,
        #                                          upsample=PatchExpand if (i_layer < self.num_layers - 1) else None,
        #                                          use_checkpoint=use_checkpoint)
        #
        #             layers_up.append(layer_up)
        #             concat_back_dim.append(concat_linear)
        #         # 이 부분 수정
        #         self.decoder_layers_layers_up[task] = layers_up
        #         self.decoder_layers_concat_back_dim[task] = concat_back_dim
        #         self.decoder_layers_norm_up[task] = norm_layer(self.embed_dim)
        #
        #         if self.final_upsample == "expand_first":
        #             up = FinalPatchExpand_X4(input_resolution=(
        #                 img_size[0] // patch_size, img_size[1] // patch_size), dim_scale=4, dim=embed_dim)
        #
        #             self.decoder_layers_up[task] = up

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    # Encoder and Bottleneck
    def forward_features(self, x):  # x : B*Task, C=3, H, W
        # print(f"x.shape : {x.shape}")  4*2, 3, 224, 224
        x = self.patch_embed(x)
        # print(f"x.shape : {x.shape}") B*Task Ph*Pw(56 * 56) C=96
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        x_downsample = []

        # Encoder layers (self.layers : encoder layer ModuleList)
        for layer in self.layers:
            x_downsample.append(x)
            x = layer(x)
        x = self.norm(x)  # B L C

        return x, x_downsample

    # Decoder and Skip connection   x, x_downsample, layers_up, concat_back_dim, norm_up
    def forward_up_features(self, x, x_downsample, layers_up, concat_back_dim=None, norm_up = None):  # PatchExpand, BasicLayerup

        if self.cfg.MODEL.MTLPROMPT.DECODER_TYPE == 'share':
            for inx, layer_up in enumerate(layers_up):

                if inx == 0:
                    x = layer_up(x)
                else:
                    x = torch.cat([x, x_downsample[3 - inx]], -1)
                    x = self.decoder_layers_concat_back_dim[inx](x)
                    x = layer_up(x)
            #print(f"forward_up_features x : {x.shape}") ([8, 3136, 96])  B*task , 224 /4 *224/4 , 96
            x = self.norm_up(x)  # B L C B*task , 224 /4 *224/4 , 96

        elif self.cfg.MODEL.MTLPROMPT.DECODER_TYPE == 'sep_last':

            for inx, layer_up in enumerate(layers_up):

                if inx == 0:
                    x = layer_up(x)
                elif inx < 3:
                    x = torch.cat([x, x_downsample[3 - inx]], -1)
                    x = self.decoder_layers_concat_back_dim[inx](x)
                    x = layer_up(x)

                else:
                    decoder_output={}
                    for task in self.tasks:
                        x = torch.cat([x, x_downsample[3 - inx]], -1)
                        x = self.decoder_layers_concat_back_dim[inx](x)
                        decoder_output[task] =  layer_up[task](x)

            #print(f"forward_up_features x : {x.shape}") ([8, 3136, 96])  B*task , 224 /4 *224/4 , 96
            for task in self.tasks:
                decoder_output[task] = self.norm_up(decoder_output[task])
            return decoder_output

        else:
            for inx, layer_up in enumerate(layers_up):
                if inx == 0:
                    x = layer_up(x)

                else:
                    x = torch.cat([x, x_downsample[3 - inx]], -1)
                    x = concat_back_dim[inx](x)
                    x = layer_up(x)
            x = norm_up(x)

        return x

    def up_x4(self, x, up):

        H, W = self.patches_resolution
        B, L, C = x.shape
        assert L == H * W, "input features has wrong size"

        if self.cfg.MODEL.MTLPROMPT.DECODER_TYPE == "sep_last" or self.cfg.MODEL.MTLPROMPT.DECODER_TYPE == "share":

            if self.final_upsample == "expand_first":
                for i, layer in enumerate(up):
                    #x = up(x)
                    x = layer(x)
                    x = x.view(B, 4 * H, 4 * W, -1)
                    x = x.permute(0, 3, 1, 2)  # B,C,H,W

        else:
            if self.final_upsample == "expand_first":
                x = up(x)
                x = x.view(B, 4 * H, 4 * W, -1)  ## 4*H 4* W
                x = x.permute(0, 3, 1, 2)  # B,C,H,W

        return x

    def forward(self, x):   # x : B*Task, C, H, W // task_id : B *

        # Forward through the encoder layers
        x, x_downsample = self.forward_features(x)  # x : B*Task, C, H, W

        # if self.cfg.MODEL.MTLPROMPT.DECODER_TYPE == "share":
        #
        #     layers_up = self.decoder_layers_layers_up  # PatchExpand, BasicLayerup
        #     concat_back_dim = self.decoder_layers_concat_back_dim
        #     norm_up = self.decoder_layers_norm_up
        #     up = self.decoder_layers_up  # Final layer up
        #
        #     x = self.forward_up_features(x, x_downsample, layers_up, concat_back_dim, norm_up)
        #     outputs = self.up_x4(x, up)  # ([4, 96, 224, 224])
        #
        # elif self.cfg.MODEL.MTLPROMPT.DECODER_TYPE == "sep_last":
        #     outputs = {}
        #     layers_up = self.decoder_layers_layers_up  # PatchExpand, BasicLayerup
        #     concat_back_dim = self.decoder_layers_concat_back_dim
        #     norm_up = self.decoder_layers_norm_up
        #     up = self.decoder_layers_up  # Final layer up
        #
        #     x = self.forward_up_features(x, x_downsample, layers_up, concat_back_dim, norm_up)
        #     for task in self.tasks:
        #         decoder_output = x[task]
        #         outputs[task] = self.up_x4(decoder_output, up)  # ([4, 96, 224, 224])
        #
        # else:
        #     outputs = {}
        #     for task in self.tasks:
        #         layers_up = self.decoder_layers_layers_up[task]
        #         concat_back_dim = self.decoder_layers_concat_back_dim[task]
        #         norm_up = self.decoder_layers_norm_up[task]
        #         up = self.decoder_layers_up[task]
        #         #dec_output = self.decoder_layers_output[task]
        #
        #         x_up = self.forward_up_features(x, x_downsample, layers_up, concat_back_dim, norm_up)
        #
        #         outputs[task] = self.up_x4(x_up, up)

        #return outputs
        return x

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * \
            self.patches_resolution[0] * \
            self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops


    def _create_task_type(self, task_id):
        # Check the task ids and create filter for each image
        task_type = task_id.clone()
        unique_task_ids = torch.unique(task_type)
        unique_task_ids_list = (
            unique_task_ids.cpu().numpy()
            if unique_task_ids.is_cuda
            else unique_task_ids.numpy()
        )
        for unique_task_id in unique_task_ids_list:
            task_type[task_type == unique_task_id] = self.task_id_2_task_idx[
                unique_task_id
            ]
        return task_type, unique_task_ids_list

















