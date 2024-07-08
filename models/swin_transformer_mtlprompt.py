
"""
# Code from https://github.com/HuCaoFighting/Swin-Unet
# Code from https://github.com/CAMTL/CA-MTL
# Code from VTAGML
"""
from models.swin_transformer import SwinTransformer
from models.swin_u_transformer import *

import numpy as np


class PromptedWindowAttention(WindowAttention):
    def __init__(self, num_prompts, dim, window_size, num_heads,
        qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super(PromptedWindowAttention, self).__init__(
            dim, window_size, num_heads, qkv_bias, qk_scale,
            attn_drop, proj_drop)
        self.num_prompts = num_prompts

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

        _C, _H, _W = relative_position_bias.shape

        relative_position_bias = torch.cat((
            torch.zeros(_C, self.num_prompts, _W, device=attn.device),
            relative_position_bias
        ), dim=1)
        relative_position_bias = torch.cat((
            torch.zeros(_C, _H + self.num_prompts, self.num_prompts, device=attn.device),
            relative_position_bias
        ), dim=-1)

        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            # incorporate prompt
            # mask: (nW, 49, 49) --> (nW, 49 + n_prompts, 49 + n_prompts)
            nW = mask.shape[0]

            # expand relative_position_bias
            mask = torch.cat((
                torch.zeros(nW, self.num_prompts, _W, device=attn.device),
                mask), dim=1)
            mask = torch.cat((
                torch.zeros(
                    nW, _H + self.num_prompts, self.num_prompts,
                    device=attn.device),
                mask), dim=-1)
            # logger.info("before", attn.shape)
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            # logger.info("after", attn.shape)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class ChannelPromptedSwinTransformerBlock(SwinTransformerBlock):
    def __init__(self, num_prompts, dim, input_resolution,
            num_heads, window_size=7, shift_size=0, mlp_ratio=4., qkv_bias=True,
            qk_scale=None, drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU,
            norm_layer=nn.LayerNorm, num_chan_prompts=None, cfg=None
            ):
        super(ChannelPromptedSwinTransformerBlock, self).__init__(
            dim, input_resolution, num_heads, window_size,
            shift_size, mlp_ratio, qkv_bias, qk_scale, drop,
            attn_drop, drop_path, act_layer, norm_layer
        )

        pixel_no = int(input_resolution[0] * input_resolution[1])

        self.dim = dim

        self.num_chan_prompts = num_prompts

        self.chan_embed_dim = cfg.MODEL.MTLPROMPT.PROMPT.CHANNEL.CHAN_EMBED_DIM

        # channel-wise attention
        self.chan_nheads = cfg.MODEL.MTLPROMPT.PROMPT.CHANNEL.CHAN_N_HEADS
        self.attn_drop = nn.Dropout(0)

        self.token_trans = nn.Linear(pixel_no, self.chan_embed_dim)

        self.token_trans1 = nn.Linear(self.chan_embed_dim, pixel_no)

        self.chan_norm = nn.BatchNorm2d(self.chan_embed_dim)

        self.attn = PromptedWindowAttention(
            num_chan_prompts,
            self.chan_embed_dim, window_size=to_2tuple(self.window_size),
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)

    # TODO Channel prompts 로 수정
    def forward(self, x, chan_prompt):

        H, W = self.input_resolution
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x)

        assert L == H * W, "input feature has wrong size, should be {}, got {}".format(H * W, L)

        x = x.view(B, H, W, C)

        OH, OW = H, W
        if self.pad_size:
            import torch.nn.functional as F
            pad_l, pad_t, pad_b, pad_r = 0, 0, self.pad_size[0], self.pad_size[1]
            x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
            H = OH + pad_b
            W = OW + pad_r

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(
                x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows --> nW*B, window_size, window_size, C
        x_windows = window_partition(shifted_x, self.window_size)
        # nW*B, window_size*window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        # un-pad
        if self.pad_size:
            x = x[:, :OH, :OW, :].contiguous()

        x = x.view(B, OH * OW, C)

        chan_x = x
        chan_x = chan_x.permute(0, 2, 1)  # (B, C, HxW)

        # add back the prompt for attn for parralel-based prompts
        # nW*B, num_prompts + window_size*window_size, C
        num_windows = int(x_windows.shape[0] / B)

        chan_prompt = chan_prompt.unsqueeze(0)
        chan_prompt = chan_prompt.expand(num_windows, -1, -1, -1)
        chan_prompt = chan_prompt.reshape((-1, self.num_chan_prompts, C))



        x_windows = torch.cat((chan_prompt, x_windows), dim=2)

        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        # change input size
        prompt_emb = attn_windows[:, :self.num_prompts, :]
        attn_windows = attn_windows[:, self.num_prompts:, :]
        # change prompt_embs's shape:
        # nW*B, num_prompts, C - B, num_prompts, C
        prompt_emb = prompt_emb.view(-1, B, self.num_prompts, C)
        prompt_emb = prompt_emb.mean(0)

        # merge windows
        attn_windows = attn_windows.view(
            -1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(
            attn_windows, self.window_size, H, W)  # B H W C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(
                shifted_x,
                shifts=(self.shift_size, self.shift_size),
                dims=(1, 2)
            )
        else:
            x = shifted_x
        # un-pad
        if self.pad_size:
            x = x[:, :OH, :OW, :].contiguous()

        # x = x.view(B, H * W, C)
        x = x.view(B, OH * OW, C)

        # add the prompt back:
        x = torch.cat((prompt_emb, x), dim=1)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x[:, self.num_prompts:, :], x[:, :self.num_prompts, :]


class PromptedSwinTransformerBlock(SwinTransformerBlock):
    def __init__(
            self, num_prompts, dim, input_resolution,
            num_heads, window_size=7, shift_size=0, mlp_ratio=4., qkv_bias=True,
            qk_scale=None, drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU,
            norm_layer=nn.LayerNorm, cfg=None
    ):
        super(PromptedSwinTransformerBlock, self).__init__(
            dim, input_resolution, num_heads, window_size,
            shift_size, mlp_ratio, qkv_bias, qk_scale, drop,
            attn_drop, drop_path, act_layer, norm_layer)

        self.num_prompts = num_prompts

        self.attn = PromptedWindowAttention(
                num_prompts,
                dim, window_size=to_2tuple(self.window_size),
                num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                attn_drop=attn_drop, proj_drop=drop)


    def forward(self, x): # Prompt + Patch
        H, W = self.input_resolution
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x)

        prompt_emb = x[:, :self.num_prompts, :]
        x = x[:, self.num_prompts:, :]
        L = L - self.num_prompts

        assert L == H * W, "input feature has wrong size, should be {}, got {}".format(H * W, L)

        x = x.view(B, H, W, C)

        OH, OW = H, W
        if self.pad_size:
            import torch.nn.functional as F
            pad_l, pad_t, pad_b, pad_r = 0, 0, self.pad_size[0], self.pad_size[1]
            x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
            H = OH + pad_b
            W = OW + pad_r

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(
                x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows --> nW*B, window_size, window_size, C
        x_windows = window_partition(shifted_x, self.window_size)
        # nW*B, window_size*window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # W-MSA/SW-MSA
        # nW*B, window_size*window_size, C

        # add back the prompt for attn for parralel-based prompts
        # nW*B, num_prompts + window_size*window_size, C
        num_windows = int(x_windows.shape[0] / B)

        prompt_emb = prompt_emb.unsqueeze(0)
        prompt_emb = prompt_emb.expand(num_windows, -1, -1, -1)
        prompt_emb = prompt_emb.reshape((-1, self.num_prompts, C))
        x_windows = torch.cat((prompt_emb, x_windows), dim=1)

        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        # change input size
        prompt_emb = attn_windows[:, :self.num_prompts, :]
        attn_windows = attn_windows[:, self.num_prompts:, :]
        # change prompt_embs's shape:
        # nW*B, num_prompts, C - B, num_prompts, C
        prompt_emb = prompt_emb.view(-1, B, self.num_prompts, C)
        prompt_emb = prompt_emb.mean(0)

        # merge windows
        attn_windows = attn_windows.view(
            -1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(
            attn_windows, self.window_size, H, W)  # B H W C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(
                shifted_x,
                shifts=(self.shift_size, self.shift_size),
                dims=(1, 2)
            )
        else:
            x = shifted_x
        # un-pad
        if self.pad_size:
            x = x[:, :OH, :OW, :].contiguous()

        # x = x.view(B, H * W, C)
        x = x.view(B, OH * OW, C)

        # add the prompt back:
        x = torch.cat((prompt_emb, x), dim=1)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x[:, self.num_prompts:, :], x[:, :self.num_prompts, :]


class PromptedPatchMerging(PatchMerging):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(
        self, num_prompts, input_resolution,
        dim, norm_layer=nn.LayerNorm
    ):
        super(PromptedPatchMerging, self).__init__(
            input_resolution, dim, norm_layer)
        self.num_prompts = num_prompts

        #self.norm_prompt = norm_layer(dim)
       # self.norm_prompt = norm_layer(2*dim)

    def upsample_prompt(self, prompt_emb):

        prompt_emb = torch.cat(
            (prompt_emb, prompt_emb, prompt_emb, prompt_emb), dim=-1)
        return prompt_emb


    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape

        # change input size
        prompt_emb = x[:, :self.num_prompts, :]
        x = x[:, self.num_prompts:, :]
        L = L - self.num_prompts
        prompt_emb = self.upsample_prompt(prompt_emb)

        assert L == H * W, "input feature has wrong size, should be {}, got {}".format(H*W, L)
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        #print(x.shape)

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = torch.cat((prompt_emb, x), dim=1)

        x = self.norm(x)
        x = self.reduction(x)

        return x[:, self.num_prompts:, :], x[:, :self.num_prompts, :]


class PromptedPatchExpand(PatchExpand):
    def __init__(self, num_prompts, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super(PromptedPatchExpand, self).__init__(input_resolution, dim, dim_scale, norm_layer)

        self.num_prompts = num_prompts

        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)


        self.prompt_downsampling = nn.Linear(dim, dim//2, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm_prompt = norm_layer(dim//2)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape

        #prompt_emb = x[:, :self.num_prompts, :]
        #x = x[:, self.num_prompts:, :]
        #L = L - self.num_prompts

        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C // 4)
        x = x.view(B, -1, C // 4)
        x = self.norm(x)

        return x


class Decoder_PromptedSwinUTransformer(Decoder_MTLdSwinUTransformer):
    def __init__(self, cfg, img_size=224, patch_size=4, in_chans=3,
                 embed_dim=96, depths=[2, 2, 18, 2], depths_decoder=[2, 2, 2, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, final_upsample="expand_first", task_classes = [14, 1], tasks=None,
                 decoder_type = None,
                 device='cuda',**kwargs):

        # super().__init__()

        super(Decoder_PromptedSwinUTransformer, self).__init__(
            cfg, img_size, patch_size, in_chans,
            embed_dim, depths, depths_decoder, num_heads,
            window_size, mlp_ratio, qkv_bias, qk_scale,
            drop_rate, attn_drop_rate, drop_path_rate,
            norm_layer, ape, patch_norm,
            use_checkpoint, final_upsample, task_classes, tasks,
            decoder_type,
            device, **kwargs
        )

        self.cfg = cfg

        self.prompt_dropout = nn.Dropout(cfg.MODEL.MTLPROMPT.PROMPT.PROMPT_DROPOUT)

        dpr_decoder = [x.item() for x in
                       torch.linspace(0, drop_path_rate, sum(depths_decoder))]  # stochastic depth decay rule

        """ Decoder Module """

        # self.layers_up = nn.ModuleList()    # up-sampling Layer 모아두는 모듈
        # self.concat_back_dim = nn.ModuleList()   #

        if cfg.MODEL.MTLPROMPT.DECODER_TYPE == "share":
            self.decoder_layers_layers_up = nn.ModuleList()  # decoder layer 에서 upsample Transformer
            self.decoder_layers_concat_back_dim = nn.ModuleList()  #
            self.decoder_layers_norm_up = nn.ModuleList()  # decoder layer 에서 upsampling 하고 normalize
            self.decoder_layers_up = nn.ModuleList()  # Final up sample 만 포함?
            self.decoder_layers_norm_up.append(norm_layer(self.embed_dim))
            # self.decoder_layers_output = nn.ModuleDict()  # Task specific heads

            # Shared decoder, specific task heads <- VTAGML 이랑 다른 부분
            for i_layer in range(self.num_layers):
                concat_linear = nn.Linear(2 * int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                                          int(embed_dim * 2 ** (
                                                      self.num_layers - 1 - i_layer))) if i_layer > 0 else nn.Identity()

                if i_layer == 0:
                    layer_up = PatchExpand(
                        input_resolution=(self.patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                                          self.patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer))),
                        dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)), dim_scale=2, norm_layer=norm_layer)
                else:
                    layer_up = BasicLayer_up(dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                                             input_resolution=(
                                                 self.patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                                                 self.patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer))),
                                             depth=depths_decoder[(self.num_layers - 1 - i_layer)],
                                             num_heads=num_heads[(self.num_layers - 1 - i_layer)],
                                             window_size=window_size,
                                             mlp_ratio=self.mlp_ratio,
                                             qkv_bias=qkv_bias, qk_scale=qk_scale,
                                             drop=drop_rate, attn_drop=attn_drop_rate,
                                             drop_path=dpr_decoder[sum(depths[:(self.num_layers - 1 - i_layer)]):sum(
                                                 depths[:(self.num_layers - 1 - i_layer) + 1])],
                                             norm_layer=norm_layer,
                                             upsample=PromptedPatchExpand if (i_layer < self.num_layers - 1) else None,
                                             block_module=PromptedSwinTransformerBlock,
                                             use_checkpoint=use_checkpoint,
                                             num_prompts=self.all_prompt_len,
                                             cfg=cfg
                                             )
                # 이 부분 수정
                # self.layers_up.append(layer_up)
                # self.concat_back_dim.append(concat_linear)
                self.decoder_layers_layers_up.append(layer_up)
                self.decoder_layers_concat_back_dim.append(concat_linear)


            if self.final_upsample == "expand_first":
                up = FinalPatchExpand_X4(input_resolution=(
                    img_size // patch_size, img_size // patch_size), dim_scale=4, dim=embed_dim)

                self.decoder_layers_up.append(up)

        elif cfg.MODEL.MTLPROMPT.DECODER_TYPE == "firstblocksep":
            self.decoder_layers_layers_up = nn.ModuleDict()  # decoder layer 에서 upsample Transformer
            #self.decoder_layers_concat_back_dim = nn.ModuleList()  #
            self.decoder_layers_norm_up = nn.ModuleList()  # decoder layer 에서 upsampling 하고 normalize
            self.decoder_layers_up = nn.ModuleDict()  # Final up sample 만 포함?

            for task in tasks:
                up = FinalPatchExpand_X4(input_resolution=(
                    img_size[0] // patch_size, img_size[1] // patch_size), dim_scale=4, dim=embed_dim)

                self.decoder_layers_up[task] = up

            i_layer = 2
            if cfg.MODEL.MTLPROMPT.PROMPT.SPATIAL.ENABLED:
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
                                              upsample=PromptedPatchExpand if (i_layer < self.num_layers - 1) else None,
                                              use_checkpoint=use_checkpoint,
                                              block_module=PromptedSwinTransformerBlock,
                                              num_prompts=self.cfg.MODEL.MTLPROMPT.PROMPT.SPATIAL.LEN,
                                              cfg=cfg)

                    self.decoder_layers_layers_up[task] = layers_up

            else:
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

        elif cfg.MODEL.MTLPROMPT.DECODER_TYPE == "sep_last":
            self.decoder_layers_layers_up = nn.ModuleList()  # decoder layer 에서 upsample Transformer
            self.decoder_layers_concat_back_dim = nn.ModuleList()  #
            self.decoder_layers_norm_up = nn.ModuleList()  # decoder layer 에서 upsampling 하고 normalize
            self.decoder_layers_up = nn.ModuleList()  # Final up sample 만 포함?

            decoder_last_layers_up = nn.ModuleDict()
            # self.decoder_layers_output = nn.ModuleDict()  # Task specific heads

            # Shared decoder, specific task heads <- VTAGML 이랑 다른 부분
            for i_layer in range(self.num_layers - 1):
                concat_linear = nn.Linear(2 * int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                                          int(embed_dim * 2 ** (
                                                  self.num_layers - 1 - i_layer))) if i_layer > 0 else nn.Identity()
                if i_layer == 0:
                    layer_up = PatchExpand(
                        input_resolution=(self.patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                                          self.patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer))),
                        dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)), dim_scale=2, norm_layer=norm_layer)
                else:
                    layer_up = BasicLayer_up(dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
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
                                             use_checkpoint=use_checkpoint,
                                             )
                # 이 부분 수정
                self.decoder_layers_layers_up.append(layer_up)
                self.decoder_layers_concat_back_dim.append(concat_linear)

            self.decoder_layers_norm_up.append(norm_layer(self.embed_dim))

            if self.final_upsample == "expand_first":
                up = FinalPatchExpand_X4(input_resolution=(
                    img_size // patch_size, img_size // patch_size), dim_scale=4, dim=embed_dim)

                self.decoder_layers_up.append(up)

            i_layer = 3
            for task in tasks:
                if cfg.MODEL.MTLPROMPT.PROMPT.SPATIAL.ENABLED:
                    last_layers_up = BasicLayer_up(dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                                                   input_resolution=(
                                                       self.patches_resolution[0] // (
                                                                   2 ** (self.num_layers - 1 - i_layer)),
                                                       self.patches_resolution[1] // (
                                                                   2 ** (self.num_layers - 1 - i_layer))),
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
                                                   upsample=PromptedPatchExpand if (
                                                               i_layer < self.num_layers - 1) else None,
                                                   use_checkpoint=use_checkpoint,
                                                   block_module=PromptedSwinTransformerBlock,
                                                   num_prompts=self.cfg.MODEL.MTLPROMPT.PROMPT.SPATIAL.LEN,
                                                   cfg=cfg)
                else:
                    last_layers_up = BasicLayer_up(dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                                                   input_resolution=(
                                                       self.patches_resolution[0] // (
                                                                   2 ** (self.num_layers - 1 - i_layer)),
                                                       self.patches_resolution[1] // (
                                                                   2 ** (self.num_layers - 1 - i_layer))),
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

                decoder_last_layers_up[task] = last_layers_up
            self.decoder_layers_layers_up.append(decoder_last_layers_up)

            concat_linear = nn.Linear(2 * int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                                      int(embed_dim * 2 ** (
                                              self.num_layers - 1 - i_layer))) if i_layer > 0 else nn.Identity()
            self.decoder_layers_concat_back_dim.append(concat_linear)

            if cfg.MODEL.MTLPROMPT.PROMPT.CHANNEL.ENABLED:
                self.decoder_channel_prompt_attn = nn.ModuleDict()

                for task in tasks:
                    chan_attn_layer = BasicLayer_up(dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                                                    input_resolution=(
                                                        self.patches_resolution[0] // (
                                                                    2 ** (self.num_layers - 1 - i_layer)),
                                                        self.patches_resolution[1] // (
                                                                    2 ** (self.num_layers - 1 - i_layer))),
                                                    depth=depths_decoder[(self.num_layers - 1 - i_layer)],
                                                    num_heads=num_heads[(self.num_layers - 1 - i_layer)],
                                                    window_size=window_size,
                                                    mlp_ratio=self.mlp_ratio,
                                                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                    drop=drop_rate, attn_drop=attn_drop_rate,
                                                    drop_path=dpr_decoder[
                                                              sum(depths_decoder[:(self.num_layers - 1 - i_layer)]):sum(
                                                                  depths_decoder[
                                                                  :(self.num_layers - 1 - i_layer) + 1])],
                                                    norm_layer=norm_layer,
                                                    upsample=PromptedPatchExpand if (
                                                                i_layer < self.num_layers - 1) else None,
                                                    use_checkpoint=use_checkpoint,
                                                    block_module=ChannelPromptedSwinTransformerBlock,
                                                    num_prompts=self.cfg.MODEL.MTLPROMPT.PROMPT.CHANNEL.LEN,
                                                    cfg=cfg)
                    self.decoder_channel_prompt_attn[task] = chan_attn_layer

        else:
            self.decoder_layers_layers_up = nn.ModuleDict()  # decoder layer 에서 upsample Transformer
            self.decoder_layers_concat_back_dim = nn.ModuleDict()  #
            self.decoder_layers_norm_up = nn.ModuleDict()  # decoder layer 에서 upsampling 하고 normalize
            self.decoder_layers_up = nn.ModuleDict()  # Final up sample 만 포함?

            for task in tasks:

                layers_up = nn.ModuleList()
                concat_back_dim = nn.ModuleList()

                for i_layer in range(self.num_layers):
                    concat_linear = nn.Linear(2 * int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                                              int(embed_dim * 2 ** (
                                                      self.num_layers - 1 - i_layer))) if i_layer > 0 else nn.Identity()
                    if i_layer == 0:
                        layer_up = PatchExpand(
                            input_resolution=(self.patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                                              self.patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer))),
                            dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)), dim_scale=2,
                            norm_layer=norm_layer)
                    else:
                        layer_up = BasicLayer_up(dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                                                 input_resolution=(
                                                     self.patches_resolution[0] // (
                                                                 2 ** (self.num_layers - 1 - i_layer)),
                                                     self.patches_resolution[1] // (
                                                                 2 ** (self.num_layers - 1 - i_layer))),
                                                 depth=depths_decoder[(self.num_layers - 1 - i_layer)],
                                                 num_heads=num_heads[(self.num_layers - 1 - i_layer)],
                                                 window_size=window_size,
                                                 mlp_ratio=self.mlp_ratio,
                                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                 drop=drop_rate, attn_drop=attn_drop_rate,
                                                 drop_path=dpr_decoder[
                                                           sum(depths[:(self.num_layers - 1 - i_layer)]):sum(
                                                               depths[:(self.num_layers - 1 - i_layer) + 1])],
                                                 norm_layer=norm_layer,
                                                 upsample=PatchExpand if (i_layer < self.num_layers - 1) else None,
                                                 use_checkpoint=use_checkpoint)

                    layers_up.append(layer_up)
                    concat_back_dim.append(concat_linear)
                # 이 부분 수정
                self.decoder_layers_layers_up[task] = layers_up
                self.decoder_layers_concat_back_dim[task] = concat_back_dim
                self.decoder_layers_norm_up[task] = norm_layer(self.embed_dim)

                if self.final_upsample == "expand_first":
                    up = FinalPatchExpand_X4(input_resolution=(
                        img_size // patch_size, img_size // patch_size), dim_scale=4, dim=embed_dim)

                    self.decoder_layers_up[task] = up

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
        if self.cfg.MODEL.MTLPROMPT.DECODER_TYPE == "firstblocksep" :
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

    def forward(self, x, spa_prompt=None):   # x : B, C, H, W // task_id : B * Task

        if self.cfg.MODEL.MTLPROMPT.DECODER_TYPE == "firstblocksep":
            outputs = {}
            layers_up =self.decoder_layers_layers_up # PatchExpand, BasicLayerup

            concat_back_dim = None
            norm_up = self.decoder_layers_norm_up
            up = self.decoder_layers_up # Final layer up

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



class PromptedSwinUTransformer(MTLSwinUNet):
    """
    (Shared) Swin Encoder <- Freeze
    (Shared | Seperated) Swin Decoder <- Unfreeze
    (Task-specific) Task heads <- Unfreeze
    """
    def __init__(self, cfg, img_size=224, patch_size=4, in_chans=3,
                 embed_dim=96, depths=[2, 2, 18, 2], depths_decoder=[2, 2, 2, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, final_upsample="expand_first", task_classes = [14, 1], tasks=None,
                 decoder_type = None,
                 device='cuda',**kwargs):

        super(PromptedSwinUTransformer, self).__init__(
            img_size, patch_size, in_chans,
            embed_dim, depths, depths_decoder, num_heads,
            window_size, mlp_ratio, qkv_bias, qk_scale,
            drop_rate, attn_drop_rate, drop_path_rate,
            norm_layer, ape, patch_norm,
            use_checkpoint, final_upsample, task_classes, tasks,
            decoder_type,
            device, cfg, **kwargs
        )

        self.cfg = cfg

        self.prompt_dropout = nn.Dropout(cfg.MODEL.MTLPROMPT.PROMPT.PROMPT_DROPOUT)

        # Shared Prompt
        self.shared_prompt_len = cfg.MODEL.MTLPROMPT.PROMPT.SHARED.LEN # Must be changed
        if cfg.MODEL.MTLPROMPT.PROMPT.SHARED.TYPE == 'DEEP':  # Hierarchy -> deep
            # Deep -> layer-wise or block-wise?
            self.deep_prompt_embeddings_0 = nn.Parameter(
                torch.zeros(
                    depths[0] , self.shared_prompt_len, embed_dim
                ))
            trunc_normal_(self.deep_prompt_embeddings_0, mean=1., std=1.)
            self.deep_prompt_embeddings_1 = nn.Parameter(
                torch.zeros(
                    depths[1], self.shared_prompt_len, embed_dim * 2
                ))
            trunc_normal_(self.deep_prompt_embeddings_1, mean=1., std=1.)
            self.deep_prompt_embeddings_2 = nn.Parameter(
                torch.zeros(
                    depths[2], self.shared_prompt_len, embed_dim * 4
                ))
            trunc_normal_(self.deep_prompt_embeddings_2, mean=1., std=1.)
            self.deep_prompt_embeddings_3 = nn.Parameter(
                torch.zeros(
                    depths[3], self.shared_prompt_len, embed_dim * 8
                ))
            trunc_normal_(self.deep_prompt_embeddings_3, mean=1., std=1.)

        elif cfg.MODEL.MTLPROMPT.PROMPT.SHARED.TYPE == 'SHALLOW': # shallow Prompting
            self.prompt_embeddings = nn.Parameter(torch.zeros(
                1, self.shared_prompt_len, embed_dim))
            trunc_normal_(self.prompt_embeddings, mean=1., std=1.)

        # stochastic depth
        dpr_encoder = [x.item() for x in
                       torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule


        """ Encoder Module """

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):  # self.num_layrs = 4, each stage
            layer = BasicLayer(
                                dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(self.patches_resolution[0] // (2 ** i_layer),
                                                 self.patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr_encoder[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PromptedPatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint,
                               block_module=PromptedSwinTransformerBlock,
                               num_prompts=self.cfg.MODEL.MTLPROMPT.PROMPT.SHARED.LEN,
                               cfg = cfg)
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)  # num_features = 32C

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

    def get_patch_embeddings(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        return x


     # Encoder and Bottleneck
    def forward_features(self, x):  # x : B*Task, C=3, H, W

        x = self.get_patch_embeddings(x)
        #print(f"x.shape after embedding : {x.shape}") # B, 3136(56*56) , 128
        x_downsample = []
        mtl_prompt_embd = None

        if self.cfg.MODEL.MTLPROMPT.PROMPT.SHARED.TYPE == 'DEEP':

            for layer, mtl_prompt_embd in zip(self.layers, [self.deep_prompt_embeddings_0, self.deep_prompt_embeddings_1,
                                                            self.deep_prompt_embeddings_2, self.deep_prompt_embeddings_3]):
                x_downsample.append(x)
                mtl_prompt_embd = self.prompt_dropout(mtl_prompt_embd)
                x, _ = layer(x, mtl_prompt_embd)

        elif self.cfg.MODEL.MTLPROMPT.PROMPT.SHARED.TYPE == 'SHALLOW':
            mtl_prompt_embd = self.prompt_embeddings
            for i, layer in enumerate(self.layers):
                mtl_prompt_embd = self.prompt_dropout(mtl_prompt_embd)
                x, mtl_prompt_embd = layer(x, mtl_prompt_embd)
                x_downsample.append(x)

        return x, x_downsample, mtl_prompt_embd    # 이 부분에서 downsamplling 값 가져오기


    def up_x4(self, x, up):
        H, W = self.patches_resolution  # 56, 56
        #print(self.patches_resolution)

        B, L, C = x.shape  # torch.Size([6, 3146, 128])
        assert L == H * W, "input features has wrong size"

        if self.final_upsample == "expand_first":

            for i, layer in enumerate(up):
                #x = up(x)
                x = layer(x)
                x = x.view(B, 4 * H, 4 * W, -1)
                x = x.permute(0, 3, 1, 2)  # B,C,H,W

        return x

#### 여기 up 부분부터!
    def forward(self, x):   # x : B, C, H, W // task_id : B * Task

        # Forward through the encoder layers
        x, x_downsample, shared_prompt = self.forward_features(x)
        return x, x_downsample, shared_prompt


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






