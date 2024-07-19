
"""
# Code from https://github.com/HuCaoFighting/Swin-Unet
# Code from https://github.com/CAMTL/CA-MTL
# Code from VTAGML
"""

from models.swin_u_transformer import *

import numpy as np


class PromptedWindowAttention(WindowAttention):
    def __init__(self, num_prompts, dim, window_size, num_heads,
        qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super(PromptedWindowAttention, self).__init__(
            dim, window_size, num_heads, qkv_bias, qk_scale,
            attn_drop, proj_drop)
        self.num_prompts = num_prompts

    def forward(self, x, mask=None, prompt_location=None):
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

        if prompt_location == "prepend":
            # expand relative_position_bias
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


    def forward(self, x, prompt_location=None): # Prompt + Patch
        H, W = self.input_resolution
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x)

        if prompt_location == "prepend":
            # change input size
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

        if prompt_location == "prepend":
            # expand prompts_embs
            # B, num_prompts, C --> nW*B, num_prompts, C
            prompt_emb = prompt_emb.unsqueeze(0)
            prompt_emb = prompt_emb.expand(num_windows, -1, -1, -1)
            prompt_emb = prompt_emb.reshape((-1, self.num_prompts, C))
            x_windows = torch.cat((prompt_emb, x_windows), dim=1)

        attn_windows = self.attn(x_windows, mask=self.attn_mask, prompt_location = prompt_location)

        # seperate prompt embs --> nW*B, num_prompts, C
        if prompt_location == "prepend":
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
        if prompt_location == "prepend":
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

    def forward(self, x, prompt_location=None):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape

        # change input size
        if prompt_location == "prepend" :
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

        if prompt_location == "prepend" :
            x = torch.cat((prompt_emb, x), dim=1)
            x = self.norm(x)
            x = self.reduction(x)
            return x[:, self.num_prompts:, :], x[:, :self.num_prompts, :]

        else :
            x = self.norm(x)
            x = self.reduction(x)
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

        if cfg.MODEL.MTLPROMPT.PROMPT.SPATIAL.ENABLED:
            self.spa_prompt_encoder = cfg.MODEL.MTLPROMPT.PROMPT.SPATIAL.ENABLED
            self.spa_prompt_len = cfg.MODEL.MTLPROMPT.PROMPT.SPATIAL.LEN

            if cfg.MODEL.MTLPROMPT.PROMPT.SPATIAL.METHOD == "prepend":
                self.spa_prompt = {}
                for task in self.tasks:
                    self.spa_prompt[task] = nn.Parameter(torch.zeros(1, self.spa_prompt_len, embed_dim)).to(device)
                    trunc_normal_(self.spa_prompt[task], mean=1., std=1.)

            elif cfg.MODEL.MTLPROMPT.PROMPT.SPATIAL.METHOD == "low-rank":  # TODO : Make Deep Prompting
                self.spa_prompt_embeddings_0 = {}
                self.spa_prompt_embeddings_1 = {}
                self.spa_prompt_embeddings_2 = {}
                self.spa_prompt_embeddings_3 = {}

                for task in self.tasks:
                    self.spa_prompt_embeddings_0[task] = nn.Parameter(
                        torch.zeros(1, self.patches_resolution[0] * self.patches_resolution[1], self.shared_prompt_len)).to(device)
                    trunc_normal_(self.spa_prompt_embeddings_0[task], mean=1., std=1.)
                    self.spa_prompt_embeddings_1[task] = nn.Parameter(
                        torch.zeros(1, self.patches_resolution[0]//2 * self.patches_resolution[1]//2, self.shared_prompt_len)).to(device)
                    trunc_normal_(self.spa_prompt_embeddings_1[task], mean=1., std=1.)
                    self.spa_prompt_embeddings_2[task] = nn.Parameter(
                        torch.zeros(1, self.patches_resolution[0]//4 * self.patches_resolution[1]//4, self.shared_prompt_len)).to(device)
                    trunc_normal_(self.spa_prompt_embeddings_2[task], mean=1., std=1.)
                    self.spa_prompt_embeddings_3[task] = nn.Parameter(
                        torch.zeros(1, self.patches_resolution[0]//8 * self.patches_resolution[1]//8, self.shared_prompt_len)).to(device)
                    trunc_normal_(self.spa_prompt_embeddings_3[task], mean=1., std=1.)

                self.spa_prompt = [self.spa_prompt_embeddings_0, self.spa_prompt_embeddings_1, self.spa_prompt_embeddings_2, self.spa_prompt_embeddings_3]

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

            if self.cfg.MODEL.MTLPROMPT.PROMPT.SPATIAL.ENABLED:
                spa_prompt = self.spa_prompt
                for i, layer in enumerate(self.layers):
                    mtl_prompt_embd = self.prompt_dropout(mtl_prompt_embd)
                    x, out, mtl_prompt_embd, spa_prompt = layer(x, mtl_prompt_embd, spa_prompt)
                    x_downsample.append(out)

            else:
                for i, layer in enumerate(self.layers):
                    mtl_prompt_embd = self.prompt_dropout(mtl_prompt_embd)
                    x, out, mtl_prompt_embd = layer(x, mtl_prompt_embd)
                    x_downsample.append(out)

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
            task_type[task_type == unique_task_id] = self.task_id_2_task_idx[unique_task_id]
        return task_type, unique_task_ids_list
