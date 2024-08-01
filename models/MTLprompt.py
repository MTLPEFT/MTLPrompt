from typing import Any

import torch.nn as nn
def mark_only_prompt_as_trainable(model: nn.Module, bias: str = "none", freeze_patch_embed: bool = False, freeze_norm: bool = False, free_relative_bias: bool = False, freeze_downsample_reduction=False) -> None:
    """Freeze all modules except LoRA's and depending on 'bias' value unfreezes bias weights.

    Args:
        model: model with prompt
        bias:
            ``"none"``: all bias weights will be frozen,
            ``"all"``: all bias weights will be unfrozen.

    Raises:
        NotImplementedError: if `bias` not in ["none", "all"]
    """

    def prompt_filter(key): return "prompt" in key

    def mope_filter(key):
        return "mope" in key and "routing" not in key

    def patch_embed_filter(
        key): return not freeze_patch_embed and "patch_embed" in key

    def norm_filter(key): return not freeze_norm and "norm" in key

    def downsample_reduction_filter(
        key): return not freeze_downsample_reduction and "downsample.reduction" in key

    def relative_position_bias_filter(
        key): return not free_relative_bias and "relative_position_bias_table" in key

    def all_filters(key):
        return prompt_filter(key) or mope_filter(key) or patch_embed_filter(key) or norm_filter(key) or downsample_reduction_filter(key) or relative_position_bias_filter(key)

    print(f" bias mode: {bias}")
    print(f" Freeze patch_embed: {freeze_patch_embed}")
    print(f" Freeze norm: {freeze_norm}")
    print(f" Freeze downsample_reduction: {freeze_downsample_reduction}")
    print(f" Freeze relative_position_bias: {free_relative_bias}")
    # freeze all layers except prompt
    for n, p in model.named_parameters():
        if not all_filters(n):
            p.requires_grad = False

    # depending on the `bias` value unfreeze bias weights
    if bias == "none":
        return
    if bias == "all":
        for n, p in model.named_parameters():
            if "bias" in n:
                p.requires_grad = True
    else:
       raise NotImplementedError


    #elif transfer_type == "prompt+bias":  https://github.com/KMnP/vpt/blob/main/src/models/vit_models.py#L205
    # for k, p in model.named_parameters():
    #     if "prompt" not in k and 'bias' not in k:
    #         p.requires_grad = False