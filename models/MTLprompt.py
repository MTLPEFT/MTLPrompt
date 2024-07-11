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
    def prompt_filter(key): return "prompt" in key # learnable한 파라미터 선택  backbone은 이미 false로 되어있고 decoder는 true로 되어있나.... 그래서 여기서는 backbone에 대해서 training할것을 구분하는 것인?? 확인 TODO
    def patch_embed_filter(
        key): return not freeze_patch_embed and "patch_embed" in key

    def norm_filter(key): return not freeze_norm and "norm" in key

    def downsample_reduction_filter(
        key): return not freeze_downsample_reduction and "downsample.reduction" in key

    def relative_position_bias_filter(
        key): return not free_relative_bias and "relative_position_bias_table" in key

    def all_filters(key):
        return prompt_filter(key) or patch_embed_filter(key) or norm_filter(key) or downsample_reduction_filter(key) or relative_position_bias_filter(key)

    print(f" bias mode: {bias}")
    print(f" Freeze patch_embed: {freeze_patch_embed}")
    print(f" Freeze norm: {freeze_norm}")
    print(f" Freeze downsample_reduction: {freeze_downsample_reduction}")
    print(f" Freeze relative_position_bias: {free_relative_bias}")
    # freeze all layers except LoRA's
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