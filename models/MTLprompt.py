import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Type, Union, Mapping

import torch
import torch.nn as nn
from torch.nn import functional as F
from typing_extensions import Self



def mark_only_mtl_as_trainable(model: nn.Module, freeze_patch_embed: bool = False, freeze_norm: bool = False, free_relative_bias: bool = False, freeze_downsample_reduction=False) -> None:
    """Freeze all modules except MTLprompt
    """
    def prompt_filter(key): return "prompt" in key

    def decoder_filter(key): return "decoders.decoders" in key

    def heads_filter(key): return "heads" in key

    def fusion_filter(key): return "fusion" in key

    def patch_embed_filter(
        key): return not freeze_patch_embed and "patch_embed" in key

    def norm_filter(key): return not freeze_norm and "norm" in key

    def downsample_reduction_filter(
        key): return not freeze_downsample_reduction and "downsample.reduction" in key

    def downsample_filter(key):
        return not freeze_downsample_reduction and "downsample" in key

    def relative_position_bias_filter(
        key): return not free_relative_bias and "relative_position_bias_table" in key

    def all_filters(key):
        return (prompt_filter(key) or decoder_filter(key) or heads_filter(key) or fusion_filter(key) or patch_embed_filter(key) or norm_filter(key) or downsample_reduction_filter(key)
                or downsample_filter(key) or relative_position_bias_filter(key))

    print(f"MTLprompt Freeze patch_embed: {freeze_patch_embed}")
    print(f"MTLprompt Freeze norm: {freeze_norm}")
    print(f"MTLprompt Freeze downsample_reduction: {freeze_downsample_reduction}")
    print(f"MTLprompt Freeze relative_position_bias: {free_relative_bias}")
    # freeze all layers except LoRA's
    for n, p in model.named_parameters():
        if not all_filters(n):
            p.requires_grad = False


def decoder_filter(key: str, value: Any) -> bool:
    return "decoder" in key


def mtlprompt_detail(model : nn.Module, detail=False) -> None:
    """
    detail : model detail info about gradrequire
    """

    trainable_params = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)
    prompt_params = sum(p.numel() for name, p in model.named_parameters()
                        if p.requires_grad and 'prompt' in name)

    fusion_params = sum(p.numel() for name, p in model.named_parameters()
                        if 'fusion' in name)

    heads_params = sum(p.numel() for name, p in model.named_parameters()
                       if 'heads' in name)

    total_model_params = sum(p.numel() for p in model.parameters())
    total_model_params_without_prompt = total_model_params - prompt_params
    decoder_params = sum(p.numel() for name, p in model.named_parameters()
                         if 'decoder' in name)
    backbone_params = sum(p.numel() for name, p in model.named_parameters()
                          if 'backbone' in name)
    decoder_without_fusion_heads = decoder_params - fusion_params - heads_params

    for name, p in model.named_parameters():
        print(f"{name} : {p.numel()}")

    print(f"""
    Number of trainable params: {trainable_params:,}
    Backbone params: {backbone_params:,}
    Decoder params:             {decoder_params:,}
    prompt params:                {prompt_params:,}
    decoder_without_fusion_heads: {decoder_without_fusion_heads:,}
    Extra params:                {(trainable_params - (prompt_params + decoder_params)):,}
    Total params:               {total_model_params:,} (trainable ratio: {trainable_params / total_model_params * 100:2.2f}%)
    Total params without prompt:  {total_model_params_without_prompt:,} (trainable ratio: {trainable_params / total_model_params_without_prompt * 100:2.2f}%)
    """)

    if detail:
        for name, p in model.named_parameters():
            print(f"{name} : {p.requires_grad}")