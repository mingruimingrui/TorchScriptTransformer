""" Script containing functions for incremental decoding.
Functions have to be written in a torchscript compliant manner.

Note this script will not be used as torchscript value resolution
appears to be broken
"""

import torch
from torch import Tensor
from typing import Tuple, Optional


@torch.jit.script
def reorder_encoder_outs(new_order, encoder_out, encoder_padding_mask):
    # type: (Tensor, Tensor, Optional[Tensor]) -> Tuple[Tensor, Optional[Tensor]]
    """ Perform inplace encoder outs reordering """
    torch.index_select(
        encoder_out, 1, new_order,
        out=encoder_out
    )
    if encoder_padding_mask is not None:
        torch.index_select(
            encoder_padding_mask, 0, new_order,
            out=encoder_padding_mask
        )
    return encoder_out, encoder_padding_mask


@torch.jit.script
def reorder_incremental_state(new_order, incremental_state):
    # type: (Tensor, List[Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]]]) -> List[Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]]]
    """ Perform inplace incremental state reordering """
    for saved_state in incremental_state:
        (
            (self_attn_saved_key, self_attn_saved_value),
            (encoder_attn_saved_key, encoder_attn_saved_value)
        ) = saved_state

        torch.index_select(
            self_attn_saved_key, 0, new_order,
            out=self_attn_saved_key
        )
        torch.index_select(
            self_attn_saved_value, 0, new_order,
            out=self_attn_saved_value
        )

        torch.index_select(
            encoder_attn_saved_key, 0, new_order,
            out=encoder_attn_saved_key
        )
        torch.index_select(
            encoder_attn_saved_value, 0, new_order,
            out=encoder_attn_saved_value
        )
    return incremental_state


@torch.jit.script
def reorder_output_buffer(new_order, out_tokens, out_scores):
    # type: (Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor]
    """ Perform inplace output buffer reordering """
    torch.index_select(out_tokens, 0, new_order, out=out_tokens)
    torch.index_select(out_scores, 0, new_order, out=out_scores)
    return out_tokens, out_scores
