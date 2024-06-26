""" Adapted from
https://github.com/pytorch/fairseq/blob/master/fairseq/modules/multihead_attention.py
"""

import torch
from torch import Tensor
from typing import Optional, Tuple


class MultiheadAttention(torch.nn.Module):
    """ Custom made Multi-head Attention for torch script transformer """

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, \
            "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.q_proj = torch.nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = torch.nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = torch.nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = torch.nn.Linear(embed_dim, embed_dim, bias=bias)

        self.reset_parameters()

    @torch.jit.unused
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.q_proj.weight, gain=2 ** -0.5)
        torch.nn.init.xavier_uniform_(self.k_proj.weight, gain=2 ** -0.5)
        torch.nn.init.xavier_uniform_(self.v_proj.weight, gain=2 ** -0.5)

        torch.nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            torch.nn.init.constant_(self.out_proj.bias, 0.)

    def _forward_qkv(
        self, q, k, v,
        bsz, embed_dim, src_len, tgt_len,
        key_padding_mask, attn_mask, need_weights
    ):
        # type: (Tensor, Tensor, Tensor, int, int, int, int, Optional[Tensor], Optional[Tensor], Optional[bool]) -> Tuple[Tensor, Optional[Tensor], Tuple[Tensor, Tensor]]
        saved_state = (
            k.view(bsz, self.num_heads, -1, self.head_dim),
            v.view(bsz, self.num_heads, -1, self.head_dim)
        )

        attn_weights = torch.bmm(q, k.transpose(1, 2))

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            attn_weights += attn_mask

        if key_padding_mask is not None:
            attn_weights = attn_weights.view(
                bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf')
            )
            attn_weights = attn_weights.view(
                bsz * self.num_heads, tgt_len, src_len)

        attn_weights = torch.nn.functional.softmax(
            attn_weights.float(), dim=-1).type_as(attn_weights)
        attn_weights = torch.nn.functional.dropout(
            attn_weights, p=self.dropout, training=self.training)

        attn = torch.bmm(attn_weights, v)
        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)

        if need_weights is None:
            need_weights = False
        if need_weights:
            attn_weights_ = attn_weights.view(
                bsz, self.num_heads, tgt_len, src_len)
            attn_weights_ = attn_weights_.sum(dim=1) / self.num_heads
        else:
            attn_weights_ = None

        return attn, attn_weights_, saved_state

    def forward_encoder_attn(
        self, x, encoder_out,
        key_padding_mask, attn_mask,
        saved_state, need_weights
    ):
        # type: (Tensor, Tensor, Optional[Tensor], Optional[Tensor], Optional[Tuple[Tensor, Tensor]], Optional[bool]) -> Tuple[Tensor, Optional[Tensor], Tuple[Tensor, Tensor]]
        """ Input format: TBC """

        tgt_len, bsz, embed_dim = x.size()

        q = self.q_proj(x)
        q = q.contiguous().view(
            tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if saved_state is None:
            k = self.k_proj(encoder_out)
            v = self.v_proj(encoder_out)
            k = k.contiguous().view(
                -1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
            v = v.contiguous().view(
                -1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        else:
            k, v = saved_state
            k = k.contiguous().view(bsz * self.num_heads, -1, self.head_dim)
            v = v.contiguous().view(bsz * self.num_heads, -1, self.head_dim)

        src_len = k.size(1)

        return self._forward_qkv(
            q=q, k=k, v=v,
            bsz=bsz, embed_dim=embed_dim, src_len=src_len, tgt_len=tgt_len,
            need_weights=need_weights,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask
        )

    def forward_self_attn(
        self, x,
        key_padding_mask, attn_mask,
        saved_state, need_weights
    ):
        # type: (Tensor, Optional[Tensor], Optional[Tensor], Optional[Tuple[Tensor, Tensor]], Optional[bool]) -> Tuple[Tensor, Optional[Tensor], Tuple[Tensor, Tensor]]

        tgt_len, bsz, embed_dim = x.size()
        q = self.q_proj(x) * self.scaling
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.contiguous().view(
            tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.contiguous().view(
            -1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.contiguous().view(
            -1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        if saved_state is not None:
            prev_key, prev_value = saved_state
            prev_key = prev_key.view(bsz * self.num_heads, -1, self.head_dim)
            prev_value = prev_value.view(
                bsz * self.num_heads, -1, self.head_dim)
            k = torch.cat((prev_key, k), dim=1)
            v = torch.cat((prev_value, v), dim=1)

        src_len = k.size(1)

        return self._forward_qkv(
            q=q, k=k, v=v,
            bsz=bsz, embed_dim=embed_dim, src_len=src_len, tgt_len=tgt_len,
            need_weights=need_weights,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask
        )

    forward = forward_self_attn
