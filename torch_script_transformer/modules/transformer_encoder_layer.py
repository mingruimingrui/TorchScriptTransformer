""" Adapted from
https://github.com/pytorch/fairseq/blob/master/fairseq/modules/transformer_layer.py
"""

import torch
from torch import Tensor
from typing import Optional

from torch_script_transformer.modules.multihead_attention \
    import MultiheadAttention


class TransformerEncoderLayer(torch.nn.Module):

    def __init__(
        self, embed_dim, num_heads, ffn_embed_dim,
        dropout=0., activation_dropout=0., attention_dropout=0.,
        normalize_before=False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.self_attn = MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attention_dropout
        )
        self.self_attn_layer_norm = torch.nn.LayerNorm(self.embed_dim)
        self.dropout = dropout
        self.activation_dropout = activation_dropout
        self.normalize_before = normalize_before
        self.fc1 = Linear(self.embed_dim, ffn_embed_dim)
        self.fc2 = Linear(ffn_embed_dim, self.embed_dim)
        self.final_layer_norm = torch.nn.LayerNorm(self.embed_dim)

    def forward(self, x, encoder_padding_mask, attn_mask):
        # type: (Tensor, Tensor, Optional[Tensor]) -> Tensor
        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        x, _, _ = self.self_attn.forward_self_attn(
            x,
            need_weights=False,
            attn_mask=attn_mask,
            key_padding_mask=encoder_padding_mask,
            saved_state=None
        )
        x = torch.nn.functional.dropout(
            x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        residual2 = x
        if self.normalize_before:
            x = self.final_layer_norm(x)
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.dropout(
            x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = torch.nn.dropout(x, p=self.dropout, training=self.training)
        x = residual2 + x
        if not self.normalize_before:
            x = self.final_layer_norm(x)

        return x


def Linear(in_features, out_features, bias=True):
    m = torch.nn.Linear(in_features, out_features, bias)
    torch.nn.init.xavier_uniform_(m.weight)
    if bias:
        torch.nn.init.constant_(m.bias, 0.)
    return m
