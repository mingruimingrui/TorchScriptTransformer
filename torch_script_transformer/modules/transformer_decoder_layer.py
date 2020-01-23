""" Adapted from
https://github.com/pytorch/fairseq/blob/master/fairseq/modules/transformer_layer.py
"""

import torch
from torch import Tensor
from typing import Optional, Tuple

from torch_script_transformer.modules.multihead_attention \
    import MultiheadAttention


class TransformerDecoderLayer(torch.nn.Module):

    def __init__(
        self, embed_dim, num_heads, ffn_embed_dim,
        dropout=0., activation_dropout=0., attention_dropout=0.,
        normalize_before=False
    ):
        self.embed_dim = embed_dim
        self.self_attn = MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=num_heads,
            dropout=attention_dropout
        )
        self.self_attn_layer_norm = torch.nn.LayerNorm(self.embed_dim)

        self.dropout = dropout
        self.activation_dropout = activation_dropout
        self.normalize_before = normalize_before

        self.encoder_attn = MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=num_heads,
            dropout=activation_dropout
        )
        self.encoder_attn_layer_norm = torch.nn.LayerNorm(self.embed_dim)

        self.fc1 = Linear(self.embed_dim, ffn_embed_dim)
        self.fc2 = Linear(ffn_embed_dim, self.embed_dim)
        self.final_layer_norm = torch.nn.LayerNorm(self.embed_dim)

    def forward(
        self, x, encoder_out, saved_state,
        encoder_padding_mask, self_attn_padding_mask, self_attn_mask,
        need_attn
    ):
        # type: (Tensor, Tensor, Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]], Tensor, Optional[Tensor], Optional[Tensor], Optional[bool]) -> Tuple(Tensor, Optional[Tensor], Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]])
        self_attn_saved_state: Optional[Tuple[Tensor, Tensor]]
        encoder_attn_saved_state: Optional[Tuple[Tensor, Tensor]]

        if saved_state is None:
            self_attn_saved_state = None
            encoder_attn_saved_state = None
        else:
            self_attn_saved_state, encoder_attn_saved_state = saved_state

        if need_attn is None:
            need_attn = False

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        x, _, self_attn_saved_state = self.self_attn.forward_self_attn(
            x,
            key_padding_mask=self_attn_padding_mask,
            attn_mask=self_attn_mask,
            saved_state=self_attn_saved_state,
            need_weights=False
        )
        x = torch.nn.functional.dropout(
            x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        residual2 = x
        if self.normalize_before:
            x = self.encoder_attn_layer_norm(x)
        x, attn, encoder_attn_saved_state = \
            self.encoder_attn.forward_encoder_attn(
                x,
                encoder_out=encoder_out,
                key_padding_mask=encoder_padding_mask,
                attn_mask=None,
                saved_state=encoder_attn_saved_state,
                need_weights=need_attn
            )
        x = torch.nn.functional.dropout(
            x, p=self.dropout, training=self.training)
        x = residual2 + x
        if not self.normalize_before:
            x = self.encoder_attn_layer_norm(x)

        residual3 = x
        if self.normalize_before:
            x = self.final_layer_norm(x)
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.dropout(
            x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = torch.nn.functional.dropout(
            x, p=self.dropout, training=self.training)
        x = residual3 + x
        if not self.normalize_before:
            x = self.final_layer_norm(x)

        return x, attn, (self_attn_saved_state, encoder_attn_saved_state)


def Linear(in_features, out_features, bias=True):
    m = torch.nn.Linear(in_features, out_features, bias)
    torch.nn.init.xavier_uniform_(m.weight)
    if bias:
        torch.nn.init.constant_(m.bias, 0.)
    return m
