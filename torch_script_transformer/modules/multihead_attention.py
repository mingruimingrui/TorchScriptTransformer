""" Adapted from
https://github.com/pytorch/fairseq/blob/master/fairseq/modules/multihead_attention.py
"""

import torch


class MultiheadAttention(torch.nn.Module):
    """ Multi-head attention with incremental decoding """

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, \
            "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.in_proj_weight = torch.nn.Parameter(
            torch.Tensor(3 * embed_dim, embed_dim))
        if bias:
            self.in_proj_bias = torch.nn.Parameter(torch.Tensor(3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = torch.nn.Linear(embed_dim, embed_dim, bias=bias)

        self.reset_parameters()

    def register_parameter(self):
        torch.nn.init.xavier_normal_(self.in_proj_weight)
        torch.nn.init.xavier_normal_(self.out_proj.weight)
        if self.in_proj_bias is not None:
            torch.nn.init.constant_(self.in_proj_bias, 0.)
            torch.nn.init.constant_(self.out_proj.bias, 0.)

    def forward(
        self, query, key, value,
        key_padding_mask=None, attn_mask=None,
        need_weights=True, saved_state=None, static_kv=False
    ):
        """ Input shape: Time x Batch x Channel """
        qkv_same = query.data_ptr() == key.data_ptr() == value.data_ptr()
        kv_same = key.data_ptr() == value.data_ptr()

        tgt_len, bsz, embed_dim = query.size()

        if qkv_same:
            q, k, v = self.in_proj_qkv(query)

        elif kv_same:
            q = self.in_proj_q(query)
            if saved_state is None or not static_kv:
                k, v = self.in_proj_kv(key)

        else:
            q = self.in_proj_q(query)
            k = self.in_proj_k(key)
            v = self.in_proj_v(value)

        q *= self.scaling
        q = q.contiguous().view(
            tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        if not (kv_same and saved_state is not None and static_kv):
            k = k.contiguous().view(
                -1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
            v = v.contiguous().view(
                -1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        if saved_state:
            prev_key, prev_value = saved_state
            if static_kv:
                k = prev_key
                v = prev_value
            else:
                k = torch.cat((prev_key, k), dim=1)
                v = torch.cat((prev_value, v), dim=1)

        saved_state = (
            k.view(bsz, self.num_heads, -1, self.head_dim),
            v.view(bsz, self.num_heads, -1, self.head_dim)
        )

        src_len = k.size(1)

        attn_weights = torch.bmm(q, k.transpose(1, 2))

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            attn_weights += attn_mask

        if key_padding_mask is not None:
            attn_weights = attn_weights.view(
                bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.float().masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf')
            ).type_as(attn_weights)
            attn_weights = attn_weights.view(
                bsz * self.num_heads, tgt_len, src_len)

        attn_weights = torch.nn.functional.softmax(
            attn_weights.float(), dim=-1).type_as(attn_weights)
        attn_weights = torch.nn.functional.dropout(
            attn_weights, p=self.dropout, training=self.training)

        attn = torch.bmm(attn_weights, v)
        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)

        if need_weights:
            attn_weights = attn_weights.view(
                bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.sum(dim=1) / self.num_heads
        else:
            attn_weights = None

        return attn, attn_weights, saved_state

    def in_proj_qkv(self, query):
        return self._in_proj(query).chunk(3, dim=-1)

    def in_proj_kv(self, key):
        return self._in_proj(key, start=self.embed_dim).chunk(2, dim=-1)

    def in_proj_q(self, query):
        return self._in_proj(query, end=self.embed_dim)

    def in_proj_k(self, key):
        return self._in_proj(key, start=self.embed_dim, end=2 * self.embed_dim)

    def in_proj_v(self, value):
        return self._in_proj(value, start=2 * self.embed_dim)

    def _in_proj(self, input, start=0, end=None):
        weight = self.in_proj_weight
        bias = self.in_proj_bias
        weight = weight[start:end, :]
        if bias is not None:
            bias = bias[start:end]
        return torch.nn.functional.linear(input, weight, bias)
