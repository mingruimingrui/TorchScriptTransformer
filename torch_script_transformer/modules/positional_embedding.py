""" Adapted from
https://github.com/pytorch/fairseq/tree/master/fairseq/modules/positional_embedding.py
https://github.com/pytorch/fairseq/tree/master/fairseq/modules/learned_positional_embedding.py
https://github.com/pytorch/fairseq/tree/master/fairseq/modules/sinusoidal_positional_embedding.py
"""

import math
import torch
from typing import Optional


class LearnedPositionalEmbedding(torch.nn.Embedding):

    def __init__(self, num_embeddings, embedding_dim, padding_idx=None):
        super().__init__(num_embeddings, embedding_dim)
        if padding_idx:
            self.weight[padding_idx] = 0

    def forward(self, seq_len, start):
        # type: (int, Optional[int]) -> Tensor
        if start is None:
            start = 0
        return self.weight[start:(start + seq_len)].detach()
        # return self.weight.narrow(0, start, seq_len)


class SinusoidalPositionalEmbedding(torch.nn.Module):

    def __init__(self, num_embeddings, embedding_dim, padding_idx=None):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.register_buffer('weight', make_sinusoidal_embeddings(
            num_embeddings, embedding_dim, padding_idx))

    def forward(self, seq_len, start):
        # type: (int, Optional[int]) -> Tensor
        if start is None:
            start = 0
        return self.weight[start:(start + seq_len)].detach()
        # return self.weight.narrow(0, start, seq_len).detach()


def PositionalEmbedding(
    num_embeddings,
    embedding_dim,
    learned=False
):
    if learned:
        m = LearnedPositionalEmbedding(num_embeddings, embedding_dim)
        torch.nn.init.xavier_normal_(m.weight)
    else:
        m = SinusoidalPositionalEmbedding(num_embeddings, embedding_dim)
    return m


def make_sinusoidal_embeddings(
    num_embeddings,
    embedding_dim,
    padding_idx=None
):
    """ Build sinusoidal embeddings.

    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
    emb = torch.arange(
        num_embeddings,
        dtype=torch.float
    ).unsqueeze(1) * emb.unsqueeze(0)
    emb = torch.cat([
        torch.sin(emb),
        torch.cos(emb)
    ], dim=1).view(num_embeddings, -1)
    if embedding_dim % 2 == 1:
        # zero pad
        emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
    if padding_idx is not None:
        emb[padding_idx, :] = 0
    return emb
