""" Adapted from
https://github.com/pytorch/fairseq/tree/master/fairseq/modules/positional_embedding.py
https://github.com/pytorch/fairseq/tree/master/fairseq/modules/learned_positional_embedding.py
https://github.com/pytorch/fairseq/tree/master/fairseq/modules/sinusoidal_positional_embedding.py
"""

import math
import torch
from torch import Tensor
from typing import Optional


class LearnedPositionalEmbedding(torch.nn.Embedding):

    def __init__(self, num_embeddings, embedding_dim, padding_idx):
        super().__init__(num_embeddings, embedding_dim)

    def reset_parameters(self):
        torch.nn.init.xavier_normal_(self.weight)
        self.weight[0] = 0

    def forward(self, input_tokens, start):
        # type: (Tensor, Optional[int]) -> Tensor
        if start is None:
            start = 0
        positions = make_positions(input_tokens, self.padding_idx) + start
        return super().forward(positions).detach()


class SinusoidalPositionalEmbedding(torch.nn.Module):

    def __init__(self, num_embeddings, embedding_dim, padding_idx):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.register_buffer('weight', make_sinusoidal_embeddings(
            num_embeddings, embedding_dim, padding_idx))

    def forward(self, input_tokens, start):
        # type: (Tensor, Optional[int]) -> Tensor
        if start is None:
            start = 0
        positions = make_positions(input_tokens, self.padding_idx) + start
        return torch.nn.functional.embedding(positions, self.weight).detach()


def PositionalEmbedding(
    num_embeddings,
    embedding_dim,
    padding_idx,
    learned=False
):
    if learned:
        m = LearnedPositionalEmbedding(
            num_embeddings, embedding_dim, padding_idx)
    else:
        m = SinusoidalPositionalEmbedding(
            num_embeddings, embedding_dim, padding_idx)
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


def make_positions(tensor, padding_idx):
    # type: (Tensor, int) -> Tensor
    """Replace non-padding symbols with their position numbers.

    Position numbers begin at 1. Padding symbols are ignored.
    """
    # The series of casts and type-conversions here are carefully
    # balanced to both work with ONNX export and XLA. In particular XLA
    # prefers ints, cumsum defaults to output longs, and ONNX doesn't know
    # how to handle the dtype kwarg in cumsum.
    mask = tensor.ne(padding_idx).int()
    positions = torch.cumsum(mask, dim=1).type_as(mask) * mask
    return positions.long()
