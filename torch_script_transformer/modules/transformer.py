""" Adapted from
https://github.com/pytorch/fairseq/blob/master/fairseq/models/transformer.py
"""

import math
import torch
import argparse
from torch import Tensor
from typing import Tuple, List, Optional

from torch_script_transformer.modules.transformer_layer \
    import TransformerEncoderLayer, TransformerDecoderLayer
from torch_script_transformer.modules.positional_embedding \
    import PositionalEmbedding


class TransformerModel(torch.nn.Module):

    def __init__(self, args, encoder, decoder):
        super().__init__()
        self.args = args
        self.encoder = encoder
        self.decoder = decoder

    def add_args(parser):
        """ Add model-specific arguments to the parser. """
        def add_argument(*args, **kwargs):
            try:
                parser.add_argument(*args, **kwargs)
            except argparse.ArgumentError:
                pass

        add_argument(
            '--encoder_layers', type=int, metavar='N',
            help='num encoder layers')
        add_argument(
            '--decoder_layers', type=int, metavar='N',
            help='num decoder layers')

        add_argument(
            '--embed_dim', type=int, metavar='N',
            help='embedding dimension')
        add_argument(
            '--ffn_embed_dim', type=int, metavar='N',
            help='embedding dimension for FFN')
        add_argument(
            '--num_attention_heads', type=int, metavar='N',
            help='num attention heads')
        add_argument(
            '--normalize_before', action='store_true',
            help='apply layernorm before each attention block')

        add_argument(
            '--learned_pos', action='store_true',
            help='use learned positional embeddings')
        # add_argument(
        #     '--share_decoder_input_output_embed', action='store_true',
        #     help='share decoder input and output embeddings')
        add_argument(
            '--share_all_embeddings', action='store_true',
            help='share encoder, decoder and output embeddings'
            ' (requires shared dictionary and embed dim)')

        add_argument(
            '--max_source_positions', default=1024, type=int, metavar='N',
            help='max number of tokens in the source sequence')
        add_argument(
            '--max_target_positions', default=1024, type=int, metavar='N',
            help='max number of tokens in the target sequence')

        add_argument(
            '--dropout', type=float, metavar='D',
            help='dropout probability')
        add_argument(
            '--attention_dropout', type=float, metavar='D',
            help='dropout probability for attention weights')
        add_argument(
            '--activation_dropout', type=float, metavar='D',
            help='dropout probability after activation in FFN.')

    @torch.jit.unused
    @classmethod
    def build_model(cls, args, src_dict, tgt_dict):
        if args is None:
            import argparse
            args = argparse.Namespace()

        base_architecture(args)

        def build_embedding(dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            emb = Embedding(num_embeddings, embed_dim, padding_idx)
            return emb

        encoder_embed_tokens = build_embedding(src_dict, args.embed_dim)

        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                msg = '--share-all-embeddings requires a joined dictionary'
                raise ValueError(msg)
            decoder_embed_tokens = encoder_embed_tokens
            # args.share_decoder_input_output_embed = True

        else:
            decoder_embed_tokens = build_embedding(tgt_dict, args.embed_dim)

        encoder = TransformerEncoder(args, src_dict, encoder_embed_tokens)
        decoder = TransformerDecoder(args, tgt_dict, decoder_embed_tokens)
        return cls(args, encoder, decoder)

    def forward(self, src_tokens, prev_output_tokens, need_attn):
        # type: (Tensor, Tensor, Optional[bool]) -> Tuple[Tensor, Optional[Tensor]]

        encoder_out, encoder_padding_mask = self.forward_encoder(src_tokens)
        logits, attn, new_incremental_state = self.decoder(
            prev_output_tokens=prev_output_tokens,
            encoder_out=encoder_out,
            encoder_padding_mask=encoder_padding_mask,
            incremental_state=None,
            need_attn=need_attn
        )

        return logits, attn

    @torch.jit.export
    def forward_encoder(self, src_tokens):
        # type: (Tensor) -> Tuple[Tensor, Optional[Tensor]]
        encoder_out, encoder_padding_mask = self.encoder(src_tokens)
        return encoder_out, encoder_padding_mask

    @torch.jit.unused
    def forward_decoder(
        self,
        prev_output_tokens,
        encoder_out,
        encoder_padding_mask,
        incremental_state,
        need_attn
    ):
        # type: (Tensor, Tensor, Optional[Tensor], Optional[List[Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]]]], Optional[bool]) -> Tuple[Tensor, Optional[Tensor], List[Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]]]]
        return self.forward_decoder(
            prev_output_tokens=prev_output_tokens,
            encoder_out=encoder_out,
            encoder_padding_mask=encoder_padding_mask,
            incremental_state=incremental_state,
            need_attn=need_attn
        )


class TransformerEncoder(torch.nn.Module):

    def __init__(self, args, dictionary, embed_tokens):
        super().__init__()

        assert len(dictionary) == embed_tokens.num_embeddings

        embed_dim = embed_tokens.embedding_dim

        self.dropout = args.dropout
        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = args.max_source_positions

        self.embed_scale = math.sqrt(embed_dim)
        self.embed_tokens = embed_tokens
        self.embed_positions = PositionalEmbedding(
            num_embeddings=self.max_source_positions,
            embedding_dim=embed_dim,
            padding_idx=dictionary.pad_index,
            learned=args.learned_pos
        )

        self.layers = torch.nn.ModuleList([
            TransformerEncoderLayer(
                args.embed_dim,
                args.num_attention_heads,
                args.ffn_embed_dim,
                dropout=args.dropout,
                activation_dropout=args.activation_dropout,
                attention_dropout=args.attention_dropout,
                normalize_before=args.normalize_before
            )
            for _ in range(args.encoder_layers)
        ])

        self.normalize_before = args.normalize_before
        self.layer_norm = torch.nn.LayerNorm(args.embed_dim)

    def forward(self, src_tokens):
        # type: (Tensor) -> Tuple[Tensor, Optional[Tensor]]

        # Look up embeddings
        x = self.embed_scale * self.embed_tokens(src_tokens)
        x = x + self.embed_positions(src_tokens, 0)
        x = torch.nn.functional.dropout(
            x, p=self.dropout, training=self.training)

        # BTC -> TBC
        x = x.transpose(0, 1)

        # Compute padding mask
        encoder_padding_mask_ = src_tokens.eq(self.padding_idx)
        if encoder_padding_mask_.any():
            encoder_padding_mask = encoder_padding_mask_
        else:
            encoder_padding_mask = None

        # encoder layers
        for layer in self.layers:
            x = layer(x, encoder_padding_mask)

        if self.normalize_before:
            x = self.layer_norm(x).type_as(x)

        return x, encoder_padding_mask

    def reorder_encoder_out(
        self,
        encoder_out,
        encoder_padding_mask,
        new_order
    ):
        # type: (Tensor, Optional[Tensor], Tensor) -> Tuple[Tensor, Optional[Tensor]]
        encoder_out = encoder_out.index_select(1, new_order)
        if encoder_padding_mask is not None:
            encoder_padding_mask = \
                encoder_padding_mask.index_select(0, new_order)
        return encoder_out, encoder_padding_mask


class TransformerDecoder(torch.nn.Module):

    def __init__(self, args, dictionary, embed_tokens):
        super().__init__()

        assert len(dictionary) == embed_tokens.num_embeddings

        embed_dim = args.embed_dim
        token_embed_dim = embed_tokens.embedding_dim

        self.dropout = args.dropout
        self.padding_idx = embed_tokens.padding_idx
        self.max_target_positions = args.max_target_positions
        # self.share_input_output_embed = args.share_decoder_input_output_embed

        self.embed_scale = math.sqrt(embed_dim)
        self.embed_tokens = embed_tokens
        self.embed_positions = PositionalEmbedding(
            num_embeddings=self.max_target_positions,
            embedding_dim=embed_dim,
            padding_idx=dictionary.pad_index
            learned=args.learned_pos
        )

        # self.proj_in_dim = None
        # self.proj_out_dim = None
        # if embed_dim != token_embed_dim:
        #     self.proj_in_dim = Linear(token_embed_dim, embed_dim, bias=False)
        #     self.proj_out_dim = Linear(embed_dim, token_embed_dim, bias=False)

        self.num_layers = args.decoder_layers
        self.layers = torch.nn.ModuleList([
            TransformerDecoderLayer(
                embed_dim=embed_dim,
                num_heads=args.num_attention_heads,
                ffn_embed_dim=args.ffn_embed_dim,
                dropout=args.dropout,
                activation_dropout=args.activation_dropout,
                attention_dropout=args.attention_dropout,
                normalize_before=args.normalize_before
            )
            for _ in range(args.decoder_layers)
        ])

        # if not self.share_input_output_embed:
        #     self.embed_out = torch.nn.Parameter(
        #         torch.Tensor(len(dictionary), token_embed_dim))
        #     torch.nn.init.xavier_normal_(self.embed_out)

        self.normalize_before = args.normalize_before
        self.layer_norm = torch.nn.LayerNorm(embed_dim)

        self.register_buffer(
            'future_mask',
            torch.triu(fill_with_neg_inf(torch.Tensor(
                self.max_target_positions, self.max_target_positions
            )), 1)
        )

    def forward(
        self,
        prev_output_tokens,
        encoder_out,
        encoder_padding_mask,
        incremental_state,
        need_attn
    ):
        # type: (Tensor, Tensor, Optional[Tensor], Optional[List[Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]]]], Optional[bool]) -> Tuple[Tensor, Optional[Tensor], List[Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]]]]
        start = 0
        if incremental_state is not None:
            start = incremental_state[0][0][0].size(2)

        # Look up embeddings
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)
        # if self.proj_in_dim is not None:
        #     x = self.proj_in_dim(x)
        x = x + self.embed_positions(prev_output_tokens, start)
        x = torch.nn.functional.dropout(
            x, p=self.dropout, training=self.training)

        # BTC -> TBC
        x = x.transpose(0, 1)

        # Compute padding mask
        self_attn_padding_mask_ = prev_output_tokens.eq(self.padding_idx)
        if self_attn_padding_mask_.any():
            self_attn_padding_mask = self_attn_padding_mask_
        else:
            self_attn_padding_mask = None

        # Decoder layers
        new_incremental_state = torch.jit.annotate(
            List[Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]]], [])
        idx = 0  # We can't use enumerate
        attn = torch.jit.annotate(Optional[Tensor], None)
        for layer in self.layers:
            # layer = self.layers[idx]
            # saved_state = None
            # self_attn_mask = None
            if incremental_state is not None:
                saved_state = incremental_state[idx]
                self_attn_mask = None
            else:
                self_attn_mask = self.buffered_future_mask(x)
                saved_state = None

            x, attn, new_saved_state = layer(
                x,
                encoder_out=encoder_out,
                encoder_padding_mask=encoder_padding_mask,
                self_attn_padding_mask=self_attn_padding_mask,
                self_attn_mask=self_attn_mask,
                saved_state=saved_state,
                need_attn=need_attn
            )
            new_incremental_state.append(new_saved_state)
            idx += 1

        if self.normalize_before:
            x = self.layer_norm(x).type_as(x)

        # TBC -> BTC
        x = x.transpose(0, 1)

        # if self.proj_out_dim:
        #     x = self.proj_out_dim(x)

        x = self.output_layer(x)

        return x, attn, new_incremental_state

    def buffered_future_mask(self, x):
        # type: (Tensor) -> Tensor
        dim = x.size(0)
        return self.future_mask[:dim, :dim].detach()

    def output_layer(self, x):
        # type: (Tensor) -> Tensor
        # if self.share_input_output_embed:
        #     return torch.nn.functional.linear(x, self.embed_tokens.weight)
        # else:
        #     return torch.nn.functional.linear(x, self.embed_out)
        return torch.nn.functional.linear(x, self.embed_tokens.weight)


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = torch.nn.Embedding(
        num_embeddings, embedding_dim, padding_idx=padding_idx)
    torch.nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    torch.nn.init.constant_(m.weight[padding_idx], 0)
    return m


def Linear(in_features, out_features, bias=True):
    m = torch.nn.Linear(in_features, out_features, bias)
    torch.nn.init.xavier_uniform_(m.weight)
    if bias:
        torch.nn.init.constant_(m.bias, 0.)
    return m


def fill_with_neg_inf(t):
    """ FP16-compatible function that fills a tensor with -inf. """
    return t.float().fill_(float('-inf')).type_as(t)


def base_architecture(args):
    args.encoder_layers = getattr(args, 'encoder_layers', 6)
    args.decoder_layers = getattr(args, 'decoder_layers', 6)

    args.embed_dim = getattr(args, 'embed_dim', 512)
    args.ffn_embed_dim = getattr(args, 'ffn_embed_dim', 2048)
    args.num_attention_heads = getattr(args, 'num_attention_heads', 8)
    args.normalize_before = getattr(args, 'normalize_before', False)

    args.learned_pos = getattr(args, 'learned_pos', False)
    # args.share_decoder_input_output_embed = \
    #     getattr(args, 'share_decoder_input_output_embed', False)
    args.share_all_embeddings = getattr(args, 'share_all_embeddings', False)

    args.max_source_positions = getattr(args, 'max_source_positions', 1024)
    args.max_target_positions = getattr(args, 'max_target_positions', 1024)

    args.dropout = getattr(args, 'dropout', 0.1)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.)
    args.activation_dropout = getattr(args, 'activation_dropout', 0.)
