#!/usr/bin/env python

from __future__ import absolute_import, unicode_literals

import os
import warnings
import argparse

import fairseq

from torch_script_transformer.modules.transformer import TransformerModel
from torch_script_transformer.utils.checkpoint_utils import CheckpointManager

if fairseq.__version__ < '0.9.0':
    warnings.warn('convert script has not been tested for fairseq={}'.format(
        fairseq.__version__))

os.environ['CUDA_VISIBLE_DEVICES'] = ''  # No need for GPU


def make_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-i', '--input', type=str, required=True,
        help='Checkpoint file containing a fairseq model')
    parser.add_argument(
        '--src_dict', type=str, required=True)
    parser.add_argument(
        '--tgt_dict', type=str, required=True)

    parser.add_argument(
        '-o', '--output', type=str, required=True,
        help='Directory to store output checkpoint file')

    return parser


class DummyTask:

    def __init__(self, src_dict_path, tgt_dict_path):
        from fairseq.data.dictionary import Dictionary
        self.src_dict = Dictionary.load(src_dict_path)
        self.tgt_dict = Dictionary.load(tgt_dict_path)

    def build_model(self, args):
        from fairseq.models import build_model
        return build_model(args, self)

    @property
    def source_dictionary(self):
        return self.src_dict

    @property
    def target_dictionary(self):
        return self.tgt_dict


def load_fairseq_model(args, task):
    from fairseq.checkpoint_utils import load_model_ensemble
    models, model_args = load_model_ensemble([args.input], task=task)
    return models[0], model_args


def make_tst_model_args(fairseq_model_args):
    tst_model_args = argparse.Namespace()

    S = fairseq_model_args  # Source
    T = tst_model_args      # Target

    assert not S.no_scale_embedding

    assert S.encoder_embed_dim == S.decoder_embed_dim
    assert S.encoder_ffn_embed_dim == S.decoder_ffn_embed_dim
    assert S.encoder_attention_heads == S.decoder_attention_heads
    assert S.encoder_normalize_before == S.decoder_normalize_before

    assert S.encoder_learned_pos == S.decoder_learned_pos
    assert S.share_decoder_input_output_embed

    T.encoder_layers = S.encoder_layers
    T.decoder_layers = S.decoder_layers

    T.embed_dim = S.encoder_embed_dim
    T.ffn_embed_dim = S.encoder_ffn_embed_dim
    T.num_attention_heads = S.encoder_attention_heads
    T.normalize_before = S.encoder_normalize_before

    T.learned_pos = S.encoder_learned_pos
    T.share_all_embeddings = S.share_all_embeddings
    T.max_source_positions = S.max_source_positions
    T.max_target_positions = S.max_target_positions

    T.dropout = S.dropout
    T.attention_dropout = S.attention_dropout
    T.activation_dropout = S.activation_dropout

    tst_model_args = T
    return tst_model_args


def main(args):
    # Load fairseq model
    task = DummyTask(args.src_dict, args.tgt_dict)
    fairseq_model, fairseq_model_args = load_fairseq_model(args, task)

    # Init torchscript transformer model
    tst_model_args = make_tst_model_args(fairseq_model_args)
    tst_model = TransformerModel.build_model(
        tst_model_args, task.src_dict, task.tgt_dict)

    fairseq_state_dict = fairseq_model.state_dict()
    tst_state_dict = tst_model.state_dict()

    for k in fairseq_state_dict:
        if k not in tst_state_dict:
            # Positional embedding and versions are the ones
            # that don't match up
            # Luckily we don't have to transfer them
            assert 'version' in k or 'positions' in k
            continue

        assert tst_state_dict[k].shape == fairseq_state_dict[k].shape
        tst_state_dict[k] = fairseq_state_dict[k].type_as(tst_state_dict[k])

    tst_model.load_state_dict(tst_state_dict)

    cp_manager = CheckpointManager(args.output, tst_model)
    cp_manager.save(0)


if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()
    main(args)
