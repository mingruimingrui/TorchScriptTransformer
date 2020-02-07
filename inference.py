#!/usr/bin/env python

""" Script for inference, evaluation and interactive inference """

from __future__ import absolute_import, unicode_literals

import sys
import math
import argparse

from time import time
from tqdm import tqdm

import torch
import numpy as np

from torch_script_transformer.utils import \
    file_utils, generator_utils, batching_utils

ATTY_STOP_WORDS = {'q', 'quit', 'quit()', 'exit', 'exit()'}


def make_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'checkpoint_file', type=str, metavar='FP',
        help='Checkpoint file')

    parser.add_argument(
        '-s', '--src_lang', type=str, required=True,
        help='Source language')
    parser.add_argument(
        '-t', '--tgt_lang', type=str, required=True,
        help='Target language')

    parser.add_argument(
        '--dictpref', type=str, metavar='FP', required=True,
        help='Dictionary prefix')
    parser.add_argument(
        '--bpepref', type=str, metavar='FP', required=True,
        help='BPE model prefix')

    parser.add_argument(
        '-i', '--input', type=argparse.FileType('r', encoding='utf-8'),
        help='Input file stream (default: stdin)')
    parser.add_argument(
        '-o', '--output', type=argparse.FileType('w', encoding='utf-8'),
        help='Output file stream (default: stdout)')
    parser.add_argument(
        '--reference_file', type=argparse.FileType('r', encoding='utf-8'),
        help='If evaluation should be done, this file should contain'
        ' the target translations')
    parser.add_argument(
        '--sacrebleu_tokenizer', type=str, default='13a',
        help='The tokenizer that sacrebleu use for evaluation')

    parser.add_argument(
        '--chunk_size', type=int, default=8192,
        help='The chunk size that sentences should be read and processed'
        ' (default: 8192)')

    parser.add_argument(
        '--show_resp_time', action='store_true',
        help='Record and show response time statistics.')
    parser.add_argument(
        '--show_pbar', action='store_true',
        help='Show process bar')
    parser.add_argument(
        '--input_size', type=int,
        help='The input file size, only used for progress bar')

    parser.add_argument(
        '--cpu', action='store_true')
    parser.add_argument(
        '--fp16', action='store_true')
    parser.add_argument(
        '--jit', action='store_true')

    preproc_configs = parser.add_argument_group('preprocessing_configs')
    preproc_configs.add_argument(
        '--no_aggressive_dash_splits',
        dest='aggressive_dash_splits', action='store_false',
        help='Moses tokenizer config')

    translation_configs = parser.add_argument_group('translation_configs')
    translation_configs.add_argument(
        '--beam_size', type=int, metavar='N', default=1,
        help='(default: 1)')

    translation_configs.add_argument(
        '--max_src_len', type=int, metavar='N', default=256,
        help='Maximum token length of source sentence to translate')
    translation_configs.add_argument(
        '--max_len', type=int, metavar='N', default=256,
        help='Hard max target length (default: 1022)')
    translation_configs.add_argument(
        '--max_len_a', type=float, metavar='F', default=1.4,
        help='Max_tgt_len = min(max_len, src_len * a + b) (default: 1.4)')
    translation_configs.add_argument(
        '--max_len_b', type=float, metavar='F', default=4.0,
        help='Max_tgt_len = min(max_len, src_len * a + b) (default: 4.0)')

    translation_configs.add_argument(
        '--len_penalty', type=float, metavar='F', default=1.0,
        help='<1 to favor shorter sentences,'
        ' >1 to favor longer sentences (default: 1.0)')
    translation_configs.add_argument(
        '--unk_penalty', type=float, metavar='F', default=0.0,
        help='<0 to favor unk generation,'
        ' >0 to penalize unk generation (default: 0.0)')

    translation_configs.add_argument(
        '--no_repeat_ngram_size', type=int, metavar='N', default=0,
        help='Not currently implemented')
    translation_configs.add_argument(
        '--init_out_w_bos', action='store_true',
        help='Target sequence starts with BOS instead of EOS')

    translation_configs.add_argument(
        '--max_batch_tokens', type=int, default=12000,
        help='The maximum number of tokens per batch.')
    translation_configs.add_argument(
        '--max_batch_sents', type=int, default=256,
        help='Maximum number of sentences per batch.')

    return parser


class DummyProgressbar(object):
    @staticmethod
    def update(*args, **kwargs):
        pass

    @staticmethod
    def close():
        pass


def get_input_output_streams(args):
    if args.input is None:
        if sys.stdin.isatty():
            # Interactive mode
            def make_interactive_input_stream():
                while True:
                    text = input('> ')
                    if text.lower() in ATTY_STOP_WORDS:
                        break
                    yield text

            args.chunk_size = 1  # Interactive mode requires chunk size 1
            args.show_pbar = False  # No need for pbar
            assert args.output is None, \
                '-o --output should not be used when in interactive mode.'

            print('Interactive mode. Enter text to translate.')
            input_stream = make_interactive_input_stream()

        else:
            # Stdin
            input_stream = sys.stdin

    else:
        # File input
        input_stream = args.input

    if args.output is None:
        # Stdout
        output_stream = sys.stdout

    else:
        # File output
        output_stream = args.output

    return input_stream, output_stream


def get_device(args):
    if args.cpu:
        return torch.device('cpu')
    else:
        return torch.device('cuda:0')


def load_tokenizer_detokenizer(args):
    from sacremoses import MosesTokenizer, MosesDetokenizer
    tokenizer = MosesTokenizer(lang=args.src_lang)
    detokenizer = MosesDetokenizer(lang=args.tgt_lang)
    return tokenizer, detokenizer


def load_src_bpe_model(args):
    from subword_nmt.apply_bpe import BPE
    src_bpe_path = '{}.{}'.format(args.bpepref, args.src_lang)
    with file_utils.open_txt_file(src_bpe_path) as codes:
        src_bpe_model = BPE(codes)
    return src_bpe_model


def load_dictionary(args):
    from torch_script_transformer.data.dictionary import Dictionary
    src_dict_filepath = '{}.{}'.format(args.dictpref, args.src_lang)
    tgt_dict_filepath = '{}.{}'.format(args.dictpref, args.tgt_lang)
    src_dict = Dictionary.load(src_dict_filepath)
    tgt_dict = Dictionary.load(tgt_dict_filepath)
    return src_dict, tgt_dict


def load_model_and_sequence_generator(args, src_dict, tgt_dict):
    from torch_script_transformer.utils.checkpoint_utils import load_checkpoint
    from torch_script_transformer.sequence_generators.beam_generator3 \
        import BeamGenerator

    model, _ = load_checkpoint(args.checkpoint_file, src_dict, tgt_dict)
    # if args.jit:
    #     model = torch.jit.script(model)
    sequence_generator = BeamGenerator(
        model, tgt_dict,
        beam_size=args.beam_size,
        max_len=args.max_len,
        max_len_a=args.max_len_a,
        max_len_b=args.max_len_b,
        len_penalty=args.len_penalty,
        unk_penalty=args.unk_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        init_out_w_bos=args.init_out_w_bos
    )
    sequence_generator.eval()

    if args.fp16:
        sequence_generator.half()
    sequence_generator.to(get_device(args))
    if args.jit:
        sequence_generator = torch.jit.script(sequence_generator)

    # Model has to be initialized first
    # Especially noticible when using JIT
    dummy_input = torch.zeros(1, args.max_src_len).long()
    dummy_input[0, -1] = src_dict.eos_index
    with torch.no_grad():
        sequence_generator(dummy_input.to(get_device(args)))

    return model, sequence_generator


@torch.no_grad()
def main(args):
    # Log args
    sys.stderr.write('{}\n'.format(args.__repr__()))

    # Load all required processor and model
    device = get_device(args)
    tokenizer, detokenizer = load_tokenizer_detokenizer(args)
    src_bpe_model = load_src_bpe_model(args)
    src_dict, tgt_dict = load_dictionary(args)
    model, sequence_generator = load_model_and_sequence_generator(
        args, src_dict, tgt_dict)

    # Get input and output stream
    input_stream, output_stream = get_input_output_streams(args)

    # Form input into chunks
    chunked_input_stream = generator_utils.chunk_wrapper(
        input_stream, chunk_size=args.chunk_size)

    def preprocess_text(text):
        text = text.strip()
        text = tokenizer.tokenize(
            text=text,
            aggressive_dash_splits=args.aggressive_dash_splits,
            return_str=True
        )
        text = src_bpe_model.process_line(text)
        token_ids = src_dict.encode_line(
            line=text,
            prepend_bos=False,
            append_eos=True
        )
        return token_ids

    def postprocess_text(token_ids):
        text = tgt_dict.string(token_ids)
        text = (text + ' ').replace('@@ ', '').strip()
        return detokenizer.detokenize(text.split())

    def form_model_inputs(sents):
        list_token_ids = []
        for s in sents:
            token_ids = preprocess_text(s)
            list_token_ids.append(token_ids)
        return batching_utils.form_batches(
            list_token_ids=list_token_ids,
            pad_idx=src_dict.pad_index,
            pad_left=True,
            max_len=args.max_src_len,
            max_batch_tokens=args.max_batch_tokens,
            max_batch_sents=args.max_batch_sents,
            do_optimal_batching=True
        )

    if args.show_pbar:
        pbar = tqdm(total=args.input_size)
    else:
        pbar = DummyProgressbar()

    # Placeholder to record translation outputs if needed
    preds = None
    if args.reference_file:
        preds = []

    resp_times = None
    if args.show_resp_time:
        resp_times = []

    model_not_initialized = True
    time_taken = 1e-3
    total_trans_sents = 0
    total_trans_tokens = 0
    total_untrans_sents = 0
    for sents in chunked_input_stream:
        # Form model inputs and also output placeholder
        (model_inputs, ignored_idxs) = form_model_inputs(sents)
        results = [None] * len(sents)

        for idxs, src_tokens, src_lengths in model_inputs:
            pbar.update(len(idxs))
            total_trans_sents += len(src_lengths)
            total_trans_tokens += sum(src_lengths)

            if model_not_initialized:
                all_sent_hypots = sequence_generator(src_tokens.to(device))
                model_not_initialized = False

            t0 = time()
            all_sent_hypots = \
                sequence_generator(src_tokens.to(device))
            t1 = time()
            time_taken += t1 - t0
            if resp_times is not None:
                resp_times.append(t1 - t0)

            # Select top hypothesis
            for i, hypots in zip(idxs, all_sent_hypots):
                hypots.sort(key=lambda x: x[2], reverse=True)
                best_hypot = hypots[0]
                results[i] = (
                    math.exp(best_hypot[2]),
                    postprocess_text(best_hypot[0].cpu())
                )

        # Account for ignored idxs
        pbar.update(len(ignored_idxs))
        total_untrans_sents += len(ignored_idxs)
        for i in ignored_idxs:
            results[i] = (0.0, sents[i])

        # Print to output
        for score, sent in results:
            output_stream.write('{:.3f}\t{}\n'.format(score, sent))
            if preds is not None:
                preds.append(sent)

    # Wrap up
    pbar.close()
    if input_stream != sys.stdin:
        input_stream.close()
    if output_stream != sys.stdout:
        output_stream.close()

    msg = '{} sentences - {} tokens translated\n'.format(
        total_trans_sents, total_trans_tokens)
    sys.stderr.write(msg)
    msg = '{} sentences not translated\n'.format(total_untrans_sents)
    sys.stderr.write(msg)
    msg = 'Translation speed: {:.1f} sents/s - {:.1f} tokens/s\n'.format(
        total_trans_sents / time_taken, total_trans_tokens / time_taken)
    sys.stderr.write(msg)
    if resp_times:
        msg = 'Response time: {:.1f} ms - Median: {:.1f} ms\n'.format(
            1000 * np.mean(resp_times), 1000 * np.median(resp_times))
        sys.stderr.write(msg)

    if not args.reference_file:
        return

    # Perform evaluation
    import sacrebleu

    refs = [l.strip() for l in args.reference_file]
    bleu_score = sacrebleu.corpus_bleu(
        preds, [refs],
        tokenize=args.sacrebleu_tokenizer,
        use_effective_order=True
    )
    sys.stderr.write(bleu_score.format() + '\n')


if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()
    main(args)
