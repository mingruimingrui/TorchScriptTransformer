#!/usr/bin/env python

import os
import argparse
import itertools
import multiprocessing
from tqdm import tqdm
from collections import defaultdict

from subword_nmt.apply_bpe import BPE

from torch_script_transformer.data.dictionary import Dictionary
from torch_script_transformer.utils import file_utils, generator_utils

_CACHE = {}


def make_parser():
    parser = argparse.ArgumentParser('Learn vocab for token embeddings.')

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
        '--trainprefs', type=str, metavar='FP', nargs='+', required=True,
        help='Train prefixes')

    parser.add_argument(
        '--num_workers', type=int, default=1,
        help='Number of processes to use')

    bpe_group = parser.add_argument_group('bpe_group')

    bpe_group.add_argument(
        '--src_bpe_path', type=str, metavar='FP', required=True,
        help='BPE model path for source language')
    bpe_group.add_argument(
        '--tgt_bpe_path', type=str, metavar='FP', required=True,
        help='BPE mdoel path for target language')

    dict_group = parser.add_argument_group('dict_group')

    dict_group.add_argument(
        '--joined_dictionary', action='store_true',
        help='Should a joined dictionary be learnt?')

    dict_group.add_argument(
        '--thresholdsrc', type=int, metavar='N', default=1,
        help='Minimum vocab occurance')
    dict_group.add_argument(
        '--thresholdtgt', type=int, metavar='N', default=1,
        help='Minimum vocab occurance')

    dict_group.add_argument(
        '--nwordssrc', type=int, metavar='N', default=-1,
        help='Max vocab size')
    dict_group.add_argument(
        '--nwordstgt', type=int, metavar='N', default=-1,
        help='Max vocab size')

    dict_group.add_argument(
        '--padding_factor', type=int, metavar='N', default=1,
        help='Pad dictionary size to be multiple of N')

    return parser


def makedirs(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def load_and_cache_bpe_models(args):
    with file_utils.open_txt_file(args.src_bpe_path, 'r') as codes:
        src_bpe_model = BPE(codes)
        _CACHE['{}_bpe_model'.format(args.src_lang)] = src_bpe_model
    with file_utils.open_txt_file(args.tgt_bpe_path, 'r') as codes:
        tgt_bpe_model = BPE(codes)
        _CACHE['{}_bpe_model'.format(args.tgt_lang)] = tgt_bpe_model
    return src_bpe_model, tgt_bpe_model


def make_corpus_iterator(args, lang):
    for pref in args.trainprefs:
        with file_utils.open_txt_file('{}.{}'.format(pref, lang), 'r') as f:
            for line in f:
                yield line


def worker_fn(params):
    chunk, l = params

    bpe_model = _CACHE['{}_bpe_model'.format(l)]
    word_counts = defaultdict(int)
    for line in chunk:
        for word in bpe_model.process_line(line).split():
            word_counts[word] += 1

    return dict(word_counts), len(chunk)


def main(args):
    makedirs(os.path.dirname(args.dictpref))

    # Initialize dictionaries
    src_dict = Dictionary()
    if args.joined_dictionary:
        tgt_dict = src_dict
    else:
        tgt_dict = Dictionary()

    pool = multiprocessing.Pool(
        args.num_workers,
        initializer=load_and_cache_bpe_models,
        initargs=(args,)
    )
    pbar = tqdm()

    # Extract word count from each training file
    for lang in [args.src_lang, args.tgt_lang]:
        if lang == args.src_lang:
            dictionary = src_dict
        else:
            dictionary = tgt_dict

        def make_corpus_iterator():
            for pref in args.trainprefs:
                filepath = '{}.{}'.format(pref, lang)
                with file_utils.open_txt_file(filepath, 'r') as f:
                    for line in f:
                        yield line.strip()

        corpus_iterator = make_corpus_iterator()
        corpus_iterator = generator_utils.chunk_wrapper(corpus_iterator)

        for word_counts, num_lines in pool.imap(
            worker_fn, zip(corpus_iterator, itertools.repeat(lang))
        ):
            pbar.update(num_lines)
            for word, count in word_counts.items():
                dictionary.add_symbol(word, count)

    pool.close()
    pbar.close()

    # Finalize dictionaries
    src_dict.finalize(
        threshold=args.thresholdsrc,
        nwords=args.nwordssrc,
        padding_factor=args.padding_factor
    )
    if not args.joined_dictionary:
        tgt_dict.finalize(
            threshold=args.thresholdtgt,
            nwords=args.nwordstgt,
            padding_factor=args.padding_factor
        )

    src_dict.save('{}.{}'.format(args.dictpref, args.src_lang))
    tgt_dict.save('{}.{}'.format(args.dictpref, args.tgt_lang))


if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()
    main(args)
