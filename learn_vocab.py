#!/usr/bin/env python

import sys
import argparse
from tqdm import tqdm
from torch_script_transformer.data.dictionary import Dictionary


def make_parser():
    parser = argparse.ArgumentParser('Learn vocab for token embeddings.')

    parser.add_argument(
        '-i', '--input', metavar='PATH',
        type=argparse.FileType('r', encoding='utf-8'), default=sys.stdin,
        help='Input file (defaults to stdin)')
    parser.add_argument(
        '-o', '--output', metavar='PATH',
        type=argparse.FileType('w', encoding='utf-8'), default=sys.stdout,
        help='Output file (defaults to stdout)')

    parser.add_argument(
        '--threshold', type=int, metavar='N', default=1,
        help='Minimum vocab occurance')
    parser.add_argument(
        '--nwords', type=int, metavar='N', default=-1,
        help='Max vocab size')
    parser.add_argument(
        '--padding_factor', type=int, metavar='N', default=1,
        help='Pad dictionary size to be multiple of N')

    parser.add_argument(
        '--show_pbar', action='store_true',
        help='Show progress bar')

    return parser


def main(args):
    dictionary = Dictionary()

    # Parse all lines
    input_stream = args.input
    if args.show_pbar:
        input_stream = tqdm(input_stream)
    for line in input_stream:
        for word in line.split():
            dictionary.add_symbol(word)

    # Finalize dictionary
    dictionary.finalize(
        threshold=args.threshold,
        nwords=args.nwords,
        padding_factor=args.padding_factor
    )

    # Write dictionary
    dictionary.save(args.output)

    if args.input is not sys.stdin:
        args.input.close()
    if args.output is not sys.stdout:
        args.output.close()


if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()
    main(args)
