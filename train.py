#!/usr/bin/env python

from __future__ import absolute_import, unicode_literals

import os
import re
import sys
import json
import argparse
from collections import Mapping, Iterable

from tensorboardX import SummaryWriter

from time import time
from tqdm import tqdm

import torch

from torch_script_transformer.data.lang_pair_dataset \
    import LangPairDataset
from torch_script_transformer.modules.transformer \
    import TransformerModel, base_architecture
from torch_script_transformer.modules.smooth_cross_entropy \
    import SmoothCrossEntropyLoss
from torch_script_transformer.optim.adam import Adam
from torch_script_transformer.utils \
    import checkpoint_utils


def make_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'checkpoint_dir', type=str, metavar='FP',
        help='Checkpoint directory')

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
        '--validprefs', type=str, metavar='FP', nargs='+',
        help='Validation prefixes')

    parser.add_argument(
        '--num_workers', type=int, default=1,
        help='Number of processes load data with')

    parser.add_argument(
        '--src_bpe_path', type=str, metavar='FP', required=True,
        help='BPE model path for source language')
    parser.add_argument(
        '--tgt_bpe_path', type=str, metavar='FP', required=True,
        help='BPE mdoel path for target language')

    train_group = parser.add_argument_group('train_group')

    train_group.add_argument(
        '--cpu', action='store_true',
        help='Do training on CPU')
    train_group.add_argument(
        '--fp16', action='store_true',
        help='Should fp16 training be done?')

    train_group.add_argument(
        '--optimizer', type=str, default='adam',
        choices=['adam'],
        help='Optimizer type')
    train_group.add_argument(
        '--adam_betas', type=str, metavar='O', default='(0.9, 0.98)',
        help='Adam beta 1 and beta 2 in the "(b1, b2)" format')
    train_group.add_argument(
        '--clip_norm', type=float, metavar='F', default=0.0,
        help='Clip norm')
    train_group.add_argument(
        '--weight_decay', type=float, default=0.0,
        help='Weight decay')

    train_group.add_argument(
        '--warmup_updates', type=int, metavar='N', default=4000,
        help='Linear warmup updates')
    train_group.add_argument(
        '--lr', type=float, metavar='F', default=0.001,
        help='Learning rate')
    train_group.add_argument(
        '--min_lr', type=float, metavar='F', default=1e-9,
        help='Min learning rate')
    train_group.add_argument(
        '--lr_scheduler', type=str, default='inverse_sqrt',
        choices=['inverse_sqrt'],
        help='Learning rate scheduler')

    # train_group.add_argument(
    #     '--loss', type=str, default='smooth_cross_entropy,
    #     choices=['smooth_cross_entropy', 'cross_entropy'])
    train_group.add_argument(
        '--label_smoothing', type=float, metavar='N', default=0.1,
        help='Label smoothing')

    train_group.add_argument(
        '--max_batch_tokens', type=int, metavar='N', default=2048,
        help='The maximum number of tokens per batch')
    train_group.add_argument(
        '--max_batch_sents', type=int, metavar='N',
        help='The maximum number of sentence per batch')

    train_group.add_argument(
        '--max_update', type=int, metavar='N', default=200000,
        help='Number of updates to do')
    train_group.add_argument(
        '--update_freq', type=int, metavar='N', default=1,
        help='Number of steps per update')

    train_group.add_argument(
        '--bpe_dropout', type=float, metavar='F', default=0.0,
        help='The BPE dropout rate')

    train_group.add_argument(
        '--save_interval', type=int, metavar='N', default=10000,
        help='Save every N updates')
    train_group.add_argument(
        '--valid_interval', type=int, metavar='N', default=10000,
        help='Validate every N updates')
    train_group.add_argument(
        '--log_interval', type=int, metavar='N', default=100,
        help='Log every N updates')

    model_group = parser.add_argument_group('model_group')
    TransformerModel.add_args(model_group)

    return parser


def add_arch_and_parse_args(parser, argv=None):
    default_args = argparse.Namespace()
    base_architecture(default_args)

    default_kwargs = {}
    for k in dir(default_args):
        if k.startswith('_'):
            continue
        default_kwargs[k] = getattr(default_args, k)

    parser.set_defaults(**default_kwargs)
    args = parser.parse_args(argv)

    return args


def get_device(args):
    if args.cpu:
        return torch.device('cpu')
    else:
        return torch.device('cuda:0')


def to_device(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, Mapping):
        return {k: to_device(v, device) for k, v in obj.items()}
    elif isinstance(obj, Iterable):
        return [to_device(e, device) for e in obj]
    else:
        return obj


def load_datasets(args):
    dataset_kwargs = {
        'src_lang': args.src_lang,
        'tgt_lang': args.tgt_lang,
        'src_code_path': args.src_bpe_path,
        'tgt_code_path': args.tgt_bpe_path,
        'src_dict_path': '{}.{}'.format(args.dictpref, args.src_lang),
        'tgt_dict_path': '{}.{}'.format(args.dictpref, args.tgt_lang),
        'src_max_pos': args.max_source_positions,
        'tgt_max_pos': args.max_target_positions,
        'src_prepend_bos': False, 'src_append_eos': True,
        'tgt_prepend_bos': True, 'tgt_append_eos': True,
        'src_pad_left': True, 'tgt_pad_left': False
    }

    train_dataset = LangPairDataset(prefixes=args.trainprefs, **dataset_kwargs)
    valid_dataset = None
    if args.validprefs:
        valid_dataset = LangPairDataset(
            prefixes=args.validprefs, **dataset_kwargs)
    return train_dataset, valid_dataset


def make_batch_iterator(args, dataset, generate_infinitely=False):
    return dataset.create_batch_iterator(
        max_batch_tokens=args.max_batch_tokens,
        max_batch_sents=args.max_batch_sents,
        bpe_dropout=args.bpe_dropout,
        generate_infinitely=generate_infinitely,
        num_workers=args.num_workers,
        prefetch=1
    )


def load_model(args, src_dict, tgt_dict):
    model = TransformerModel.build_model(args, src_dict, tgt_dict)
    model = model.to(get_device(args))
    if args.fp16:
        model = model.half()
    return model


def make_loss_fn(args, dataset):
    loss_fn = SmoothCrossEntropyLoss(
        len(dataset.tgt_dict),
        smoothing=args.label_smoothing,
        reduction='mean'
    )
    return loss_fn


def determine_lr(args, update_nb):
    if update_nb <= args.warmup_updates:
        lr = args.lr / args.warmup_updates * (update_nb)
        return max(lr, args.min_lr)

    if args.lr_scheduler == 'inverse_sqrt':
        lr = args.lr * (args.warmup_updates / update_nb) ** 0.5

    return max(lr, args.min_lr)


def make_optimizer(args, model):

    if args.optimizer == 'adam':
        # Parse adam betas from string
        adam_beta_pattern = re.compile(r'\(([ 0-9\.]+),([ 0-9\.]+)\)')
        match = adam_beta_pattern.match(args.adam_betas)
        adam_betas = (float(match.group(1)), float(match.group(2)))

        optimizer = Adam(
            model.parameters(),
            lr=determine_lr(args, 1),
            betas=adam_betas,
            weight_decay=args.weight_decay,
        )

        # optimizer = torch.optim.Adam(
        #     model.parameters(),
        #     lr=determine_lr(args, 1),
        #     betas=adam_betas,
        #     weight_decay=args.weight_decay
        # )

    else:
        raise ValueError(
            '{} is not a valid --optimizer'.format(args.optimizer))

    return optimizer


def forward_loss(model, loss_fn, batch, device, ignore_idx):
    with torch.no_grad():
        src_tokens = batch[0]
        tgt_tokens = batch[2]
        src_tokens = to_device(src_tokens, device)
        tgt_tokens = to_device(tgt_tokens, device)
        prev_output_tokens = tgt_tokens[:, :-1]
        labels = tgt_tokens[:, 1:]

        keep_pos = labels != ignore_idx
        labels = labels[keep_pos]

    logits, _ = model(src_tokens, prev_output_tokens, False)
    logits = logits[keep_pos]

    loss = loss_fn(logits, labels)
    return loss


def ppdict(metrics):
    print(json.dumps({
        k: ('{:.3g}'.format(v) if isinstance(v, float) else v)
        for k, v in metrics.items()
    }))


@torch.no_grad()
def validate(args, dataset, model, loss_fn, update_nb=None, writer=None):
    start_time = time()
    model.eval()
    device = get_device(args)
    ignore_idx = dataset.tgt_dict.pad_index

    total_loss = 0
    total_num_batches = 0
    total_num_sents = 0
    total_num_tokens = 0
    for batch in make_batch_iterator(args, dataset, False):
        src_lengths = batch[1]
        total_num_batches += 1
        total_num_sents += len(src_lengths)
        total_num_tokens += int(src_lengths.sum())

        loss = forward_loss(model, loss_fn, batch, device, ignore_idx)
        total_loss += float(loss.item())

    time_taken = time() - start_time

    metrics = {
        'update_nb': update_nb,
        'valid_loss': total_loss / total_num_batches,
        'time_taken': time_taken,
        'bps': total_num_batches / time_taken,
        'sps': total_num_sents / time_taken,
        'wps': total_num_tokens / time_taken
    }
    sys.stderr.write('\r')
    sys.stderr.flush()
    ppdict(metrics)
    if writer:
        writer.add_scalar('valid_loss', metrics['valid_loss'], update_nb)

    return metrics


def train_one_update(
    args, batch_iterator, ignore_idx, model,
    loss_fn, optimizer, update_nb,
    writer=None, cur_num_sents=None, cur_num_tokens=None,
    verbose=False
):
    start_time = time()
    model.train()
    device = get_device(args)
    optimizer.lr = determine_lr(args, update_nb)

    total_loss = 0
    total_num_batches = 0
    total_num_sents = 0
    total_num_tokens = 0

    optimizer.zero_grad()
    for _ in range(args.update_freq):
        batch = next(batch_iterator)

        src_lengths = batch[1]
        total_num_batches += 1
        total_num_sents += len(src_lengths)
        total_num_tokens += int(src_lengths.sum())

        loss = forward_loss(model, loss_fn, batch, device, ignore_idx)
        loss.backward()
        total_loss += float(loss.item())
    optimizer.step()

    time_taken = time() - start_time

    metrics = {
        'update_nb': update_nb,
        'loss': total_loss / total_num_batches,
        'time_taken': time_taken,
        'bps': total_num_batches / time_taken,
        'sps': total_num_sents / time_taken,
        'wps': total_num_tokens / time_taken,
        'lr': float(optimizer.lr)
    }
    if cur_num_sents is not None:
        metrics['num_sents_elapsed'] = cur_num_sents + total_num_sents
    if cur_num_tokens is not None:
        metrics['num_tokens_elapsed'] = cur_num_tokens + total_num_tokens
    if verbose:
        sys.stderr.write('\r')
        sys.stderr.flush()
        ppdict(metrics)

    if writer:
        writer.add_scalar('loss', metrics['loss'], update_nb)
        writer.add_scalar('bps', metrics['bps'], update_nb)
        writer.add_scalar('sps', metrics['sps'], update_nb)
        writer.add_scalar('wps', metrics['wps'], update_nb)
        writer.add_scalar('lr', metrics['lr'], update_nb)
        if cur_num_sents:
            writer.add_scalar(
                'num_sents_elapsed', metrics['num_sents_elapsed'], update_nb)
        if cur_num_sents:
            writer.add_scalar(
                'num_tokens_elapsed', metrics['num_tokens_elapsed'], update_nb)

    return metrics


def main(args):
    start_time = time()
    print(args)

    print('Loading dataset')
    dataset, valid_dataset = load_datasets(args)

    print('Loading model')
    model = load_model(args, dataset.src_dict, dataset.tgt_dict)

    print('Making other training misc')
    loss_fn = make_loss_fn(args, dataset)
    optimizer = make_optimizer(args, model)

    batch_iterator = make_batch_iterator(args, dataset, True)
    ignore_idx = dataset.tgt_dict.pad_index
    num_sents_elapsed = 0
    num_tokens_elapsed = 0

    writer = SummaryWriter(os.path.join(args.checkpoint_dir, 'train'))
    valid_writer = SummaryWriter(os.path.join(args.checkpoint_dir, 'valid'))
    pbar = tqdm(total=args.max_update, ncols=80)

    for update_nb in range(1, args.max_update + 1):
        pbar.update(1)
        metrics = train_one_update(
            args, batch_iterator, ignore_idx, model,
            loss_fn, optimizer, update_nb, writer=writer,
            cur_num_sents=num_sents_elapsed,
            cur_num_tokens=num_tokens_elapsed,
            verbose=update_nb % args.log_interval == 0
        )
        num_sents_elapsed = metrics['num_sents_elapsed']
        num_tokens_elapsed = metrics['num_tokens_elapsed']

        if update_nb % args.valid_interval == 0:
            validate(args, dataset, model, loss_fn, update_nb, valid_writer)

        if update_nb % args.save_interval == 0:
            checkpoint_utils.save_model(model, args.checkpoint_dir, update_nb)

    validate(args, dataset, model, loss_fn, update_nb, valid_writer)
    checkpoint_utils.save_model(model, args.checkpoint_dir, update_nb)

    time_taken = time() - start_time()
    print('Training done in {:.1f}s'.format(time_taken))


if __name__ == "__main__":
    parser = make_parser()
    args = add_arch_and_parse_args(parser)
    main(args)
