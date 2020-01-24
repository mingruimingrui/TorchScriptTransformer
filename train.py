#!/usr/bin/env python

from __future__ import absolute_import, unicode_literals

import re
import sys
import json
import argparse
from collections import Mapping, Iterable

from time import time
from tqdm import tqdm

import torch

from torch_script_transformer.data.lang_pair_dataset \
    import LangPairDataset
from torch_script_transformer.modules.transformer \
    import TransformerModel, base_architecture
from torch_script_transformer.modules.smooth_cross_entropy \
    import SmoothCrossEntropyLoss
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


def parse_args(parser, argv=None):
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
        reduction='sum'
    )
    return loss_fn


def determine_lr(args, update_nb):
    if update_nb <= args.warmup_updates:
        lr = args.lr / args.warmup_updates * (update_nb + 1)
        return max(lr, args.min_lr)

    warmup_elapsed = update_nb - args.warmup_updates

    if args.lr_scheduler == 'inverse_sqrt':
        lr = args.lr * warmup_elapsed ** -0.5

    return max(lr, args.min_lr)


def make_optimizer(args, model):

    if args.optimizer == 'adam':
        # Parse adam betas from string
        adam_beta_pattern = re.compile(r'\(([ 0-9\.]+),([ 0-9\.]+)\)')
        match = adam_beta_pattern.match(args.adam_betas)
        adam_betas = (float(match.group(1)), float(match.group(2)))

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=determine_lr(args, 0),
            betas=adam_betas,
            weight_decay=args.weight_decay
        )

    else:
        raise ValueError(
            '{} is not a valid --optimizer'.format(args.optimizer))

    return optimizer


def compute_loss(model, loss_fn, batch, device, ignore_index):
    batch = to_device(batch, device)
    with torch.no_grad():
        src_tokens, src_lengths, tgt_tokens, _ = batch
        prev_output_tokens = tgt_tokens[:, :-1]
        labels = tgt_tokens[:, 1:]

        keep_pos = labels != ignore_index
        labels = labels[keep_pos]

    logits, _ = model(src_tokens, src_lengths, prev_output_tokens)
    logits = logits[keep_pos]

    loss = loss_fn(logits, labels)

    return loss


class Metric(object):

    def __init__(self, decay_rate=0.99):
        self.meters = {}
        assert 0 <= decay_rate < 1
        self.decay_rate = decay_rate
        self.update_rate = 1 - decay_rate

    def __repr__(self):
        return json.dumps({
            k: ('{:.2f}'.format(v) if isinstance(v, float) else v)
            for k, v in self.meters.items()
        })

    @torch.no_grad()
    def update(self, **kwargs):
        for key, value in kwargs.items():
            self.meters[key] = value

    @torch.no_grad()
    def update_with_decay(self, **kwargs):
        for key, value in kwargs.items():
            cur_value = self.meters.get(key, value)
            new_value = cur_value * self.decay_rate + value * self.update_rate
            self.meters[key] = new_value

    def print_metric(self):
        sys.stderr.write('\r')
        sys.stderr.flush()
        print(self)


@torch.no_grad()
def validate(args, dataset, model, loss_fn, update_nb=None):
    start_time = time()
    model.eval()
    device = get_device(args)
    ignore_index = dataset.tgt_dict.pad_index

    total_loss = 0
    total_num_batchs = 0
    total_num_sents = 0
    total_num_tokens = 0
    for batch in make_batch_iterator(args, dataset, False):
        src_lengths = batch[1]
        total_num_batchs += 1
        total_num_sents += len(src_lengths)
        total_num_tokens += int(src_lengths.sum())

        loss = compute_loss(model, loss_fn, batch, device, ignore_index)
        total_loss += float(loss.item())

    time_taken = time() - start_time
    metrics = Metric()
    if update_nb is not None:
        metrics.update(update_nb=update_nb)
    metrics.update(
        valid_loss=total_loss / total_num_tokens,
        time_taken=time_taken,
        bps=total_num_batchs / time_taken,
        sps=total_num_sents / time_taken,
        wps=total_num_tokens / time_taken
    )

    metrics.print_metric()

    return metrics


def train_one_update(
    args, batch_iterator, model,
    loss_fn, optimizer,
    update_nb, ignore_index, metrics
):
    model.train()
    device = get_device(args)
    optimizer.lr = determine_lr(args, update_nb)
    optimizer.zero_grad()

    update_start_time = time()
    total_loss = 0
    total_num_batchs = 0
    total_num_sents = 0
    total_num_tokens = 0

    for _ in range(args.update_freq):
        batch = next(batch_iterator)

        src_lengths = batch[1]
        total_num_batchs += 1
        total_num_sents += len(src_lengths)
        total_num_tokens += int(src_lengths.sum())

        loss = compute_loss(model, loss_fn, batch, device, ignore_index)
        loss.backward()

        total_loss += float(loss.item())

    optimizer.step()

    update_time_taken = time() - update_start_time
    metrics.update_with_decay(
        loss=total_loss / total_num_tokens,
        bps=total_num_batchs / update_time_taken,
        sps=total_num_sents / update_time_taken,
        wps=total_num_tokens / update_time_taken
    )


def train(args, dataset, valid_dataset, model, loss_fn, optimizer):
    start_time = time()
    model_ = torch.jit.script(model)
    model.train()
    ignore_index = dataset.tgt_dict.pad_index

    batch_iterator = make_batch_iterator(args, dataset, True)
    pbar = tqdm(total=args.max_update, ncols=80)
    metrics = Metric()
    metrics.update(update_nb=0)

    for update_nb in range(args.max_update):
        pbar.update(1)
        train_one_update(
            args, batch_iterator,
            model, loss_fn, optimizer,
            update_nb, ignore_index, metrics
        )

        if update_nb % args.log_interval == 0:
            metrics.update(
                update_nb=update_nb,
                time_elapsed=time() - start_time,
                lr=float(optimizer.lr)
            )
            metrics.print_metric()

        if update_nb % args.valid_interval == 0:
            validate(args, valid_dataset, model_, loss_fn, update_nb)

        if update_nb % args.save_interval == 0:
            checkpoint_utils.save_model(model, args.checkpoint_dir, update_nb)

    time_taken = time() - start_time
    metrics.update(
        update_nb=args.max_update,
        time_elapsed=time_taken
    )
    sys.stdout.write('\r')
    metrics.print_metric()
    validate(args, valid_dataset, model_, loss_fn, args.max_update)
    checkpoint_utils.save_model(model, args.checkpoint_dir, update_nb)
    print('Training done in {:.1f}s'.format(time_taken))


def main(args):
    print('Loading dataset')
    train_dataset, valid_dataset = load_datasets(args)

    print('Loading model')
    model = load_model(args, train_dataset.src_dict, train_dataset.tgt_dict)

    print('Making other training misc')
    loss_fn = make_loss_fn(args, train_dataset)
    optimizer = make_optimizer(args, model)

    train(args, train_dataset, valid_dataset, model, loss_fn, optimizer)


if __name__ == "__main__":
    parser = make_parser()
    args = parse_args(parser)
    main(args)
