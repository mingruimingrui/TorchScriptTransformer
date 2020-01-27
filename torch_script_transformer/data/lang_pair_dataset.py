""" Not a torch.utils.data.Dataset.
A generator is used instead to avoid loading entire corpus
into memory all at once.
"""

from __future__ import unicode_literals

import os
import random
import threading
import multiprocessing

import torch
from subword_nmt.apply_bpe import BPE

from torch_script_transformer.data.dictionary import Dictionary
from torch_script_transformer.utils.coordinator import \
    Coordinator, CoordinatorStoppedException, \
    coordinated_get, coordinated_put
from torch_script_transformer.utils import file_utils, generator_utils


def _pad_token_ids(list_token_ids, pad_idx, pad_left=False):
    batch_size = len(list_token_ids)
    max_len = max(len(t) for t in list_token_ids)
    padded_token_ids = torch.LongTensor(batch_size, max_len).fill_(pad_idx)
    for i, token_ids in enumerate(list_token_ids):
        if pad_left:
            padded_token_ids[i, -len(token_ids):] = token_ids
        else:
            padded_token_ids[i, :len(token_ids)] = token_ids
    return padded_token_ids


def _worker_loop(
    coord, chunk_queue, batch_queue,
    max_batch_tokens, max_batch_sents, bpe_dropout,
    src_bpe_model, tgt_bpe_model,
    src_dict, tgt_dict,
    src_max_pos, tgt_max_pos,
    src_prepend_bos, src_append_eos,
    tgt_prepend_bos, tgt_append_eos,
    tgt_replace_bos_w_eos,
    src_pad_left, tgt_pad_left
):
    def form_batches(list_token_ids):
        all_batches = []
        src_batch = []
        max_src_width = 0
        tgt_batch = []
        max_tgt_width = 0

        for src_token_ids, tgt_token_ids in list_token_ids:
            next_src_width = max(len(src_token_ids), max_src_width)
            next_tgt_width = max(len(tgt_token_ids), max_tgt_width)
            next_size = next_src_width * (len(src_batch) + 1) + \
                next_tgt_width * (len(tgt_batch) + 1)

            # Check if entry can be appended to current batch
            if (
                next_size > max_batch_tokens or
                (max_batch_sents and len(src_batch) >= max_batch_sents)
            ):
                all_batches.append((src_batch, tgt_batch))
                src_batch = []
                max_src_width = 0
                tgt_batch = []
                max_tgt_width = 0

            # Append entry to current batch
            src_batch.append(src_token_ids)
            max_src_width = max(len(src_token_ids), max_src_width)
            tgt_batch.append(tgt_token_ids)
            max_tgt_width = max(len(tgt_token_ids), max_tgt_width)

        if len(src_batch) > 0:
            all_batches.append((src_batch, tgt_batch))

        return all_batches

    while not coord.should_stop():
        try:
            chunk = coordinated_get(coord, chunk_queue)
        except CoordinatorStoppedException:
            break

        # Form chunk into token ids
        chunk_token_ids = []
        for src_line, tgt_line in chunk:
            src_token_ids = src_bpe_model.process_line(src_line, bpe_dropout)
            src_token_ids = src_dict.encode_line(
                src_token_ids, src_prepend_bos, src_append_eos)

            tgt_token_ids = tgt_bpe_model.process_line(tgt_line, bpe_dropout)
            tgt_token_ids = tgt_dict.encode_line(
                tgt_token_ids, tgt_prepend_bos, tgt_append_eos)
            if tgt_prepend_bos and tgt_replace_bos_w_eos:
                tgt_token_ids[0] = tgt_dict.eos_index

            if len(src_token_ids) > src_max_pos:
                continue
            if len(tgt_token_ids) > tgt_max_pos:
                continue
            chunk_token_ids.append((src_token_ids, tgt_token_ids))

        # Sort by src len and tgt len
        chunk_token_ids.sort(key=lambda x: len(x[1]))
        chunk_token_ids.sort(key=lambda x: len(x[0]))

        # Determine how to batch token ids
        all_batches = form_batches(chunk_token_ids)

        # Form batches into tensors
        all_tensor_batches = []
        for src_batch, tgt_batch in all_batches:
            src_lengths = torch.LongTensor([len(t) for t in src_batch])
            tgt_lengths = torch.LongTensor([len(t) for t in tgt_batch])
            src_tokens = _pad_token_ids(
                src_batch, src_dict.pad_index, pad_left=src_pad_left)
            tgt_tokens = _pad_token_ids(
                tgt_batch, tgt_dict.pad_index, pad_left=tgt_pad_left)
            all_tensor_batches.append((
                src_tokens, src_lengths,
                tgt_tokens, tgt_lengths
            ))
        random.shuffle(all_tensor_batches)

        # Place batch into batch queue
        try:
            coordinated_put(coord, batch_queue, all_tensor_batches)
        except CoordinatorStoppedException:
            break


class LangPairDataset(object):
    """ Language pair dataset for training and validation purpose """

    def __init__(
        self, prefixes, src_lang, tgt_lang,
        src_code_path, tgt_code_path,
        src_dict_path, tgt_dict_path,
        src_max_pos=1024, tgt_max_pos=1024,
        src_prepend_bos=False, src_append_eos=True,
        tgt_prepend_bos=True, tgt_append_eos=True,
        tgt_replace_bos_w_eos=False,
        src_pad_left=True, tgt_pad_left=False
    ):
        filepaths = []
        file_sizes = []
        for p in prefixes:
            src_filepath = '{}.{}'.format(p, src_lang)
            tgt_filepath = '{}.{}'.format(p, tgt_lang)
            assert os.path.isfile(src_filepath)
            assert os.path.isfile(tgt_filepath)
            filepaths.append((src_filepath, tgt_filepath))
            file_sizes.append(file_utils.get_file_size(src_filepath))
        self.filepaths = filepaths
        self.file_sizes = file_sizes

        # Load codes and dictionary
        with file_utils.open_txt_file(src_code_path) as codes:
            self.src_bpe_model = BPE(codes)
        with file_utils.open_txt_file(tgt_code_path) as codes:
            self.tgt_bpe_model = BPE(codes)

        self.src_dict = Dictionary.load(src_dict_path)
        self.tgt_dict = Dictionary.load(tgt_dict_path)

        # Save other variables
        self.src_max_pos = src_max_pos
        self.tgt_max_pos = tgt_max_pos

        self.src_prepend_bos = src_prepend_bos
        self.src_append_eos = src_append_eos

        self.tgt_prepend_bos = tgt_prepend_bos
        self.tgt_append_eos = tgt_append_eos

        self.tgt_replace_bos_w_eos = tgt_replace_bos_w_eos

        self.src_pad_left = src_pad_left
        self.tgt_pad_left = tgt_pad_left

    def create_raw_text_iterator(self, generate_infinitely=False):
        while True:
            for src_filepath, tgt_filepath in self.filepaths:
                fsrc = file_utils.open_txt_file(src_filepath, 'r')
                ftgt = file_utils.open_txt_file(tgt_filepath, 'r')

                for src_line, tgt_line in zip(fsrc, ftgt):
                    yield src_line, tgt_line

                fsrc.close()
                ftgt.close()

            if not generate_infinitely:
                break

    def create_batch_iterator(
        self,
        max_batch_tokens=1024,
        max_batch_sents=None,
        bpe_dropout=0.0,
        generate_infinitely=False,
        chunk_size=8192,
        num_workers=1,
        prefetch=1
    ):
        total_num_chunks = sum(self.file_sizes) // chunk_size
        if total_num_chunks * chunk_size < sum(self.file_sizes):
            total_num_chunks += 1

        assert isinstance(num_workers, int)
        assert num_workers >= 1
        assert isinstance(prefetch, int)
        assert prefetch >= 1

        with multiprocessing.Manager() as manager:
            event = manager.Event()
            coord = Coordinator(event)
            chunk_queue = manager.Queue(1)
            batch_queue = manager.Queue(prefetch)

            # Use thread to fill chunk queue
            def fill_chunk_queue():
                coord_stopped = False
                for chunk in generator_utils.chunk_wrapper(
                    self.create_raw_text_iterator(generate_infinitely),
                    chunk_size=chunk_size
                ):
                    while not coord.should_stop():
                        try:
                            coordinated_put(coord, chunk_queue, chunk)
                        except CoordinatorStoppedException:
                            coord_stopped = True
                            break
                    if coord_stopped:
                        break

            chunk_queue_filler = threading.Thread(target=fill_chunk_queue)
            chunk_queue_filler.setDaemon(True)
            chunk_queue_filler.start()

            # Use subprocesses to process chunks
            workers = []
            for _ in range(num_workers):
                w = multiprocessing.Process(
                    target=_worker_loop,
                    args=(
                        coord, chunk_queue, batch_queue,
                        max_batch_tokens, max_batch_sents, bpe_dropout,
                        self.src_bpe_model, self.tgt_bpe_model,
                        self.src_dict, self.tgt_dict,
                        self.src_max_pos, self.tgt_max_pos,
                        self.src_prepend_bos, self.src_append_eos,
                        self.tgt_prepend_bos, self.tgt_append_eos,
                        self.tgt_replace_bos_w_eos,
                        self.src_pad_left, self.tgt_pad_left
                    )
                )
                w.start()
                workers.append(w)

            if generate_infinitely:
                while True:
                    batches = coordinated_get(coord, batch_queue)
                    for batch in batches:
                        yield batch

            else:
                for _ in range(total_num_chunks):
                    batches = coordinated_get(coord, batch_queue)
                    for batch in batches:
                        yield batch

            # Join all coords and subprocesses
            coord.request_stop()
            chunk_queue_filler.join()
            for w in workers:
                w.join()
