""" Utility functions to help form batches for translation input
TODO: Generalize form_pair_batches and move it here.
"""

import torch


def pad_token_ids(list_token_ids, pad_idx, pad_left=False):
    batch_size = len(list_token_ids)
    max_len = max(len(t) for t in list_token_ids)
    padded_token_ids = torch.LongTensor(batch_size, max_len).fill_(pad_idx)
    for i, token_ids in enumerate(list_token_ids):
        if pad_left:
            padded_token_ids[i, -len(token_ids):] = token_ids
        else:
            padded_token_ids[i, :len(token_ids)] = token_ids
    return padded_token_ids


def form_batches(
    list_token_ids, pad_idx, pad_left=True, max_len=1024,
    max_batch_tokens=8192, max_batch_sents=None,
    do_optimal_batching=False
):
    all_batches = []
    cur_batch = []
    max_width = 0

    # Make idxs to keep track of original order
    # Optionally also do optimal batching here
    idxs = list(range(len(list_token_ids)))
    if do_optimal_batching:
        idxs.sort(key=lambda i: len(list_token_ids[i]))
    list_token_ids = [list_token_ids[i] for i in idxs]

    # Identify idxs for each batch
    ignored_idxs = []
    for idx in idxs:
        token_ids = list_token_ids[idx]
        if len(token_ids) > max_len:
            ignored_idxs.append(idx)
            continue

        # Check if entry can be appended to current batch
        next_width = max(max_width, len(token_ids))
        next_size = next_width * (len(cur_batch) + 1)
        if (
            next_size > max_batch_tokens or
            (max_batch_sents and len(cur_batch) >= max_batch_sents)
        ):
            all_batches.append(cur_batch)
            cur_batch = []
            max_width = 0

        # Append entry to cur_batch
        cur_batch.append(idx)
        max_width = max(max_width, len(token_ids))

    if len(cur_batch) > 0:
        all_batches.append(cur_batch)

    # cur_batch only contain indexes at the moment
    # Manipulate into batches
    # cur_batch: List[Tuple[batch_idxs, token_tensor]]
    all_batches = [(batch_idxs, pad_token_ids(
        [list_token_ids[i] for i in batch_idxs],
        pad_idx, pad_left=pad_left
    )) for batch_idxs in all_batches]

    return all_batches, ignored_idxs
