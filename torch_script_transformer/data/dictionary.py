# -*- coding: utf-8 -*-

""" Adapted from
https://github.com/pytorch/fairseq/blob/master/fairseq/data/dictionary.py
"""

from __future__ import unicode_literals

import io
import torch
from six import text_type
from collections import Counter


class Dictionary(object):

    def __init__(
        self,
        bos="<s>",
        pad="<pad>",
        eos="</s>",
        unk="<unk>",
        extra_special_symbols=None
    ):
        self.symbols = []
        self.count = []
        self.indices = {}

        self.bos_word = bos
        self.pad_word = pad
        self.eos_word = eos
        self.unk_word = unk

        self.bos_index = self.add_symbol(bos)
        self.pad_index = self.add_symbol(pad)
        self.eos_index = self.add_symbol(eos)
        self.unk_index = self.add_symbol(unk)

        if extra_special_symbols:
            for s in extra_special_symbols:
                self.add_symbol(s)
        self.nspecial = len(self.symbols)

        self._string_ignores = {self.bos_index, self.eos_index, self.pad_index}

    def __eq__(self, other):
        return self.indices == other.indices

    def __getitem__(self, idx):
        if idx < len(self.symbols):
            return self.symbols[idx]
        return self.unk_word

    def __len__(self):
        """Returns the number of symbols in the dictionary"""
        return len(self.symbols)

    def __contains__(self, sym):
        return sym in self.indices

    def add_symbol(self, word, n=1):
        assert isinstance(word, text_type)
        if word in self.indices:
            idx = self.indices[word]
            self.count[idx] += n
        else:
            idx = len(self.symbols)
            self.indices[word] = idx
            self.symbols.append(word)
            self.count.append(n)
        return idx

    def index(self, sym):
        """ Returns the index of the specified symbol """
        return self.indices.get(sym, self.unk_index)

    def string(self, tensor):
        """ Helper for converting a tensor of token indices to a string. """
        if torch.is_tensor(tensor):
            tensor = tensor.data.cpu()

        if torch.is_tensor(tensor) and tensor.dim() == 2:
            return '\n'.join(self.string(t) for t in tensor)

        return ' '.join(
            self.__getitem__(i) for i in tensor
            if int(i) not in self._string_ignores
        )

    def encode_line(self, line, prepend_bos=False, append_eos=False):
        idxs = [self.index(w) for w in line.split()]
        if prepend_bos:
            idxs = [self.bos_index] + idxs
        if append_eos:
            idxs.append(self.eos_index)
        return torch.LongTensor(idxs)

    def finalize(self, threshold=1, nwords=-1, padding_factor=1):
        """ Sort symbols by frequency in descending order, ignoring special ones.
        Args:
            - threshold defines the minimum word count
            - nwords defines the total number of words in the final dictionary,
                including special symbols
            - padding_factor can be used to pad the dictionary size to be a
                multiple of 8, which is important on some hardware (e.g.,
                Nvidia Tensor Cores).
        """
        if nwords <= 0:
            nwords = len(self)

        new_indices = dict(zip(
            self.symbols[:self.nspecial],
            range(self.nspecial)
        ))
        new_symbols = self.symbols[:self.nspecial]
        new_count = self.count[:self.nspecial]

        c = Counter(dict(sorted(zip(
            self.symbols[self.nspecial:],
            self.count[self.nspecial:]
        ))))
        for symbol, count in c.most_common(nwords - self.nspecial):
            if count >= threshold:
                new_indices[symbol] = len(new_symbols)
                new_symbols.append(symbol)
                new_count.append(count)
            else:
                break

        assert len(new_symbols) == len(new_indices)

        self.count = list(new_count)
        self.symbols = list(new_symbols)
        self.indices = dict(new_indices)

        if padding_factor > 1:
            i = 0
            while len(self) % padding_factor != 0:
                symbol = "madeupword{:04d}".format(i)
                self.add_symbol(symbol, n=0)
                i += 1

    def bos(self):
        """Helper to get index of beginning-of-sentence symbol"""
        return self.bos_index

    def pad(self):
        """Helper to get index of pad symbol"""
        return self.pad_index

    def eos(self):
        """Helper to get index of end-of-sentence symbol"""
        return self.eos_index

    def unk(self):
        """Helper to get index of unk symbol"""
        return self.unk_index

    @classmethod
    def load(cls, filepath, *args, **kwargs):
        """Loads the dictionary from a text file with the format:
        ```
        <symbol0> <count0>
        <symbol1> <count1>
        ...
        ```
        """
        d = cls(*args, **kwargs)
        d.add_from_file(filepath)
        return d

    def add_from_file(self, filepath):
        with io.open(filepath, 'r', encoding='utf-8', newline='\n') as f:
            for line in f:
                word, count = line.split(' ')
                count = int(count)
                self.indices[word] = len(self.symbols)
                self.symbols.append(word)
                self.count.append(count)

    def save(self, filepath):
        """Stores dictionary into a text file"""
        with io.open(filepath, 'w', encoding='utf-8', newline='\n') as f:
            for word, count in zip(
                self.symbols[self.nspecial:],
                self.count[self.nspecial:]
            ):
                f.write('{} {}\n'.format(word, count))
