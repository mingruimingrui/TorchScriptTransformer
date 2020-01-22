from __future__ import unicode_literals

import io


def open_txt_file(filepath, mode='r'):
    return io.open(filepath, mode=mode, encoding='utf-8', newline='\n')


def get_file_size(filepath):
    file_size = 0
    for _ in open_txt_file(filepath, 'r'):
        file_size += 1
    return file_size
