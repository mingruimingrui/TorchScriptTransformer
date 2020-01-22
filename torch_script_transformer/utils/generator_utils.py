from __future__ import unicode_literals

import threading
from six.moves import queue

from torch_script_transformer.utils.coordinator import Coordinator, \
    CoordinatorStoppedException, coordinated_get, coordinated_put


DEFAULT_CHUNK_SIZE = 8192


def prefetch_wrapper(g, size=1):
    """ Wrapper around a generator object that performs prefetching.
    Note that this function uses threading.
    Also original generator will become dangerous to use. """
    assert isinstance(size, int) and size >= 1
    coord = Coordinator()
    cache = queue.Queue(size)

    def _fill_queue():
        for obj in g:
            if coord.should_stop():
                break
            coordinated_put(coord, cache, obj)
        coord.request_stop()
        return

    worker = threading.Thread(target=_fill_queue)
    worker.setDaemon(True)
    worker.start()

    while not coord.should_stop():
        try:
            obj = coordinated_get(coord, cache)
        except CoordinatorStoppedException:
            pass
        yield obj

    worker.join()


def chunk_wrapper(g, chunk_size=DEFAULT_CHUNK_SIZE):
    chunk = []
    for e in g:
        chunk.append(e)
        if len(chunk) >= chunk_size:
            yield chunk
            chunk = []
    if len(chunk) > 0:
        yield chunk


def flatten_wrapper(g):
    for chunk in g:
        for e in chunk:
            yield e
