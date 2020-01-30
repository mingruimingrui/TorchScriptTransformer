"""
Coordinator for coordinating multiple processes / threads. Taken from
https://github.com/facebookresearch/Detectron/blob/master/detectron/utils/coordinator.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import threading
import traceback
import contextlib
from six.moves import queue

__all__ = [
    'CoordinatorStoppedException',
    'Coordinator',
    'coordinated_get',
    'coordinated_put'
]


DEFAULT_TIMEOUT = 0.1


class CoordinatorStoppedException(Exception):
    pass


class Coordinator(object):

    def __init__(self, event=None):
        if event is None:
            event = threading.Event()
        self._event = event

    def request_stop(self):
        self._event.set()

    def should_stop(self):
        return self._event.is_set()

    def wait_for_stop(self):
        return self._event.wait()

    @contextlib.contextmanager
    def stop_on_exception(self):
        try:
            yield
        except Exception:
            if not self.should_stop():
                traceback.print_exc()
                self.request_stop()


def coordinated_get(coordinator, q):
    while not coordinator.should_stop():
        try:
            return q.get(block=True, timeout=DEFAULT_TIMEOUT)
        except queue.Empty:  # multiprocessing also uses this
            continue
    raise CoordinatorStoppedException('Coordinator stopped during get()')


def coordinated_put(coordinator, q, element):
    while not coordinator.should_stop():
        try:
            q.put(element, block=True, timeout=DEFAULT_TIMEOUT)
            return
        except queue.Full:  # multiprocessing also uses this
            continue
    raise CoordinatorStoppedException('Coordinator stopped during put()')
