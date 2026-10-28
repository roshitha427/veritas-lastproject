"""Short-lived, isolated snapshots for repeated identical analysis requests."""
from collections import OrderedDict
from concurrent.futures import Future
from copy import deepcopy
import threading
import time


class ResultCache:
    def __init__(self, ttl=300, capacity=32, clock=time.monotonic):
        self.ttl, self.capacity, self.clock = ttl, capacity, clock
        self._values = OrderedDict()
        self._pending = {}
        self._lock = threading.Lock()

    def get_or_compute(self, key, compute, reusable=lambda value: True):
        with self._lock:
            cached = self._values.get(key)
            if cached and self.clock() - cached[0] < self.ttl:
                self._values.move_to_end(key)
                return deepcopy(cached[1])
            self._values.pop(key, None)
            future = self._pending.get(key)
            owner = future is None
            if owner:
                future = Future()
                self._pending[key] = future
        if not owner:
            return deepcopy(future.result())
        try:
            value = compute()
            snapshot = deepcopy(value)
            can_reuse = reusable(value)
            with self._lock:
                if can_reuse:
                    self._values[key] = (self.clock(), snapshot)
                    while len(self._values) > self.capacity:
                        self._values.popitem(last=False)
                future.set_result(snapshot)
                self._pending.pop(key, None)
            return deepcopy(snapshot)
        except BaseException as exc:
            with self._lock:
                future.set_exception(exc)
                self._pending.pop(key, None)
            raise
