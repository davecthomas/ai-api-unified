# cache_store.py
"""
The store behind the library's process-shared caches, and the one way to
replace it.

The library keeps two kinds of state a deployment may want to share across
processes: cached responses (middleware/response_cache.py) and fallback
cooldowns (completions/ai_fallback_cooldown.py). Both read and write through
this module's `CacheStore`: three methods over bytes, with a TTL on every
entry, so a store needs no knowledge of what it holds.

The default store is in memory, per process: an ordered dict with a lock,
least-recently-used eviction at `max_entries`, and an expiry per entry. It
ships with the library and needs nothing installed. An application that
runs several processes and wants them to share one cache hands the library
its own store, backed by whatever it already runs:

    class RedisStore:
        def get(self, key: str) -> bytes | None: ...
        def set(self, key: str, value: bytes, ttl_seconds: float) -> None: ...
        def delete(self, key: str) -> None: ...

    AIFactory.set_cache_store(RedisStore())

The library never imports a client for any external store. Every user of the
store treats it as unreliable: a `get` that raises is a miss, a `set` or
`delete` that raises is logged and ignored, so a store outage can never fail
a call.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from typing import Protocol, runtime_checkable

_LOGGER: logging.Logger = logging.getLogger(__name__)

DEFAULT_MAX_ENTRIES: int = 10_000


@runtime_checkable
class CacheStore(Protocol):
    """Three methods over bytes; see the module docstring."""

    def get(self, key: str) -> bytes | None:
        """The stored value, or None when absent or expired."""

    def set(self, key: str, value: bytes, ttl_seconds: float) -> None:
        """Stores the value for ttl_seconds; replaces an existing entry."""

    def delete(self, key: str) -> None:
        """Removes the entry, if present."""


class InMemoryCacheStore:
    """
    The default store: a thread-safe LRU dict with an expiry per entry.

    Attributes:
        max_entries: Entries kept before the least recently used is evicted.
        callable_now: Monotonic clock; tests replace it.
    """

    def __init__(
        self,
        max_entries: int = DEFAULT_MAX_ENTRIES,
        *,
        callable_now: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be at least 1")
        self.max_entries: int = max_entries
        self.callable_now: Callable[[], float] = callable_now
        self._lock: threading.Lock = threading.Lock()
        # key -> (expires_at, value); insertion order is recency.
        self._entries: OrderedDict[str, tuple[float, bytes]] = OrderedDict()

    def get(self, key: str) -> bytes | None:
        with self._lock:
            entry: tuple[float, bytes] | None = self._entries.get(key)
            if entry is None:
                # Early return: absent.
                return None
            float_expires, value = entry
            if float_expires <= self.callable_now():
                del self._entries[key]
                # Early return: expired, and now gone.
                return None
            self._entries.move_to_end(key)
            # Normal return with the live value.
            return value

    def set(self, key: str, value: bytes, ttl_seconds: float) -> None:
        if ttl_seconds <= 0:
            self.delete(key)
            # Early return: a non-positive TTL stores nothing.
            return None
        with self._lock:
            self._entries[key] = (self.callable_now() + ttl_seconds, value)
            self._entries.move_to_end(key)
            # Loop while over capacity, dropping the least recently used.
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)
        # Normal return after storing.
        return None

    def delete(self, key: str) -> None:
        with self._lock:
            self._entries.pop(key, None)
        # Normal return after removing.
        return None

    def clear(self) -> None:
        """Drops every entry (tests, or a deliberate flush)."""
        with self._lock:
            self._entries.clear()

    def __len__(self) -> int:
        # Normal return with the entry count, expired entries included.
        return len(self._entries)


_LOCK: threading.Lock = threading.Lock()
_STORE: CacheStore | None = None


def get_cache_store() -> CacheStore:
    """The process-wide store, created on first use."""
    global _STORE
    with _LOCK:
        if _STORE is None:
            _STORE = InMemoryCacheStore()
        # Normal return with the current store.
        return _STORE


def set_cache_store(store: CacheStore | None) -> None:
    """
    Replaces the process-wide store for every cache in the library. None
    restores the default in-memory store on next use.

    Raises:
        TypeError: The object lacks get, set, or delete.
    """
    global _STORE
    if store is not None and not isinstance(store, CacheStore):
        raise TypeError(
            "A cache store needs get(key), set(key, value, ttl_seconds), and "
            "delete(key); see ai_api_unified.middleware.cache_store.CacheStore."
        )
    with _LOCK:
        _STORE = store
    # Normal return after swapping the store.
    return None


def store_get(key: str) -> bytes | None:
    """A get that fails open: a raising store is a miss."""
    try:
        # Normal return with the store's answer.
        return get_cache_store().get(key)
    except Exception as exception:
        _LOGGER.warning(
            "cache_store_get_failed key_prefix=%s error_type=%s",
            key.split(":", 1)[0],
            exception.__class__.__name__,
        )
        # Early return: treated as a miss.
        return None


def store_set(key: str, value: bytes, ttl_seconds: float) -> None:
    """A set that fails open: a raising store is logged and ignored."""
    try:
        get_cache_store().set(key, value, ttl_seconds)
    except Exception as exception:
        _LOGGER.warning(
            "cache_store_set_failed key_prefix=%s error_type=%s",
            key.split(":", 1)[0],
            exception.__class__.__name__,
        )
    # Normal return either way.
    return None


def store_delete(key: str) -> None:
    """A delete that fails open."""
    try:
        get_cache_store().delete(key)
    except Exception as exception:
        _LOGGER.warning(
            "cache_store_delete_failed key_prefix=%s error_type=%s",
            key.split(":", 1)[0],
            exception.__class__.__name__,
        )
    # Normal return either way.
    return None
