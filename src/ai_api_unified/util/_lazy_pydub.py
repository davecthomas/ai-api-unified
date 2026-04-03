"""
Lazy loader for pydub.

Importing pydub normally triggers a DeprecationWarning because it
pulls in the soon-to-be-removed std-lib module `audioop`.  We avoid
that by importing pydub only when audio functions are actually used.
Usage:
    from ai_api_unified.util._lazy_pydub import AudioSegment, play
"""

from __future__ import annotations

import importlib
import sys
import types
import warnings
from typing import Any

# Python 3.13 removed the 'audioop' module.
# We interpret "audioop-lts" as a drop-in replacement if installed.
if sys.version_info >= (3, 13):
    try:
        import audioop
    except ImportError:
        try:
            import audioop_lts as audioop

            sys.modules["audioop"] = audioop
        except ImportError:
            pass


class _LazyPydub(types.ModuleType):
    """Proxy module that imports pydub upon first attribute access."""

    _pydub: types.ModuleType | None = None

    def _load(self) -> types.ModuleType:
        if self._pydub is None:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="'audioop' is deprecated",
                    category=DeprecationWarning,
                )
                self._pydub = importlib.import_module("pydub")
        return self._pydub

    def __getattr__(self, item: str) -> Any:  # noqa: D401
        return getattr(self._load(), item)


# Module-level proxy so callers can do plain attribute access.
_lazy_pydub = _LazyPydub("pydub_lazy")

# Re-export the two most common symbols.
AudioSegment: Any = _lazy_pydub.AudioSegment


def play(*args, **kwargs):
    """Lazily import and call pydub.playback.play."""
    return importlib.import_module("pydub.playback").play(*args, **kwargs)


def get_CouldntDecodeError():
    """Lazily import and return pydub.exceptions.CouldntDecodeError."""
    return importlib.import_module("pydub.exceptions").CouldntDecodeError


# For convenience, provide a property-like alias
class _CouldntDecodeErrorProxy:
    def __getattr__(self, name):
        if name == "CouldntDecodeError":
            return get_CouldntDecodeError()
        raise AttributeError(f"module has no attribute {name!r}")


CouldntDecodeError = _CouldntDecodeErrorProxy().CouldntDecodeError
