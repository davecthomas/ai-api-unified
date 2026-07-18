"""
Lazy loader for pydub.

pydub is an audio-only dependency shipped by the `voice` extra (see
pyproject.toml), and importing it emits SyntaxWarnings from pydub/utils.py
plus a RuntimeWarning when ffmpeg/avconv is absent. Text-only consumers must
never trigger that import, so this module defers it until an audio feature
is actually used: `import ai_api_unified` and every completions/embeddings
path stay pydub-free.

Usage:
    from ai_api_unified.util._lazy_pydub import AudioSegment, play, \
        get_CouldntDecodeError

    segment = AudioSegment.from_file(path)      # imports pydub here
    except get_CouldntDecodeError() as exc:     # evaluated at handling time
"""

from __future__ import annotations

import importlib
import sys
import types
import warnings
from typing import Any

from ai_api_unified.ai_provider_exceptions import (
    AiProviderDependencyUnavailableError,
)

VOICE_EXTRA_INSTALL_HINT: str = (
    "Audio features require the 'voice' extra: "
    "pip install 'ai-api-unified[voice]' (pydub and, on Python 3.13+, "
    "audioop-lts). The azure_tts and elevenlabs extras include it."
)

_pydub_module: types.ModuleType | None = None


def _install_audioop_shim() -> None:
    """
    Registers audioop-lts as 'audioop' on Python 3.13+, where the std-lib
    module was removed. Deferred until pydub loads so importing this module
    stays free of audio dependencies.
    """
    if sys.version_info < (3, 13):
        # Early return because the std-lib audioop still exists.
        return None
    try:
        import audioop  # noqa: F401
    except ImportError:
        try:
            import audioop_lts as audioop  # type: ignore[import-not-found]

            sys.modules["audioop"] = audioop
        except ImportError:
            # pydub's own import will surface the missing-dependency error.
            pass
    # Normal return after the shim attempt.
    return None


def _load_pydub() -> types.ModuleType:
    """
    Imports pydub on first use, raising the typed dependency error when the
    voice extra is not installed.
    """
    global _pydub_module
    if _pydub_module is None:
        _install_audioop_shim()
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="'audioop' is deprecated",
                category=DeprecationWarning,
            )
            try:
                _pydub_module = importlib.import_module("pydub")
            except ImportError as exception:
                # ImportError (not just ModuleNotFoundError) so a broken or
                # partial pydub install also surfaces the install hint.
                raise AiProviderDependencyUnavailableError(
                    VOICE_EXTRA_INSTALL_HINT
                ) from exception
    # Normal return with the loaded pydub module.
    return _pydub_module


class _LazyAudioSegmentProxy:
    """
    Class-like proxy for pydub.AudioSegment.

    Attribute access (from_file, from_raw, silent, ...) and construction
    trigger the real import; holding or annotating with the proxy does not.
    """

    def _real(self) -> Any:
        return _load_pydub().AudioSegment

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._real()(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real(), name)

    def __repr__(self) -> str:
        return "<lazy pydub.AudioSegment proxy>"


AudioSegment: Any = _LazyAudioSegmentProxy()


def play(*args: Any, **kwargs: Any) -> Any:
    """Lazily import and call pydub.playback.play."""
    _load_pydub()
    return importlib.import_module("pydub.playback").play(*args, **kwargs)


def get_CouldntDecodeError() -> type[Exception]:
    """
    Lazily import and return pydub.exceptions.CouldntDecodeError.

    Call this inside the `except` clause expression — Python evaluates it
    only when an exception is being handled, after pydub has already loaded:

        except get_CouldntDecodeError() as exc:
    """
    _load_pydub()
    return importlib.import_module("pydub.exceptions").CouldntDecodeError


def __getattr__(name: str) -> Any:
    """
    Backward-compatible module attribute for eager importers.

    `from ai_api_unified.util._lazy_pydub import CouldntDecodeError` resolves
    the real exception class immediately (importing pydub); library code uses
    get_CouldntDecodeError() instead so imports stay audio-free.
    """
    if name == "CouldntDecodeError":
        return get_CouldntDecodeError()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
