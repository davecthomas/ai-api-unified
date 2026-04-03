"""
Typed exceptions for optional provider dependency and runtime failures.

These exceptions centralize provider-loading failures so factories can expose
consistent, actionable errors to callers.
"""

from __future__ import annotations


class AiProviderError(RuntimeError):
    """
    Base exception for all provider resolution and loading failures.
    """


class AiProviderDependencyUnavailableError(AiProviderError):
    """
    Raised when a selected provider requires an optional dependency extra that
    is not installed in the current environment.
    """


class AiProviderConfigurationError(AiProviderError):
    """
    Raised when provider metadata or engine selection is invalid.
    """


class AiProviderRuntimeError(AiProviderError):
    """
    Raised when provider loading fails due to runtime issues unrelated to
    missing dependency extras.
    """
