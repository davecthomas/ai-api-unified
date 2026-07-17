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


class AiProviderCapabilityUnsupportedError(AiProviderError):
    """
    Raised when a caller requests an operation or input modality that the
    configured provider model does not support, per its capabilities descriptor.
    """


class AiProviderRequestError(AiProviderRuntimeError):
    """
    Raised when a provider API request fails with an HTTP-level error.

    Carries the provider HTTP status code so caller-owned backoff logic can
    classify 429/5xx/529 responses uniformly across engines.

    Attributes:
        status_code: HTTP status code reported by the provider, or None when
            the failure happened before a status was available (for example a
            connection error or client-side timeout).
        provider_engine: Engine selector token of the provider that failed.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        provider_engine: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code: int | None = status_code
        self.provider_engine: str | None = provider_engine
