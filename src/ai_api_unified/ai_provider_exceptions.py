"""
Typed exceptions for optional provider dependency and runtime failures.

These exceptions centralize provider-loading failures so factories can expose
consistent, actionable errors to callers.
"""

from __future__ import annotations

from enum import Enum


class AiFallbackReason(str, Enum):
    """
    Why a failed provider request is a candidate for retry on another model.

    Set on AiProviderRequestError by each engine from the provider's own error
    codes, since an HTTP status alone does not separate a rate limit from an
    exhausted quota. Errors that a different model would not fix (validation,
    authentication, a content refusal, a caller-set timeout) carry no reason.

    Members:
        UNAVAILABLE: The provider or model cannot serve right now: 5xx,
            overloaded, model not ready, or the connection failed.
        RATE_LIMITED: A transient 429; the engine's own backoff runs first.
        QUOTA_EXHAUSTED: Billing, credit, or quota is used up, so retrying
            the same model is pointless.
        MODEL_UNAVAILABLE: The model id is unknown, retired, or not offered
            in the configured region. Usually a configuration error, so a
            fallback layer should not act on it unless told to.
    """

    UNAVAILABLE = "unavailable"
    RATE_LIMITED = "rate_limited"
    QUOTA_EXHAUSTED = "quota_exhausted"
    MODEL_UNAVAILABLE = "model_unavailable"


# Sentinel for "derive fallback_reason from status_code"; distinct from None,
# which means the error is not a fallback trigger.
DERIVE_FALLBACK_REASON: object = object()

# Reasons the same model may clear on its own, so in-engine backoff applies.
FROZENSET_TRANSIENT_FALLBACK_REASONS: frozenset[AiFallbackReason] = frozenset(
    {AiFallbackReason.UNAVAILABLE, AiFallbackReason.RATE_LIMITED}
)


def classify_fallback_reason_by_status(
    status_code: int | None,
) -> AiFallbackReason | None:
    """
    Maps an HTTP status to a fallback reason using only the status.

    Engines refine this with provider error codes; this is the shared
    baseline for statuses whose meaning does not vary by provider.

    Args:
        status_code: Provider HTTP status, or None for a connection failure.

    Returns:
        The reason, or None when the status is not a fallback trigger.
    """
    if status_code is None:
        # Early return: no status means the provider was unreachable.
        return AiFallbackReason.UNAVAILABLE
    if status_code == 429:
        # Early return for a rate limit.
        return AiFallbackReason.RATE_LIMITED
    if status_code == 404:
        # Early return for an unknown model or endpoint.
        return AiFallbackReason.MODEL_UNAVAILABLE
    if status_code in (500, 502, 503, 504, 529):
        # Early return for a server-side outage or overload.
        return AiFallbackReason.UNAVAILABLE
    # Normal return: other statuses are caller or request problems.
    return None


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
        fallback_reason: Why another model might serve this request, or None
            when a different model would not help; see AiFallbackReason.
            Engines set it from provider error codes. When not supplied, it
            is derived from status_code alone; pass None explicitly to mark
            an error that must not trigger a fallback.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        provider_engine: str | None = None,
        fallback_reason: AiFallbackReason | None | object = DERIVE_FALLBACK_REASON,
    ) -> None:
        super().__init__(message)
        self.status_code: int | None = status_code
        self.provider_engine: str | None = provider_engine
        self.fallback_reason: AiFallbackReason | None
        if fallback_reason is DERIVE_FALLBACK_REASON:
            self.fallback_reason = classify_fallback_reason_by_status(status_code)
        else:
            self.fallback_reason = fallback_reason  # type: ignore[assignment]

    @property
    def is_transient(self) -> bool:
        """True when the same model may recover, so backoff is worth trying."""
        # Normal return with the transient-reason membership.
        return self.fallback_reason in FROZENSET_TRANSIENT_FALLBACK_REASONS
