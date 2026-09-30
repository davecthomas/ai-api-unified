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


class _DeriveFallbackReason:
    """Sentinel type: derive fallback_reason from status_code."""


# Sentinel for "derive fallback_reason from status_code"; distinct from None,
# which means the error is not a fallback trigger.
DERIVE_FALLBACK_REASON: _DeriveFallbackReason = _DeriveFallbackReason()

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
    baseline for statuses whose meaning does not vary by provider. A missing
    status is not a trigger here: it covers connection failures, which are
    one, but also client-side timeouts, missing credentials, and malformed
    responses, which are not. Engines that can tell them apart pass the
    reason explicitly (see classify_transport_fallback_reason).

    Args:
        status_code: Provider HTTP status, or None when none was available.

    Returns:
        The reason, or None when the status is not a fallback trigger.
    """
    if status_code is None:
        # Early return: without a status the failure cannot be classified.
        return None
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


def classify_transport_fallback_reason(bool_timeout: bool) -> AiFallbackReason | None:
    """
    Classifies a failure that happened before any HTTP status arrived.

    A timeout is the caller's own limit, so it is no reason to change model;
    a failed connection means the host is unreachable.

    Args:
        bool_timeout: True when the failure was a client-side timeout.

    Returns:
        None for a timeout, UNAVAILABLE for a connection failure.
    """
    # Normal return with the transport classification.
    return None if bool_timeout else AiFallbackReason.UNAVAILABLE


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
        fallback_reason: (
            AiFallbackReason | None | _DeriveFallbackReason
        ) = DERIVE_FALLBACK_REASON,
    ) -> None:
        super().__init__(message)
        self.status_code: int | None = status_code
        self.provider_engine: str | None = provider_engine
        self.fallback_reason: AiFallbackReason | None
        if isinstance(fallback_reason, _DeriveFallbackReason):
            self.fallback_reason = classify_fallback_reason_by_status(status_code)
        else:
            self.fallback_reason = fallback_reason

    @property
    def is_transient(self) -> bool:
        """True when the same model may recover, so backoff is worth trying."""
        # Normal return with the transient-reason membership.
        return self.fallback_reason in FROZENSET_TRANSIENT_FALLBACK_REASONS
