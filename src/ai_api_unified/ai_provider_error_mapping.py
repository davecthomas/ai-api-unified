# ai_provider_error_mapping.py
"""
Maps provider SDK errors to AiProviderRequestError for the clients that do
not map their own: embeddings, images, videos, and voice.

Completions engines map their SDK's errors inside each engine, with the
codes their providers document. The media clients share these SDKs, so
they share the same mapping here instead of raising the SDK's exception, or
a RuntimeError wrapped around it, to the caller. Every provider failure
then raises one type that says why the call failed (`status_code`) and
whether another model could serve it (`fallback_reason`).

The mapper reads the exception and its cause chain, because the media
clients often wrap the SDK error in a RuntimeError with context of their
own. An exception from no known SDK, or a validation error the library
raised itself, is returned as None and propagates unchanged.
"""

from __future__ import annotations

from typing import Any

from .ai_provider_exceptions import (
    AiFallbackReason,
    AiProviderRequestError,
    classify_fallback_reason_by_status,
    classify_transport_fallback_reason,
)

# How far down the cause chain to look for the SDK's own error.
INT_MAX_CAUSE_DEPTH: int = 5


def _cause_chain(exception: BaseException) -> list[BaseException]:
    """The exception, then its explicit or implicit causes, nearest first."""
    list_chain: list[BaseException] = []
    current: BaseException | None = exception
    # Loop down the chain, stopping at a cycle or the depth limit.
    while current is not None and len(list_chain) < INT_MAX_CAUSE_DEPTH:
        if any(current is seen for seen in list_chain):
            break
        list_chain.append(current)
        current = current.__cause__ or current.__context__
    # Normal return with the chain.
    return list_chain


def _module_of(exception: BaseException) -> str:
    """The top-level package that defined the exception's class."""
    # Normal return with the module path, or "" when absent.
    return type(exception).__module__ or ""


def _int_status(value: Any) -> int | None:
    """An HTTP status from an attribute that may be an int, enum, or digits."""
    value = getattr(value, "value", value)
    if isinstance(value, int) and not isinstance(value, bool):
        # Early return with the integer status.
        return value
    if isinstance(value, str) and value.isdigit():
        # Early return with the parsed status.
        return int(value)
    # Normal return: no status here.
    return None


def _from_openai(
    exception: BaseException, str_engine: str
) -> AiProviderRequestError | None:
    import openai

    if isinstance(exception, openai.APIStatusError):
        from .completions.ai_openai_completions import AiOpenAICompletions

        # Early return with the status and OpenAI's code-aware reason.
        return AiProviderRequestError(
            f"OpenAI request failed with status {exception.status_code}: "
            f"{exception.message}",
            status_code=exception.status_code,
            provider_engine=str_engine,
            fallback_reason=AiOpenAICompletions._classify_fallback_reason(exception),
        )
    if isinstance(exception, (openai.APITimeoutError, openai.APIConnectionError)):
        # Early return for a failure before any status.
        return AiProviderRequestError(
            f"OpenAI request failed before a status was available: {exception}",
            status_code=None,
            provider_engine=str_engine,
            fallback_reason=classify_transport_fallback_reason(
                isinstance(exception, openai.APITimeoutError)
            ),
        )
    # Normal return: not an OpenAI transport error.
    return None


def _from_google(
    exception: BaseException, str_engine: str
) -> AiProviderRequestError | None:
    from .ai_google_base import classify_gemini_fallback_reason

    int_status: int | None = None
    # Loop over the attributes Google errors carry a status in.
    for str_attr in ("code", "status"):
        int_status = _int_status(getattr(exception, str_attr, None))
        if int_status is not None:
            break
    if int_status is None:
        int_status = _int_status(
            getattr(getattr(exception, "response", None), "status_code", None)
        )
    reason: AiFallbackReason | None = classify_gemini_fallback_reason(
        int_status, str(exception)
    )
    # Normal return with the typed error.
    return AiProviderRequestError(
        f"Google request failed: {exception}",
        status_code=int_status,
        provider_engine=str_engine,
        fallback_reason=reason,
        # Google quotas are per model, so another model on the key can serve;
        # only a rejected key is account-wide.
        account_wide=reason is AiFallbackReason.AUTH_REJECTED,
    )


def _from_botocore(
    exception: BaseException, str_engine: str
) -> AiProviderRequestError | None:
    from botocore.exceptions import BotoCoreError, ClientError, ReadTimeoutError

    if isinstance(exception, ClientError):
        from .completions.ai_bedrock_completions import AiBedrockCompletions

        dict_response: dict[str, Any] = getattr(exception, "response", None) or {}
        int_status: int | None = _int_status(
            dict_response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        )
        str_code, _ = AiBedrockCompletions._client_error_code_and_message(exception)
        # Early return with Bedrock's code-aware reason.
        return AiProviderRequestError(
            f"Bedrock request failed: {exception}",
            status_code=int_status,
            provider_engine=str_engine,
            fallback_reason=AiBedrockCompletions._classify_fallback_reason(
                exception, int_status
            ),
            account_wide=(
                False
                if str_code in AiBedrockCompletions.FROZENSET_PER_MODEL_ERROR_CODES
                else None
            ),
        )
    if isinstance(exception, BotoCoreError):
        # Early return for a failure before any status.
        return AiProviderRequestError(
            f"Bedrock request failed before a status was available: {exception}",
            status_code=None,
            provider_engine=str_engine,
            fallback_reason=classify_transport_fallback_reason(
                isinstance(exception, ReadTimeoutError)
            ),
        )
    # Normal return: not a botocore transport error.
    return None


def _from_httpx(
    exception: BaseException, str_engine: str
) -> AiProviderRequestError | None:
    import httpx

    if isinstance(exception, httpx.HTTPStatusError):
        int_status: int | None = exception.response.status_code
        # Early return with the status-derived reason.
        return AiProviderRequestError(
            f"Request failed with status {int_status}: {exception}",
            status_code=int_status,
            provider_engine=str_engine,
        )
    if isinstance(exception, httpx.TransportError):
        # Early return for a failure before any status.
        return AiProviderRequestError(
            f"Request failed before a status was available: {exception}",
            status_code=None,
            provider_engine=str_engine,
            fallback_reason=classify_transport_fallback_reason(
                isinstance(exception, httpx.TimeoutException)
            ),
        )
    # Normal return: not an httpx transport error.
    return None


def _from_status_attribute(
    exception: BaseException, str_engine: str, str_vendor: str
) -> AiProviderRequestError | None:
    """Voyage and ElevenLabs errors carry an HTTP status attribute."""
    int_status: int | None = None
    # Loop over the attribute names these SDKs use.
    for str_attr in ("http_status", "status_code"):
        int_status = _int_status(getattr(exception, str_attr, None))
        if int_status is not None:
            break
    # Normal return with the status-derived reason (None without a status).
    return AiProviderRequestError(
        f"{str_vendor} request failed: {exception}",
        status_code=int_status,
        provider_engine=str_engine,
        fallback_reason=classify_fallback_reason_by_status(int_status),
    )


def _map_one(
    exception: BaseException, str_engine: str
) -> AiProviderRequestError | None:
    """Maps one exception from a known SDK, or None."""
    str_module: str = _module_of(exception)
    if str_module.startswith("openai"):
        # Early return with the OpenAI mapping.
        return _from_openai(exception, str_engine)
    if str_module.startswith("google"):
        # Early return with the Google mapping.
        return _from_google(exception, str_engine)
    if str_module.startswith("botocore"):
        # Early return with the botocore mapping.
        return _from_botocore(exception, str_engine)
    if str_module.startswith("httpx"):
        # Early return with the httpx mapping.
        return _from_httpx(exception, str_engine)
    if str_module.startswith("voyageai"):
        # Early return with the Voyage mapping.
        return _from_status_attribute(exception, str_engine, "Voyage")
    if str_module.startswith("elevenlabs"):
        # Early return with the ElevenLabs mapping.
        return _from_status_attribute(exception, str_engine, "ElevenLabs")
    # Normal return: not an SDK this module maps.
    return None


def map_provider_exception(
    exception: BaseException, *, provider_engine: str
) -> AiProviderRequestError | None:
    """
    The typed request error for a provider failure, or None.

    Args:
        exception: What the provider call raised.
        provider_engine: Engine label for the error.

    Returns:
        The exception itself when it is already an AiProviderRequestError, a
        copy of one found deeper in the chain, a new one mapped from the
        first SDK error in the chain, or
        None when nothing in the chain came from a known SDK.
    """
    # Loop down the cause chain for a typed error or an SDK error.
    for int_depth, candidate in enumerate(_cause_chain(exception)):
        if isinstance(candidate, AiProviderRequestError):
            if int_depth == 0:
                # Early return: already typed.
                return candidate
            # Early return with a copy, so the caller can raise it from the
            # wrapper without making the chain a cycle.
            return AiProviderRequestError(
                str(exception),
                status_code=candidate.status_code,
                provider_engine=candidate.provider_engine,
                fallback_reason=candidate.fallback_reason,
                account_wide=candidate.account_wide,
            )
        try:
            typed: AiProviderRequestError | None = _map_one(candidate, provider_engine)
        except ImportError:
            # The SDK that raised it is importable by definition; a missing
            # completions module means nothing better to say here.
            typed = None
        if typed is not None:
            # Early return with the mapped error.
            return typed
    # Normal return: nothing in the chain came from a known SDK.
    return None
