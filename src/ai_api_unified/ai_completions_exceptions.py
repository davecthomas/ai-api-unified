"""
Typed exceptions for completions-specific runtime failures.

These exceptions centralize structured-response token-limit failures so every
provider can surface the same actionable error contract to callers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ai_api_unified.ai_provider_exceptions import (
    AiFallbackReason,
    AiProviderRequestError,
)

if TYPE_CHECKING:
    from ai_api_unified.ai_base import AITokenUsage


class StructuredResponseTokenLimitError(RuntimeError):
    """
    Raised when structured generation cannot complete because the response token
    budget is too small or the provider truncates the JSON output.
    """

    def __init__(
        self,
        *,
        message: str,
        provider_name: str,
        model_name: str,
        max_response_tokens: int,
        minimum_supported_tokens: int,
        finish_reason: str | None = None,
        raw_output_char_count: int | None = None,
        usage: AITokenUsage | None = None,
    ) -> None:
        """
        Stores stable token-limit failure metadata for callers and observability.

        Args:
            message: Human-readable error message explaining the failure and remediation.
            provider_name: Library provider or engine name that raised the error.
            model_name: Concrete provider model name used for the structured call.
            max_response_tokens: Token limit requested by the caller for structured output.
            minimum_supported_tokens: Library-enforced minimum structured token budget.
            finish_reason: Optional provider finish reason that indicated truncation.
            raw_output_char_count: Optional raw provider output size when truncation occurred.
            usage: Provider-reported token usage of the truncated answer, when
                the provider returned one, so the billed failure is costed.
                None for a preflight failure, where no call was made.

        Returns:
            None after initializing the typed structured token-limit exception.
        """
        self.provider_name: str = provider_name
        self.model_name: str = model_name
        self.max_response_tokens: int = max_response_tokens
        self.minimum_supported_tokens: int = minimum_supported_tokens
        self.finish_reason: str | None = finish_reason
        self.raw_output_char_count: int | None = raw_output_char_count
        self.usage: AITokenUsage | None = usage
        super().__init__(message)
        # Normal return after storing the structured token-limit failure metadata.
        return None


class StructuredOutputValidationError(AiProviderRequestError, ValueError):
    """
    Raised when a structured answer came back billed but unusable: not valid
    JSON, not a JSON object, or failing the response model.

    A ValueError, as this failure always was, so existing handlers keep
    working. Also an AiProviderRequestError with
    fallback_reason=INVALID_OUTPUT, so a fallback chain or router that opts
    into that reason tries the next model. Carries the answer's usage so the
    request's cost report bills the discarded call.

    Attributes:
        usage: Provider-reported token usage of the discarded answer, or None
            when the provider reported none.
        raw_text: The answer's text after output redaction, for diagnostics.
        finish_reason: The normalized finish reason of the answer, as a string.
    """

    def __init__(
        self,
        message: str,
        *,
        usage: AITokenUsage | None = None,
        raw_text: str | None = None,
        finish_reason: str | None = None,
        provider_engine: str | None = None,
    ) -> None:
        super().__init__(
            message,
            status_code=None,
            provider_engine=provider_engine,
            fallback_reason=AiFallbackReason.INVALID_OUTPUT,
            account_wide=False,
        )
        self.usage: AITokenUsage | None = usage
        self.raw_text: str | None = raw_text
        self.finish_reason: str | None = finish_reason
        # Normal return after storing the discarded answer's metadata.
        return None
