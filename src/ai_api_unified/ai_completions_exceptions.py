"""
Typed exceptions for completions-specific runtime failures.

These exceptions centralize structured-response token-limit failures so every
provider can surface the same actionable error contract to callers.
"""

from __future__ import annotations


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

        Returns:
            None after initializing the typed structured token-limit exception.
        """
        self.provider_name: str = provider_name
        self.model_name: str = model_name
        self.max_response_tokens: int = max_response_tokens
        self.minimum_supported_tokens: int = minimum_supported_tokens
        self.finish_reason: str | None = finish_reason
        self.raw_output_char_count: int | None = raw_output_char_count
        super().__init__(message)
        # Normal return after storing the structured token-limit failure metadata.
        return None
