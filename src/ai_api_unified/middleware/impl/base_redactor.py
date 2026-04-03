from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class RedactionResult:
    """
    Provider-agnostic redaction result contract returned by middleware redactors.

    Attributes:
        str_sanitized_text: Final sanitized text produced by the redactor.
        list_str_detected_categories: Canonical middleware categories detected for this text.
        int_redaction_count: Number of redacted spans emitted into the final sanitized text.
    """

    str_sanitized_text: str
    list_str_detected_categories: list[str] = field(default_factory=list)
    int_redaction_count: int = 0


class BaseRedactorLayer(Protocol):
    """
    Protocol defining the required interface for any PII text redactor implementation.
    This guarantees that the middleware (AiApiPiiMiddleware) doesn't need to know
    about the underlying provider (Presidio, regex, etc.).
    """

    @property
    def str_engine_cache_namespace(self) -> str:
        """
        Identifies the concrete redaction engine family for cache partitioning.

        Args:
            None

        Returns:
            Stable namespace string used to isolate cache entries by implementation
            family (for example, "presidio_pii_redactor").
        """
        ...

    @property
    def bool_uses_shared_engine_cache(self) -> bool:
        """
        Signals whether this redactor reuses shared detector/anonymizer engines.

        Args:
            None

        Returns:
            True when the implementation expects caller-safe shared engine caching,
            or False when no reusable engine cache is used.
        """
        ...

    @property
    def tuple_str_engine_cache_identity(self) -> tuple[str, ...]:
        """
        Provides a deterministic cache identity key for engine reuse decisions.

        Args:
            None

        Returns:
            Tuple of stable string components that uniquely identify the effective
            engine configuration for this redactor instance.
        """
        ...

    def sanitize_with_result(self, str_text: str) -> RedactionResult:
        """
        Filters text and returns a structured redaction result.

        Args:
            str_text: The raw text string to be sanitized.

        Returns:
            A RedactionResult containing the sanitized text plus category metadata.
        """
        ...

    def sanitize_text(self, str_text: str) -> str:
        """
        Filters the given string to remove or redact sensitive information.

        Args:
            str_text: The raw text string to be sanitized.

        Returns:
            A string containing the sanitized text with sensitive data redacted.
        """
        ...
