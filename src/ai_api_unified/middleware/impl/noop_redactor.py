import logging

from .base_redactor import BaseRedactorLayer, RedactionResult

logger: logging.Logger = logging.getLogger(__name__)

CONSTANT_NOOP_CACHE_NAMESPACE: str = "noop_redactor"


class NoOpRedactor(BaseRedactorLayer):
    """
    A fallback redactor that does absolutely nothing.
    This allows the middleware pipeline to remain intact even if Presidio is not installed,
    providing a transparent pass-through for systems not utilizing PII redaction.
    """

    @property
    def str_engine_cache_namespace(self) -> str:
        """
        Returns the cache namespace for the no-op redactor implementation.

        Args:
            None

        Returns:
            Stable namespace string identifying this implementation family.
        """
        # Normal return with no-op namespace identity.
        return CONSTANT_NOOP_CACHE_NAMESPACE

    @property
    def bool_uses_shared_engine_cache(self) -> bool:
        """
        Signals whether this implementation uses shared detector/anonymizer caches.

        Args:
            None

        Returns:
            Always returns False because no-op redaction does not initialize engines.
        """
        # Normal return because no-op redaction does not use reusable engines.
        return False

    @property
    def tuple_str_engine_cache_identity(self) -> tuple[str, ...]:
        """
        Returns deterministic cache identity components for this implementation.

        Args:
            None

        Returns:
            Single-item tuple containing the no-op cache namespace.
        """
        # Normal return with static no-op cache identity.
        return (self.str_engine_cache_namespace,)

    def sanitize_with_result(self, str_text: str) -> RedactionResult:
        """
        No-op implementation that intentionally bypasses redaction logic.

        Args:
            str_text: The raw text string intended for the AI provider.

        Returns:
            A RedactionResult where sanitized text matches the original input and
            category metadata lists are empty.
        """
        redaction_result: RedactionResult = RedactionResult(
            str_sanitized_text=str_text,
            int_redaction_count=0,
        )
        # Normal return with no-op redaction metadata.
        return redaction_result

    def sanitize_text(self, str_text: str) -> str:
        """
        No-op implementation that intentionally bypasses redaction logic.

        Args:
            str_text: The raw text string intended for the AI provider.

        Returns:
            The exact, unmodified string passed as input.
        """
        redaction_result: RedactionResult = self.sanitize_with_result(str_text)
        # Normal return with pass-through text.
        return redaction_result.str_sanitized_text
