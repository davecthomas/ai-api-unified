"""
PII Anonymization Layer for ai_api_unified.

This module provides the AiApiPiiMiddleware to prevent PII
from being sent to or received from AI providers. It dynamically loads
the necessary redaction engine based on the environment.
"""

from __future__ import annotations

import importlib
import logging
from typing import Any

from .impl.base_redactor import BaseRedactorLayer, RedactionResult
from .middleware import InstrumentedAiApiMiddleware, MiddlewareProcessingResult
from .middleware_config import MiddlewareConfig, PiiRedactionSettingsModel
from .redaction_exceptions import (
    PiiRedactionDependencyUnavailableError,
    PiiRedactionError,
    PiiRedactionRuntimeError,
)

logger: logging.Logger = logging.getLogger(__name__)

MIDDLEWARE_MODULE: str = ".impl._presidio_redactor"
PII_REDACTION: str = "pii_redaction"
DIRECTION_KEY: str = "direction"
DIRECTION_INPUT_ONLY: str = "input_only"
DIRECTION_OUTPUT_ONLY: str = "output_only"
DIRECTION_INPUT_OUTPUT: str = "input_output"
DIRECTION_NONE: str = "none"
STRICT_MODE_DEFAULT: bool = False
STRICT_MODE_DEPENDENCY_ERROR_MESSAGE: str = (
    "PII redaction strict mode is enabled, but redaction dependencies are unavailable. "
    "Install `ai-api-unified[middleware-pii-redaction]` and ensure required "
    "runtime assets are installed (for example, spaCy model `en_core_web_lg`)."
)


def _get_configured_redactor(
    configuration: PiiRedactionSettingsModel,
    bool_strict_mode: bool,
) -> BaseRedactorLayer:
    """
    Factory function to instantiate the correct redactor implementation.
    Attempts to dynamically load the Presidio implementation.
    If dependencies are missing, behavior is controlled by strict mode.

    Args:
        configuration: Typed PII configuration settings model.
        bool_strict_mode: Enables fail-closed dependency behavior when true.

    Returns:
        An instantiated subclass of BaseRedactorLayer (either PiiRedactor or NoOpRedactor).
    """
    try:
        # Dynamically import the Presidio implementation to avoid static type checker headaches
        # in the core library and to keep the dependency truly optional.
        module: Any = importlib.import_module(MIDDLEWARE_MODULE, package=__package__)
        return module.PiiRedactor(dict_config=configuration.model_dump())
    except ModuleNotFoundError as exception:
        if bool_strict_mode:
            logger.critical(STRICT_MODE_DEPENDENCY_ERROR_MESSAGE)
            # Early return via exception because strict mode forbids fail-open dependency fallback.
            raise PiiRedactionDependencyUnavailableError(
                STRICT_MODE_DEPENDENCY_ERROR_MESSAGE
            ) from exception
        logger.warning(
            "PII redaction dependencies are unavailable; falling back to NoOpRedactor "
            "because strict mode is disabled."
        )
        # Fall back to the No-Op implementation only when strict mode is disabled.
        from .impl.noop_redactor import NoOpRedactor

        return NoOpRedactor()


class AiApiPiiMiddleware(InstrumentedAiApiMiddleware):
    """
    Middleware to intercept and sanitize inputs and outputs across the AI API.
    """

    def __init__(self) -> None:
        config: MiddlewareConfig = MiddlewareConfig()
        pii_settings: PiiRedactionSettingsModel | None = config.get_middleware_settings(
            PII_REDACTION
        )

        self.bool_enabled: bool = pii_settings is not None
        self.bool_strict_mode: bool = STRICT_MODE_DEFAULT
        self._redactor: BaseRedactorLayer | None

        if self.bool_enabled and pii_settings is not None:
            self.str_direction = pii_settings.direction
            self.bool_strict_mode = pii_settings.strict_mode
            self._redactor = _get_configured_redactor(
                configuration=pii_settings,
                bool_strict_mode=self.bool_strict_mode,
            )
        else:
            self.str_direction = DIRECTION_NONE
            self._redactor = None

    @property
    def str_middleware_name(self) -> str:
        """
        Returns the stable middleware identifier used by shared observability hooks.

        Args:
            None

        Returns:
            Stable middleware name used in audit and timing log payloads.
        """
        # Normal return with the canonical middleware identifier.
        return PII_REDACTION

    def _sanitize_text_with_metadata(
        self,
        str_text: str,
    ) -> MiddlewareProcessingResult:
        """
        Runs PII redaction and translates the result into shared middleware metadata.

        Args:
            str_text: Raw text processed by the middleware.

        Returns:
            MiddlewareProcessingResult containing sanitized text plus PII-specific
            security-action metadata for the shared observability wrapper.
        """
        if self._redactor is None:
            # Early return because there is no configured redactor instance.
            return MiddlewareProcessingResult(str_output_text=str_text)
        redaction_result: RedactionResult = self._redactor.sanitize_with_result(
            str_text
        )
        # Normal return with translated shared observability metadata.
        return MiddlewareProcessingResult(
            str_output_text=redaction_result.str_sanitized_text,
            bool_security_control_applied=redaction_result.int_redaction_count > 0,
            int_security_action_count=redaction_result.int_redaction_count,
            tuple_str_categories=tuple(redaction_result.list_str_detected_categories),
        )

    def process_input(self, str_text: str) -> str:
        """
        Filters the raw prompt bound for an AI provider to remove sensitive PII data.

        Args:
            str_text: The raw text string (prompt or context) intended for the LLM.

        Returns:
            A string containing the sanitized text with PII replaced by safe placeholder tags,
            or the original string if the middleware direction does not cover inputs.
        """
        if (
            self.bool_enabled
            and self.str_direction in (DIRECTION_INPUT_ONLY, DIRECTION_INPUT_OUTPUT)
            and self._redactor
        ):
            try:
                # Normal return with sanitized input text from the redaction result contract.
                return self._execute_with_observability(
                    str_text=str_text,
                    str_direction="input",
                    callable_execute=self._sanitize_text_with_metadata,
                )
            except PiiRedactionError:
                raise
            except Exception as exception:
                logger.exception(
                    "Unexpected PII redaction failure while processing input text."
                )
                raise PiiRedactionRuntimeError(
                    "PII redaction failed while processing input text."
                ) from exception
        return str_text

    def process_output(self, str_text: str) -> str:
        """
        Filters the raw completion returned by an AI provider to prevent PII ingestion/logging.

        Args:
            str_text: The raw text string (completion or response) returned by the LLM.

        Returns:
            A string containing the sanitized text with PII replaced by safe placeholder tags,
            or the original string if the middleware direction does not cover outputs.
        """
        if (
            self.bool_enabled
            and self.str_direction in (DIRECTION_OUTPUT_ONLY, DIRECTION_INPUT_OUTPUT)
            and self._redactor
        ):
            try:
                # Normal return with sanitized output text from the redaction result contract.
                return self._execute_with_observability(
                    str_text=str_text,
                    str_direction="output",
                    callable_execute=self._sanitize_text_with_metadata,
                )
            except PiiRedactionError:
                raise
            except Exception as exception:
                logger.exception(
                    "Unexpected PII redaction failure while processing output text."
                )
                raise PiiRedactionRuntimeError(
                    "PII redaction failed while processing output text."
                ) from exception
        return str_text
