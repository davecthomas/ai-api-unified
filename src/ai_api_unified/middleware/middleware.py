"""Shared middleware interfaces and observability helpers."""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Generic, TypeVar

AUDIT_LOGGER_NAME: str = "ai_api_unified.middleware.audit"
METRICS_LOGGER_NAME: str = "ai_api_unified.middleware.metrics"
SECURITY_CONTROL_AUDIT_EVENT_LOG_MESSAGE: str = (
    "security_control_applied control=%s direction=%s redaction_count=%s "
    "categories=%s"
)
MIDDLEWARE_TIMING_EVENT_LOG_MESSAGE: str = (
    "middleware_execution_timing middleware=%s direction=%s elapsed_ms=%.3f "
    "redaction_count=%s ms_per_redaction=%s categories=%s"
)
MiddlewareReturnType = TypeVar("MiddlewareReturnType")


@dataclass
class MiddlewareProcessingResult:
    """
    Shared metadata contract returned by one middleware execution step.

    Attributes:
        str_output_text: Final output text returned to the caller after middleware processing.
        bool_security_control_applied: True when the middleware applied one or more
            security-relevant actions such as redactions or filter hits.
        int_security_action_count: Number of security-relevant actions applied during
            the middleware execution step.
        tuple_str_categories: Stable tuple of middleware-specific categories associated
            with the security actions, such as canonical PII categories.
    """

    str_output_text: str
    bool_security_control_applied: bool = False
    int_security_action_count: int = 0
    tuple_str_categories: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class MiddlewareExecutionTimingResult(Generic[MiddlewareReturnType]):
    """
    Stores the result of one timed middleware execution helper invocation.

    Attributes:
        typed_result: Arbitrary middleware result payload returned by the executed callable.
        float_elapsed_ms: Wall-clock execution duration in milliseconds.
    """

    typed_result: MiddlewareReturnType
    float_elapsed_ms: float


def execute_middleware_with_timing(
    callable_execute: Callable[[], MiddlewareReturnType],
) -> MiddlewareExecutionTimingResult[MiddlewareReturnType]:
    """
    Executes a middleware callable and measures wall-clock elapsed time.

    Args:
        callable_execute: Zero-argument callable that performs one middleware step and
            returns any typed result payload.

    Returns:
        MiddlewareExecutionTimingResult containing the callable result and elapsed milliseconds.
    """
    float_started_at: float = time.perf_counter()
    typed_result: MiddlewareReturnType = callable_execute()
    float_elapsed_ms: float = (time.perf_counter() - float_started_at) * 1000.0
    # Normal return with the callable result and measured elapsed time.
    return MiddlewareExecutionTimingResult(
        typed_result=typed_result,
        float_elapsed_ms=float_elapsed_ms,
    )


def log_middleware_observability_events(
    str_middleware_name: str,
    str_direction: str,
    float_elapsed_ms: float,
    int_security_action_count: int = 0,
    tuple_str_categories: tuple[str, ...] = (),
    bool_security_control_applied: bool = False,
    audit_logger: logging.Logger | None = None,
    metrics_logger: logging.Logger | None = None,
) -> None:
    """
    Emits shared middleware timing and optional audit logs using metadata only.

    Args:
        str_middleware_name: Stable middleware identifier included in emitted logs.
        str_direction: Middleware execution direction label such as `input`, `output`,
            or another lifecycle direction used by the caller.
        float_elapsed_ms: Wall-clock execution duration in milliseconds.
        int_security_action_count: Count of security-relevant actions applied during
            middleware execution.
        tuple_str_categories: Stable middleware-specific categories associated with
            any security-relevant actions.
        bool_security_control_applied: True when one or more security-relevant actions
            were applied during the middleware step.
        audit_logger: Optional logger override for audit events. Defaults to the shared
            middleware audit logger when omitted.
        metrics_logger: Optional logger override for timing events. Defaults to the shared
            middleware metrics logger when omitted.

    Returns:
        None after metadata-only observability logging completes.
    """
    resolved_audit_logger: logging.Logger = audit_logger or logging.getLogger(
        AUDIT_LOGGER_NAME
    )
    resolved_metrics_logger: logging.Logger = metrics_logger or logging.getLogger(
        METRICS_LOGGER_NAME
    )
    str_ms_per_redaction: str = "none"
    if int_security_action_count > 0:
        str_ms_per_redaction = f"{float_elapsed_ms / int_security_action_count:.3f}"
    if bool_security_control_applied:
        resolved_audit_logger.info(
            SECURITY_CONTROL_AUDIT_EVENT_LOG_MESSAGE,
            str_middleware_name,
            str_direction,
            int_security_action_count,
            tuple_str_categories,
        )
    resolved_metrics_logger.info(
        MIDDLEWARE_TIMING_EVENT_LOG_MESSAGE,
        str_middleware_name,
        str_direction,
        float_elapsed_ms,
        int_security_action_count,
        str_ms_per_redaction,
        tuple_str_categories,
    )
    # Normal return after metadata-only observability logging.
    return None


class AiApiMiddleware(ABC):
    """
    Abstract base class for AI API middleware components.

    This pattern enables a seamless middleware approach for intercepting and
    transforming data sent to and received from AI providers. Concrete
    implementations can provide capabilities such as PII redaction,
    observability, caching, or prompt injection filtering.
    """

    @abstractmethod
    def process_input(self, str_text: str) -> str:
        """
        Processes or sanitizes text bound for an AI provider (e.g., a prompt).
        Must be implemented by concrete middleware classes.

        Args:
            str_text: The raw text string intended for the AI provider.

        Returns:
            A string containing the processed or sanitized prompt text.
        """
        pass

    @abstractmethod
    def process_output(self, str_text: str) -> str:
        """
        Processes or sanitizes text returned by an AI provider (e.g., a completion).
        Must be implemented by concrete middleware classes.

        Args:
            str_text: The raw text string returned by the AI provider.

        Returns:
            A string containing the processed or sanitized completion text.
        """
        pass


class InstrumentedAiApiMiddleware(AiApiMiddleware, ABC):
    """
    Shared concrete middleware base providing timing and audit observability.

    Concrete middleware implementations remain responsible for domain-specific
    processing and for producing a `MiddlewareProcessingResult` describing the
    security-relevant outcome of one middleware execution step.
    """

    audit_logger: logging.Logger = logging.getLogger(AUDIT_LOGGER_NAME)
    metrics_logger: logging.Logger = logging.getLogger(METRICS_LOGGER_NAME)

    @property
    @abstractmethod
    def str_middleware_name(self) -> str:
        """
        Returns the stable middleware identifier used in observability events.

        Args:
            None

        Returns:
            Stable middleware name string included in timing and audit logs.
        """
        ...

    def _log_observability_events(
        self,
        str_direction: str,
        middleware_processing_result: MiddlewareProcessingResult,
        float_elapsed_ms: float,
    ) -> None:
        """
        Emits shared middleware timing and security-control logs using metadata only.

        Args:
            str_direction: Effective middleware direction for this execution (`input` or `output`).
            middleware_processing_result: Shared middleware execution result containing
                output text and security-action metadata.
            float_elapsed_ms: Wall-clock execution duration in milliseconds.

        Returns:
            None after metadata-only observability logging completes.
        """
        log_middleware_observability_events(
            str_middleware_name=self.str_middleware_name,
            str_direction=str_direction,
            float_elapsed_ms=float_elapsed_ms,
            int_security_action_count=middleware_processing_result.int_security_action_count,
            tuple_str_categories=middleware_processing_result.tuple_str_categories,
            bool_security_control_applied=middleware_processing_result.bool_security_control_applied,
            audit_logger=self.audit_logger,
            metrics_logger=self.metrics_logger,
        )
        # Normal return after metadata-only observability logging.
        return None

    def _execute_with_observability(
        self,
        str_text: str,
        str_direction: str,
        callable_execute: Callable[[str], MiddlewareProcessingResult],
    ) -> str:
        """
        Runs one middleware execution step and emits shared observability logs.

        Args:
            str_text: Raw text passed into the middleware step.
            str_direction: Execution direction label for this step (`input` or `output`).
            callable_execute: Middleware-specific callable returning a
                `MiddlewareProcessingResult` for the provided text.

        Returns:
            Final processed text returned by the middleware implementation.
        """
        middleware_timing_result: MiddlewareExecutionTimingResult[
            MiddlewareProcessingResult
        ] = execute_middleware_with_timing(
            callable_execute=lambda: callable_execute(str_text)
        )
        self._log_observability_events(
            str_direction=str_direction,
            middleware_processing_result=middleware_timing_result.typed_result,
            float_elapsed_ms=middleware_timing_result.float_elapsed_ms,
        )
        # Normal return with processed middleware output text.
        return middleware_timing_result.typed_result.str_output_text
