# ai_bedrock_base.py
"""
Shared Amazon Bedrock runtime utilities used by Bedrock-backed clients.

Centralizes client construction, retry configuration, and JSON invocation helpers so
that individual provider clients (completions, images, etc.) can focus on domain
specific logic.
"""

from __future__ import annotations

import json
import logging
import random
import time
from typing import Any, Callable, ClassVar, TypeVar

from .ai_base import AIBase
from .util.env_settings import EnvSettings

T = TypeVar("T")


try:
    import boto3  # type: ignore
    from botocore.exceptions import (  # type: ignore
        BotoCoreError,
        ClientError,
        ParamValidationError,
    )

    BEDROCK_RUNTIME_AVAILABLE = True
except ImportError as bedrock_import_error:  # pragma: no cover - optional dependency
    BEDROCK_RUNTIME_AVAILABLE = False

_LOGGER: logging.Logger = logging.getLogger(__name__)

if BEDROCK_RUNTIME_AVAILABLE:

    class AIBedrockBase(AIBase):
        """
        Base class for Amazon Bedrock providers.

        Handles runtime client creation, exponential backoff with jitter, and JSON
        invocation helpers leveraged by higher-level Bedrock services.
        """

        BEDROCK_MAX_RETRIES: ClassVar[int] = 4
        BEDROCK_BACKOFF_SECONDS: ClassVar[list[float]] = [1.0, 2.0, 4.0, 8.0]
        BEDROCK_MAX_JITTER_SECONDS: ClassVar[float] = 0.5
        NON_RETRYABLE_ERROR_CODES: ClassVar[set[str]] = {
            "ValidationException",
            "InvalidRequestException",
            "AccessDeniedException",
            "ResourceNotFoundException",
            "UnsupportedMediaTypeException",
            "ServiceQuotaExceededException",
            "ModelErrorException",
            "ModelStreamErrorException",
        }
        NON_RETRYABLE_ERROR_TYPES: ClassVar[tuple[type[Exception], ...]] = (
            ParamValidationError,
        )

        def __init__(
            self,
            model: str | None = None,
            *,
            bedrock_client: Any | None = None,
            region: str | None = None,
        ) -> None:
            super().__init__(model=model)
            self.env: EnvSettings = EnvSettings()
            self.region: str = region or self.env.get_setting("AWS_REGION", "us-east-1")
            self.backoff_delays: list[float] = list(self.BEDROCK_BACKOFF_SECONDS)
            self.client = (
                bedrock_client
                if bedrock_client is not None
                else self._create_runtime_client()
            )

        def _create_runtime_client(self) -> Any:
            """
            Instantiate the Bedrock runtime client using boto3.
            """
            try:
                return boto3.client(  # type: ignore[union-attr]
                    service_name="bedrock-runtime",
                    region_name=self.region,
                )
            except Exception as client_error:  # pragma: no cover - network side effects
                raise RuntimeError(
                    "Failed to create Bedrock runtime client. Verify AWS credentials and AWS_REGION."
                ) from client_error

        def _compute_backoff_seconds(self, base_delay: float) -> float:
            """
            Return the sleep duration for a retry attempt, adding bounded jitter to avoid thundering herds.
            """
            jitter: float = random.uniform(0.0, self.BEDROCK_MAX_JITTER_SECONDS)
            return base_delay + jitter

        def _sleep_with_backoff(self, base_delay: float) -> None:
            """
            Sleep for the computed backoff interval (base delay plus jitter).
            """
            time.sleep(self._compute_backoff_seconds(base_delay))

        def _execute_with_retries(
            self,
            *,
            operation: Callable[[], T],
            trace_name: str,
        ) -> T:
            """
            Run the provided callable with bounded exponential backoff and jitter. Logs retry attempts
            using structured metadata and re-raises the final exception with useful context.
            """

            last_error: Exception | None = None
            max_attempts: int = len(self.backoff_delays)

            for attempt_index, delay in enumerate(self.backoff_delays, start=1):
                try:
                    return operation()
                except (
                    ClientError,
                    BotoCoreError,
                ) as aws_error:  # pragma: no cover - requires AWS
                    last_error = aws_error
                    error_code: str | None = None
                    if isinstance(aws_error, ClientError):
                        error_code = aws_error.response.get("Error", {}).get("Code")
                        if error_code in self.NON_RETRYABLE_ERROR_CODES:
                            _LOGGER.error(
                                "bedrock_operation_non_retryable",
                                extra={
                                    "operation": trace_name,
                                    "attempt": attempt_index,
                                    "error_type": aws_error.__class__.__name__,
                                    "error_code": error_code,
                                },
                            )
                            raise RuntimeError(
                                f"Bedrock operation '{trace_name}' failed with non-retryable error {error_code}."
                            ) from aws_error

                    if isinstance(aws_error, self.NON_RETRYABLE_ERROR_TYPES):
                        _LOGGER.error(
                            "bedrock_operation_non_retryable",
                            extra={
                                "operation": trace_name,
                                "attempt": attempt_index,
                                "error_type": aws_error.__class__.__name__,
                                "error_code": error_code,
                            },
                        )
                        raise RuntimeError(
                            f"Bedrock operation '{trace_name}' failed with non-retryable error."
                        ) from aws_error

                    if attempt_index == max_attempts:
                        break
                    sleep_seconds: float = self._compute_backoff_seconds(delay)
                    _LOGGER.warning(
                        "bedrock_operation_retry",
                        extra={
                            "operation": trace_name,
                            "attempt": attempt_index,
                            "retry_in_seconds": round(sleep_seconds, 3),
                            "error_type": aws_error.__class__.__name__,
                        },
                    )
                    time.sleep(sleep_seconds)
                except Exception as unexpected_error:
                    last_error = unexpected_error
                    if isinstance(unexpected_error, self.NON_RETRYABLE_ERROR_TYPES):
                        _LOGGER.error(
                            "bedrock_operation_non_retryable",
                            extra={
                                "operation": trace_name,
                                "attempt": attempt_index,
                                "error_type": unexpected_error.__class__.__name__,
                            },
                        )
                        raise RuntimeError(
                            f"Bedrock operation '{trace_name}' failed with non-retryable error."
                        ) from unexpected_error
                    if attempt_index == max_attempts:
                        break
                    sleep_seconds = self._compute_backoff_seconds(delay)
                    _LOGGER.warning(
                        "bedrock_operation_retry_unexpected",
                        extra={
                            "operation": trace_name,
                            "attempt": attempt_index,
                            "retry_in_seconds": round(sleep_seconds, 3),
                            "error_type": unexpected_error.__class__.__name__,
                        },
                    )
                    time.sleep(sleep_seconds)

            assert (
                last_error is not None
            )  # for mypy; loop guarantees assignment when breaking
            raise RuntimeError(
                f"Bedrock operation '{trace_name}' failed after {max_attempts} attempts."
            ) from last_error

        def _invoke_bedrock_json(
            self,
            *,
            model_id: str,
            payload: dict[str, Any],
            accept: str = "application/json",
            content_type: str = "application/json",
            trace_name: str = "invoke_model",
        ) -> dict[str, Any]:
            """
            Invoke the Bedrock runtime with a JSON payload and parse the JSON response.
            """

            request_bytes: bytes = json.dumps(payload).encode("utf-8")

            def _operation() -> dict[str, Any]:
                response: dict[str, Any] = self.client.invoke_model(  # type: ignore[union-attr]
                    modelId=model_id,
                    body=request_bytes,
                    contentType=content_type,
                    accept=accept,
                )
                response_body = response.get("body")
                if hasattr(response_body, "read"):
                    raw_bytes = response_body.read()
                else:
                    raw_bytes = response_body

                if isinstance(raw_bytes, bytes):
                    raw_text = raw_bytes.decode("utf-8")
                elif raw_bytes is None:
                    raw_text = ""
                else:
                    raw_text = str(raw_bytes)

                if not raw_text:
                    return {}

                try:
                    return json.loads(raw_text)
                except json.JSONDecodeError as decode_error:
                    raise ValueError(
                        f"Bedrock response for {model_id} was not valid JSON."
                    ) from decode_error

            return self._execute_with_retries(
                operation=_operation, trace_name=trace_name
            )

else:  # pragma: no cover - boto3 optional dependency

    class AIBedrockBase:  # type: ignore
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError(
                "boto3 is required for Amazon Bedrock features. Install the 'bedrock' optional dependency group."
            )
