from __future__ import (
    annotations,
)  # Keep annotations as strings to avoid import-order issues.

import logging
import secrets
import time
from typing import Any, Callable, ClassVar, TypeVar

from google import genai
from google.api_core import exceptions as gexc  # Provided by google-* libs
from google.auth.exceptions import DefaultCredentialsError
from google.genai import errors as gerr
from google.genai import pagers

from ai_api_unified.util.env_settings import EnvSettings

T = TypeVar("T")

_LOGGER: logging.Logger = logging.getLogger(__name__)


class AIGoogleBase:
    """
    Shared utils for Google clients.

    Semantics preserved:
    - WARNING on retryable attempts prior to sleeping
    - ERROR on non-retryable failures
    - ERROR when retries are exhausted
    - Jitter via `secrets` (no new deps)

    Usage:
        return self._retry_with_exponential_backoff(lambda: client.call(...))
    """

    # Subclasses may override these directly as attributes.
    max_retries: int = 5
    initial_delay: float = 1.0
    backoff_multiplier: float = 2.0
    max_jitter: float = 1.0  # seconds
    retryable_http_status_codes: ClassVar[set[int]] = {408, 429, 500, 502, 503, 504}
    retryable_message_hints: ClassVar[tuple[str, ...]] = (
        "rate limit",
        "quota",
        "temporarily unavailable",
        "deadline exceeded",
        "backend error",
        "connection reset",
        "network",
    )

    GOOGLE_AUTH_METHOD_API_KEY: ClassVar[str] = "api_key"
    GOOGLE_AUTH_METHOD_SERVICE_ACCOUNT: ClassVar[str] = "service_account"

    def _resolve_google_auth_method(
        self,
        env_settings: EnvSettings,
        *,
        use_api_key: bool = False,
    ) -> str:
        """
        Resolves the active Google authentication mode for the current request.

        Args:
            env_settings: Shared environment/settings accessor.
            use_api_key: Explicit override used by legacy call sites.

        Returns:
            Normalized Google auth method token.
        """
        auth_method_value: object = env_settings.get_setting(
            "GOOGLE_AUTH_METHOD",
            self.GOOGLE_AUTH_METHOD_API_KEY,
        )
        auth_method: str = (
            str(auth_method_value).strip().lower()
            if auth_method_value is not None
            else self.GOOGLE_AUTH_METHOD_API_KEY
        )
        if use_api_key:
            return self.GOOGLE_AUTH_METHOD_API_KEY
        if not auth_method:
            return self.GOOGLE_AUTH_METHOD_API_KEY
        return auth_method

    def get_client(
        self,
        model: str,
        *,
        use_api_key: bool = False,
    ) -> genai.Client:
        """
        Initialize the Google Gemini client with either Vertex AI credentials or an API key.

        Parameters
        ----------
        model:
            Model identifier to validate once the client is created.
        use_api_key:
            Backward-compatible explicit override that forces API-key mode.
        """

        env_settings: EnvSettings = EnvSettings()
        auth_method: str = self._resolve_google_auth_method(
            env_settings,
            use_api_key=use_api_key,
        )

        try:
            if auth_method == self.GOOGLE_AUTH_METHOD_API_KEY:
                api_key: str | None = env_settings.get_setting(
                    "GOOGLE_GEMINI_API_KEY", None
                )
                if not api_key:
                    raise RuntimeError(
                        "GOOGLE_GEMINI_API_KEY is not configured but API-key auth was selected."
                    )
                client: genai.Client = genai.Client(api_key=api_key)
            else:
                project_id: str = env_settings.get_setting("GOOGLE_PROJECT_ID", "")
                location: str = env_settings.get_setting(
                    "GOOGLE_LOCATION", "us-central1"
                )
                client = genai.Client(
                    vertexai=True,
                    project=project_id,
                    location=location,
                )

            try:
                model_resource: genai.types.Model | None = client.models.get(
                    model=model
                )
                if model_resource is None:
                    _LOGGER.info(
                        "Successfully initialized Google Gemini model: %s",
                        model,
                    )
                else:
                    _LOGGER.info(
                        "Successfully initialized Google Gemini model: %s",
                        model_resource.name,
                    )
                return client
            except TypeError as type_error:
                _LOGGER.error(
                    "Invalid model type for Google Gemini operations: %s",
                    type_error,
                )
                raise RuntimeError(
                    f"Invalid model type for Google Gemini operations: {type_error}"
                ) from type_error
            except Exception as model_error:
                _LOGGER.warning(
                    "Failed to initialize model %s: %s",
                    model,
                    model_error,
                )
                raise

        except DefaultCredentialsError as cred_error:
            raise RuntimeError(
                "Google authentication failed. Please ensure GOOGLE_APPLICATION_CREDENTIALS "
                "environment variable points to a valid service account JSON file, or "
                "set GOOGLE_AUTH_METHOD=api_key with GOOGLE_GEMINI_API_KEY. "
                f"Error: {cred_error}"
            ) from cred_error
        except ValueError as value_error:
            raise RuntimeError(
                "Invalid or missing configuration for Google Gemini client. "
                "Provide GOOGLE_GEMINI_API_KEY when GOOGLE_AUTH_METHOD=api_key, or set "
                "GOOGLE_APPLICATION_CREDENTIALS / GOOGLE_PROJECT_ID / GOOGLE_LOCATION "
                "for service-account / Vertex AI mode."
            ) from value_error
        except Exception as init_error:
            raise RuntimeError(
                f"Failed to initialize Google Gemini client: {init_error}"
            ) from init_error

    def list_models(
        self,
        client: genai.Client,
        *,
        name_filter: str | None = None,
    ) -> list[str]:
        """Return available Gemini model identifiers filtered by substring."""

        list_model_names: list[str] = []
        try:
            # The for loop handles paging automatically
            list_models: pagers.Pager = client.models.list()
            for model_metadata in list_models:
                model_name: str = getattr(model_metadata, "name", "")
                if not model_name:
                    continue
                if name_filter and name_filter not in model_name:
                    continue
                list_model_names.append(model_name)
        except Exception as model_error:
            _LOGGER.warning(
                "Failed to list Google Gemini models: %s",
                model_error,
            )
        return list_model_names

    def _retry_with_exponential_backoff(
        self,
        operation: Callable[[], T],
        *,
        max_retries: int | None = None,
        initial_delay: float | None = None,
        backoff_multiplier: float | None = None,
        max_jitter: float | None = None,
    ) -> T:
        """
        Execute `operation()` with exponential backoff on transient Google/HTTP errors.

        Parameters
        ----------
        operation : Callable[[], T]
            Zero-argument callable performing the API call.
        max_retries, initial_delay, backoff_multiplier, max_jitter :
            Optional per-call overrides; otherwise instance attributes are used.
        logger : logging.Logger | None
            Optional explicit logger; defaults to module-level _LOGGER.

        Returns
        -------
        T : Result from `operation()`.

        Raises
        ------
        Exception : Re-raises last exception on non-retryable or after exhaustion.
        """

        effective_max_retries: int = (
            max_retries if max_retries is not None else self.max_retries
        )
        effective_initial_delay: float = (
            initial_delay if initial_delay is not None else self.initial_delay
        )
        effective_backoff_multiplier: float = (
            backoff_multiplier
            if backoff_multiplier is not None
            else self.backoff_multiplier
        )
        effective_max_jitter: float = (
            max_jitter if max_jitter is not None else self.max_jitter
        )

        def _extract_status_code(error: Exception) -> int | None:
            status_code: int | None = None

            if hasattr(error, "code"):
                error_any: Any = error
                code_value = error_any.code
                if isinstance(code_value, int):
                    status_code = code_value
                elif hasattr(code_value, "value"):
                    value_candidate = code_value.value
                    if isinstance(value_candidate, int):
                        status_code = value_candidate

            if status_code is None and hasattr(error, "status"):
                error_any = error
                status_value = error_any.status
                if isinstance(status_value, int):
                    status_code = status_value
                elif isinstance(status_value, str) and status_value.isdigit():
                    status_code = int(status_value)

            if status_code is None and hasattr(error, "response"):
                error_any = error
                response = error_any.response
                if response is not None and hasattr(response, "status_code"):
                    status_candidate = response.status_code
                    if isinstance(status_candidate, int):
                        status_code = status_candidate
                    elif (
                        isinstance(status_candidate, str) and status_candidate.isdigit()
                    ):
                        status_code = int(status_candidate)

            return status_code

        def _should_retry_message(message_text: str) -> bool:
            lower_text: str = message_text.lower()
            return any(hint in lower_text for hint in self.retryable_message_hints)

        def _retry_later(
            exception: Exception,
            *,
            warning_context: str,
            exhaustion_context: str,
        ) -> None:
            nonlocal attempt_index

            if attempt_index >= effective_max_retries:
                _LOGGER.error(
                    "%s retries exhausted after %d attempts: %s",
                    exhaustion_context,
                    effective_max_retries,
                    exception,
                )
                raise RuntimeError(
                    f"{exhaustion_context} retries exhausted after {effective_max_retries} attempts: {exception}"
                ) from exception

            base_delay: float = effective_initial_delay * (
                effective_backoff_multiplier**attempt_index
            )
            jitter_fraction: float = secrets.randbelow(1_000_000) / 1_000_000.0
            sleep_seconds: float = base_delay + (jitter_fraction * effective_max_jitter)

            _LOGGER.warning(
                "%s on attempt %d/%d: %s. Retrying in %.2fs...",
                warning_context,
                attempt_index + 1,
                effective_max_retries,
                exception,
                sleep_seconds,
            )
            time.sleep(sleep_seconds)
            attempt_index += 1

        attempt_index: int = 0
        while True:
            try:
                return operation()

            # --- Explicit non-retryable Google API classes ---
            except (
                gexc.InvalidArgument,
                gexc.PermissionDenied,
                gexc.Unauthenticated,
            ) as exc:
                status_code: int | None = _extract_status_code(exc)
                _LOGGER.error(
                    "Non-retryable Google API error%s: %s",
                    f" ({status_code})" if status_code is not None else "",
                    exc,
                )
                raise RuntimeError(
                    f"Google API error{f' ({status_code})' if status_code is not None else ''}: {exc}"
                ) from exc

            # --- Canonical retryable Google API classes ---
            except (
                gexc.ResourceExhausted,  # 429 / quota
                gexc.ServiceUnavailable,  # 503
                gexc.DeadlineExceeded,  # 504-ish
                gexc.InternalServerError,  # 500
                gexc.Aborted,  # safe to retry
                gexc.RetryError,  # wrapped retries
            ) as exc:
                _retry_later(
                    exc,
                    warning_context="Retryable Google API error",
                    exhaustion_context="Google API",
                )

            # --- Other Google API errors with status/message available ---
            except gexc.GoogleAPICallError as api_error:
                status_code: int | None = _extract_status_code(api_error)

                if status_code == 403:
                    _LOGGER.error(
                        "Google access forbidden (403): %s. Check IAM permissions and that the relevant API is enabled.",
                        api_error,
                    )
                    raise RuntimeError(
                        "Access forbidden (403) when calling Google API. Verify IAM permissions and API enablement."
                    ) from api_error

                message_text: str = str(api_error)
                if (
                    status_code in self.retryable_http_status_codes
                    or _should_retry_message(message_text)
                ):
                    _retry_later(
                        api_error,
                        warning_context="Transient Google API error",
                        exhaustion_context="Google API",
                    )
                    continue

                _LOGGER.error(
                    "Non-retryable Google API error%s: %s",
                    f" ({status_code})" if status_code is not None else "",
                    api_error,
                )
                raise RuntimeError(
                    f"Google API error{f' ({status_code})' if status_code is not None else ''}: {api_error}"
                ) from api_error

            # --- Gemini python client errors ---
            except gerr.ServerError as gemini_server_error:
                _retry_later(
                    gemini_server_error,
                    warning_context="Google Gemini server error",
                    exhaustion_context="Google Gemini server error",
                )

            except gerr.ClientError as gemini_error:
                status_code: int | None = _extract_status_code(gemini_error)

                if status_code == 403:
                    _LOGGER.error(
                        "Google Gemini access forbidden (403): %s. Possibly due to the Vertex AI API not being enabled or insufficient IAM permissions.",
                        gemini_error,
                    )
                    raise RuntimeError(
                        "Access forbidden (403) when calling Google Gemini. Verify IAM permissions and API enablement."
                    ) from gemini_error

                message_text_gemini: str = str(gemini_error)
                if (
                    status_code in self.retryable_http_status_codes
                    or _should_retry_message(message_text_gemini)
                ):
                    _retry_later(
                        gemini_error,
                        warning_context="Transient Google Gemini error",
                        exhaustion_context="Google Gemini",
                    )
                    continue

                _LOGGER.error(
                    "Non-retryable Google Gemini error%s: %s",
                    f" ({status_code})" if status_code is not None else "",
                    gemini_error,
                )
                raise RuntimeError(
                    f"Google Gemini error{f' ({status_code})' if status_code is not None else ''}: {gemini_error}"
                ) from gemini_error

            except gerr.APIError as gemini_api_error:
                status_code_api: int | None = _extract_status_code(gemini_api_error)
                message_text_api: str = str(gemini_api_error)

                if (
                    status_code_api in self.retryable_http_status_codes
                    or _should_retry_message(message_text_api)
                ):
                    _retry_later(
                        gemini_api_error,
                        warning_context="Transient Google Gemini API error",
                        exhaustion_context="Google Gemini API",
                    )
                    continue

                _LOGGER.error(
                    "Non-retryable Google Gemini API error%s: %s",
                    f" ({status_code_api})" if status_code_api is not None else "",
                    gemini_api_error,
                )
                raise RuntimeError(
                    f"Google Gemini API error{f' ({status_code_api})' if status_code_api is not None else ''}: {gemini_api_error}"
                ) from gemini_api_error

            # --- Anything else: non-retryable ---
            except Exception as unexpected:
                _LOGGER.error("Unexpected non-retryable error: %s", unexpected)
                raise
