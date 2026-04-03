from __future__ import annotations

import logging
import time

import pytest

from ai_api_unified.middleware.middleware import (
    AUDIT_LOGGER_NAME,
    METRICS_LOGGER_NAME,
    execute_middleware_with_timing,
    log_middleware_observability_events,
)
from ai_api_unified.middleware.middleware_config import (
    CAPABILITIES_KEY,
    DIRECTION_KEY,
    LOG_LEVEL_KEY,
    OBSERVABILITY,
    OBSERVABILITY_CAPABILITIES_DROPPED_VALUES_WARNING_MESSAGE,
    OBSERVABILITY_CAPABILITIES_INVALID_TYPE_WARNING_MESSAGE,
    TOKEN_COUNT_MODE_KEY,
    MiddlewareConfig,
    MiddlewareConfigurationModel,
    MiddlewareEntryModel,
    ObservabilitySettingsModel,
)
from ai_api_unified.middleware.observability import (
    NoOpObservabilityMiddleware,
    get_observability_middleware,
)

TEST_LOGGER_MIDDLEWARE_CONFIG: str = (
    "ai_api_unified.middleware.middleware_config"
)
TEST_DIRECTION_INPUT_OUTPUT_RAW: str = "INPUT_OUTPUT"
TEST_CAPABILITY_COMPLETIONS_RAW: str = "COMPLETIONS"
TEST_CAPABILITY_TTS_RAW: str = "tts"
TEST_INVALID_CAPABILITY_RAW: str = "videos"
TEST_LOG_LEVEL_DEBUG_RAW: str = "debug"
TEST_TOKEN_COUNT_MODE_PROVIDER_ONLY_RAW: str = "PROVIDER_ONLY"
TEST_TIMED_RESULT_VALUE: str = "timed-result"
TEST_CALL_CONTEXT: dict[str, str] = {"provider": "openai"}
TEST_RESULT_SUMMARY: dict[str, int] = {"output_chars": 42}
TEST_ERROR_MESSAGE: str = "boom"


def _set_configuration(
    middleware_config: MiddlewareConfig,
    list_entries: list[MiddlewareEntryModel],
) -> None:
    """
    Replaces the internal configuration with test-controlled middleware entries.

    Args:
        middleware_config: MiddlewareConfig instance under test.
        list_entries: Typed middleware entries to inject for this test case.

    Returns:
        None after replacing the active in-memory middleware configuration.
    """
    middleware_config._configuration = MiddlewareConfigurationModel(
        list_middleware=list_entries
    )
    # Normal return after test configuration injection.
    return None


def test_get_observability_settings_returns_none_when_component_missing() -> None:
    """
    Ensures missing observability middleware config keeps the component disabled.

    Args:
        None

    Returns:
        None for normal test completion.
    """
    middleware_config: MiddlewareConfig = MiddlewareConfig()
    _set_configuration(middleware_config=middleware_config, list_entries=[])

    observability_settings: ObservabilitySettingsModel | None = (
        middleware_config.get_observability_settings()
    )

    assert observability_settings is None


def test_get_observability_settings_applies_defaults_for_enabled_component() -> None:
    """
    Ensures enabled observability middleware can rely on default settings.

    Args:
        None

    Returns:
        None for normal test completion.
    """
    middleware_config: MiddlewareConfig = MiddlewareConfig()
    _set_configuration(
        middleware_config=middleware_config,
        list_entries=[MiddlewareEntryModel(name=OBSERVABILITY, enabled=True)],
    )

    observability_settings: ObservabilitySettingsModel | None = (
        middleware_config.get_observability_settings()
    )

    assert observability_settings is not None
    assert observability_settings.direction == "input_output"
    assert observability_settings.capabilities == []
    assert observability_settings.log_level == "INFO"
    assert observability_settings.token_count_mode == "provider_or_estimate"
    assert observability_settings.include_provider_usage is True
    assert observability_settings.emit_error_events is True


def test_get_observability_settings_normalizes_supported_values() -> None:
    """
    Ensures observability settings normalize direction, capabilities, and modes.

    Args:
        None

    Returns:
        None for normal test completion.
    """
    middleware_config: MiddlewareConfig = MiddlewareConfig()
    _set_configuration(
        middleware_config=middleware_config,
        list_entries=[
            MiddlewareEntryModel(
                name=OBSERVABILITY,
                enabled=True,
                settings={
                    DIRECTION_KEY: TEST_DIRECTION_INPUT_OUTPUT_RAW,
                    CAPABILITIES_KEY: [
                        TEST_CAPABILITY_COMPLETIONS_RAW,
                        TEST_CAPABILITY_TTS_RAW,
                        TEST_CAPABILITY_COMPLETIONS_RAW,
                    ],
                    LOG_LEVEL_KEY: TEST_LOG_LEVEL_DEBUG_RAW,
                    TOKEN_COUNT_MODE_KEY: TEST_TOKEN_COUNT_MODE_PROVIDER_ONLY_RAW,
                },
            )
        ],
    )

    observability_settings: ObservabilitySettingsModel | None = (
        middleware_config.get_observability_settings()
    )

    assert observability_settings is not None
    assert observability_settings.direction == "input_output"
    assert observability_settings.capabilities == ["completions", "tts"]
    assert observability_settings.log_level == "DEBUG"
    assert observability_settings.token_count_mode == "provider_only"


def test_get_observability_settings_warns_and_defaults_for_invalid_capabilities_type(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """
    Ensures unsupported capability value types fall back to the default allow-list.

    Args:
        caplog: Pytest fixture used to capture logger output for assertions.

    Returns:
        None for normal test completion.
    """
    middleware_config: MiddlewareConfig = MiddlewareConfig()
    _set_configuration(
        middleware_config=middleware_config,
        list_entries=[
            MiddlewareEntryModel(
                name=OBSERVABILITY,
                enabled=True,
                settings={CAPABILITIES_KEY: 12345},
            )
        ],
    )
    caplog.set_level(logging.WARNING, logger=TEST_LOGGER_MIDDLEWARE_CONFIG)

    observability_settings: ObservabilitySettingsModel | None = (
        middleware_config.get_observability_settings()
    )

    assert observability_settings is not None
    assert observability_settings.capabilities == []
    assert any(
        log_record.message
        == OBSERVABILITY_CAPABILITIES_INVALID_TYPE_WARNING_MESSAGE
        % (CAPABILITIES_KEY, "int")
        for log_record in caplog.records
    )


def test_get_observability_settings_warns_and_drops_unsupported_capabilities(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """
    Ensures unsupported capability tokens are dropped while supported ones remain.

    Args:
        caplog: Pytest fixture used to capture logger output for assertions.

    Returns:
        None for normal test completion.
    """
    middleware_config: MiddlewareConfig = MiddlewareConfig()
    _set_configuration(
        middleware_config=middleware_config,
        list_entries=[
            MiddlewareEntryModel(
                name=OBSERVABILITY,
                enabled=True,
                settings={
                    CAPABILITIES_KEY: [
                        TEST_CAPABILITY_COMPLETIONS_RAW,
                        TEST_INVALID_CAPABILITY_RAW,
                    ]
                },
            )
        ],
    )
    caplog.set_level(logging.WARNING, logger=TEST_LOGGER_MIDDLEWARE_CONFIG)

    observability_settings: ObservabilitySettingsModel | None = (
        middleware_config.get_observability_settings()
    )

    assert observability_settings is not None
    assert observability_settings.capabilities == ["completions"]
    assert any(
        log_record.message
        == OBSERVABILITY_CAPABILITIES_DROPPED_VALUES_WARNING_MESSAGE
        % (CAPABILITIES_KEY, [TEST_INVALID_CAPABILITY_RAW])
        for log_record in caplog.records
    )


def test_get_observability_settings_sorts_set_capabilities_deterministically() -> None:
    """
    Ensures set-like capability inputs are normalized into a deterministic sorted order.

    Args:
        None

    Returns:
        None for normal test completion.
    """
    middleware_config: MiddlewareConfig = MiddlewareConfig()
    _set_configuration(
        middleware_config=middleware_config,
        list_entries=[
            MiddlewareEntryModel(
                name=OBSERVABILITY,
                enabled=True,
                settings={CAPABILITIES_KEY: {"tts", "completions"}},
            )
        ],
    )

    observability_settings: ObservabilitySettingsModel | None = (
        middleware_config.get_observability_settings()
    )

    assert observability_settings is not None
    assert observability_settings.capabilities == ["completions", "tts"]


def test_get_middleware_settings_returns_typed_observability_model() -> None:
    """
    Ensures the generic middleware-settings path supports observability config.

    Args:
        None

    Returns:
        None for normal test completion.
    """
    middleware_config: MiddlewareConfig = MiddlewareConfig()
    _set_configuration(
        middleware_config=middleware_config,
        list_entries=[MiddlewareEntryModel(name=OBSERVABILITY, enabled=True)],
    )

    middleware_settings = middleware_config.get_middleware_settings(OBSERVABILITY)

    assert middleware_settings is not None
    assert isinstance(middleware_settings, ObservabilitySettingsModel)


def test_execute_middleware_with_timing_returns_result_and_elapsed_ms() -> None:
    """
    Ensures the shared timing helper measures arbitrary middleware work.

    Args:
        None

    Returns:
        None for normal test completion.
    """

    def _timed_callable() -> str:
        """
        Sleeps briefly so the shared timing helper records measurable elapsed time.

        Args:
            None

        Returns:
            Deterministic test value returned through the timing helper.
        """
        time.sleep(0.001)
        # Normal return with the deterministic timing helper payload.
        return TEST_TIMED_RESULT_VALUE

    timing_result = execute_middleware_with_timing(callable_execute=_timed_callable)

    assert timing_result.typed_result == TEST_TIMED_RESULT_VALUE
    assert timing_result.float_elapsed_ms >= 0.0


def test_log_middleware_observability_events_emits_metrics_and_audit_when_requested(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """
    Ensures the shared logging helper emits both metrics and audit records when needed.

    Args:
        caplog: Pytest fixture used to capture logger output for assertions.

    Returns:
        None for normal test completion.
    """
    caplog.set_level(logging.INFO, logger=AUDIT_LOGGER_NAME)
    caplog.set_level(logging.INFO, logger=METRICS_LOGGER_NAME)

    log_middleware_observability_events(
        str_middleware_name=OBSERVABILITY,
        str_direction="input",
        float_elapsed_ms=12.34,
        int_security_action_count=1,
        tuple_str_categories=("policy",),
        bool_security_control_applied=True,
    )

    assert any(
        log_record.name == AUDIT_LOGGER_NAME
        and log_record.message
        == "security_control_applied control=observability direction=input redaction_count=1 categories=('policy',)"
        for log_record in caplog.records
    )
    assert any(
        log_record.name == METRICS_LOGGER_NAME
        and "middleware_execution_timing middleware=observability direction=input"
        in log_record.message
        for log_record in caplog.records
    )


def test_noop_observability_middleware_is_disabled_and_safe() -> None:
    """
    Ensures the Phase A no-op observability middleware safely accepts lifecycle calls.

    Args:
        None

    Returns:
        None for normal test completion.
    """
    middleware: NoOpObservabilityMiddleware = NoOpObservabilityMiddleware()

    middleware.before_call(call_context=TEST_CALL_CONTEXT)
    middleware.after_call(
        call_context=TEST_CALL_CONTEXT,
        call_result_summary=TEST_RESULT_SUMMARY,
    )
    middleware.on_error(
        call_context=TEST_CALL_CONTEXT,
        exception=RuntimeError(TEST_ERROR_MESSAGE),
        float_elapsed_ms=5.0,
    )

    assert middleware.bool_enabled is False


def test_get_observability_middleware_returns_noop_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Ensures the observability middleware factory returns the no-op implementation by default.

    Args:
        monkeypatch: Pytest monkeypatch fixture used to override config lookup.

    Returns:
        None for normal test completion.
    """

    def _get_fake_observability_settings(
        self,
    ) -> ObservabilitySettingsModel | None:
        """
        Returns no observability settings so the disabled path can be exercised.

        Args:
            self: MiddlewareConfig instance created by the middleware factory helper.

        Returns:
            None so the factory uses the no-op implementation.
        """
        # Early return because the test explicitly exercises the disabled path.
        return None

    monkeypatch.setattr(
        "ai_api_unified.middleware.middleware_config.MiddlewareConfig.get_observability_settings",
        _get_fake_observability_settings,
    )

    middleware = get_observability_middleware()

    assert isinstance(middleware, NoOpObservabilityMiddleware)
