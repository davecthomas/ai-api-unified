import logging
from pathlib import Path
from textwrap import dedent
from typing import Any

import pytest

from ai_api_unified.middleware.middleware_config import (
    ADDRESS_DETECTION_ENABLED_KEY,
    ADDRESS_DETECTION_PROVIDER_KEY,
    ALLOWED_ENTITIES_KEY,
    CAPABILITIES_KEY,
    COUNTRY_SCOPE_KEY,
    DIRECTION_KEY,
    DETECTION_PROFILE_KEY,
    EMIT_ERROR_EVENTS_KEY,
    ENTITY_LABEL_MAP_KEY,
    ENV_CONFIGURATION_PATH,
    LOG_LEVEL_KEY,
    LANGUAGE_KEY,
    MIDDLEWARE_EMPTY_FILE_WARNING_MESSAGE,
    MIDDLEWARE_PATH_SET_BUT_MISSING_WARNING_MESSAGE,
    OBSERVABILITY,
    OBSERVABILITY_CAPABILITIES_DROPPED_VALUES_WARNING_MESSAGE,
    PII_REDACTION,
    PII_ALLOWED_ENTITIES_DROPPED_VALUES_WARNING_MESSAGE,
    PII_ALLOWED_ENTITIES_INVALID_TYPE_WARNING_MESSAGE,
    PII_ENTITY_LABEL_MAP_DROPPED_KEYS_WARNING_MESSAGE,
    PII_ENTITY_LABEL_MAP_INVALID_TYPE_WARNING_MESSAGE,
    PROXIMITY_WINDOW_CHARS_KEY,
    PII_RECOGNIZERS_INVALID_VALUE_WARNING_MESSAGE,
    PII_RECOGNIZERS_MISSING_INFO_MESSAGE,
    PII_RECOGNIZER_RULE_INVALID_TYPE_WARNING_MESSAGE,
    PII_SETTINGS_INVALID_VALUE_WARNING_MESSAGE,
    RECOGNIZER_CC_LAST4_KEY,
    RECOGNIZER_CONFIDENCE_THRESHOLD_KEY,
    RECOGNIZER_CONTEXT_TERMS_KEY,
    RECOGNIZER_DOB_KEY,
    RECOGNIZER_ENABLED_KEY,
    RECOGNIZER_SSN_LAST4_KEY,
    REDACTION_RECOGNIZERS_KEY,
    SPAN_CONFLICT_POLICY_KEY,
    STRICT_MODE_KEY,
    TOKEN_COUNT_MODE_KEY,
    MiddlewareConfig,
    MiddlewareConfigurationModel,
    MiddlewareEntryModel,
    ObservabilitySettingsModel,
    PiiRedactionSettingsModel,
    RedactionRecognizersSettingsModel,
)

TEST_YAML_FILE_MIDDLEWARE_CONFIG: str = "middleware.yaml"
TEST_YAML_FILE_MIDDLEWARE_INVALID_ROOT: str = "middleware_invalid_root.yaml"
TEST_YAML_FILE_MIDDLEWARE_NOT_LIST: str = "middleware_not_list.yaml"
TEST_YAML_FILE_MIDDLEWARE_INVALID_SYNTAX: str = "middleware_invalid_yaml.yaml"
TEST_YAML_FILE_MIDDLEWARE_RECOGNIZERS: str = "middleware_recognizers.yaml"
TEST_YAML_FILE_MIDDLEWARE_RECOGNIZERS_INVALID: str = (
    "middleware_recognizers_invalid.yaml"
)
TEST_YAML_FILE_OBSERVABILITY_CONFIG: str = "middleware_observability.yaml"
TEST_YAML_FILE_OBSERVABILITY_INVALID: str = "middleware_observability_invalid.yaml"
TEST_DIRECTION_INPUT_ONLY: str = "input_only"
TEST_DIRECTION_OUTPUT_ONLY: str = "output_only"
TEST_DIRECTION_INPUT_OUTPUT: str = "input_output"
TEST_DIRECTION_INPUT_OUTPUT_RAW: str = "INPUT_OUTPUT"
TEST_DIRECTION_INVALID_RAW: str = "unsupported-direction"
TEST_BOOL_TRUE_RAW: str = "yes"
TEST_DETECTION_PROFILE_HIGH_ACCURACY_RAW: str = "HIGH_ACCURACY"
TEST_DETECTION_PROFILE_HIGH_ACCURACY_NORMALIZED: str = "high_accuracy"
TEST_DETECTION_PROFILE_BALANCED_NORMALIZED: str = "balanced"
TEST_DETECTION_PROFILE_INVALID_RAW: str = "ultra-accuracy"
TEST_LANGUAGE_RAW: str = "EN"
TEST_LANGUAGE_NORMALIZED: str = "en"
TEST_COUNTRY_SCOPE_RAW: str = "us"
TEST_COUNTRY_SCOPE_NORMALIZED: str = "US"
TEST_ADDRESS_DETECTION_ENABLED_RAW: str = "1"
TEST_ADDRESS_DETECTION_PROVIDER_RAW: str = "USADDRESS"
TEST_ADDRESS_DETECTION_PROVIDER_NORMALIZED: str = "usaddress"
TEST_SPAN_CONFLICT_POLICY_RAW: str = "PREFER_USADDRESS_LONGEST"
TEST_SPAN_CONFLICT_POLICY_NORMALIZED: str = "prefer_usaddress_longest"
TEST_ALLOWED_ENTITY_NAME_RAW: str = "name"
TEST_ALLOWED_ENTITY_NAME_NORMALIZED: str = "NAME"
TEST_CUSTOM_LABEL_KEY_NAME_RAW: str = "name"
TEST_CUSTOM_LABEL_KEY_INVALID_PERSON_RAW: str = "person"
TEST_CUSTOM_LABEL_VALUE_FULL_NAME: str = "FullName"
TEST_RECOGNIZER_PROXIMITY_WINDOW_INJECTED: int = 42
TEST_RECOGNIZER_PROXIMITY_WINDOW_YAML_RAW: str = "44"
TEST_RECOGNIZER_PROXIMITY_WINDOW_YAML_EXPECTED: int = 44
TEST_RECOGNIZER_ENABLED_FALSE_RAW: str = "false"
TEST_RECOGNIZER_CONFIDENCE_THRESHOLD_RAW: str = "0.66"
TEST_RECOGNIZER_CONFIDENCE_THRESHOLD_EXPECTED: float = 0.66
TEST_RECOGNIZER_SSN_CONTEXT_TERM_TOKEN: str = "ssn"
TEST_RECOGNIZER_SSN_NEGATIVE_CONTEXT_TERM_TOKEN: str = "credit card"
TEST_RECOGNIZER_CC_CONTEXT_TERM_TOKEN: str = "cc"
TEST_RECOGNIZER_CC_BRAND_CONTEXT_TERM_TOKEN: str = "visa"
TEST_RECOGNIZER_CONTEXT_TERM_INJECTED: str = "ssn suffix"
TEST_RECOGNIZER_CONTEXT_TERM_YAML: str = "ssn ending"
TEST_RECOGNIZER_CONTEXT_TERM_TRIMMED_RAW: str = "  ssn suffix  "
TEST_RECOGNIZER_CONTEXT_TERM_NORMALIZED: str = "ssn suffix"
TEST_RECOGNIZER_CONTEXT_TERM_NONSTRING_RAW: int = 123
TEST_RECOGNIZER_CONTEXT_TERM_NONE_RAW: str | None = None
LIST_MIXED_RECOGNIZER_CONTEXT_TERMS_RAW: list[str | int | None] = [
    TEST_RECOGNIZER_CONTEXT_TERM_TRIMMED_RAW,
    TEST_RECOGNIZER_CONTEXT_TERM_NONSTRING_RAW,
    TEST_RECOGNIZER_CONTEXT_TERM_NONE_RAW,
]
TEST_RECOGNIZER_INVALID_WINDOW_RAW: str = "invalid-window"
TEST_RECOGNIZER_INVALID_CC_RULE_RAW: str = "invalid-cc-config"
TEST_RECOGNIZER_INVALID_CONFIDENCE_THRESHOLD_RAW: str = "invalid-threshold"
TEST_RECOGNIZER_INVALID_CONTEXT_TERMS_RAW: int = 12345
DICT_RECOGNIZER_INVALID_ENABLED_RAW: dict[str, bool] = {"invalid": True}
TEST_ALLOWED_ENTITIES_INVALID_TYPE_RAW: int = 12345
TEST_ALLOWED_ENTITY_INVALID_TOKEN_RAW: str = "person"
TEST_ENTITY_LABEL_MAP_INVALID_TYPE_RAW: str = "invalid-label-map"
TEST_ENTITY_LABEL_MAP_DROPPED_KEY_RAW: str = "person"
TEST_YAML_FILE_MIDDLEWARE_EMPTY: str = "middleware_empty.yaml"
TEST_OBSERVABILITY_CAPABILITY_IMAGES_RAW: str = "IMAGES"
TEST_OBSERVABILITY_CAPABILITY_TTS_RAW: str = "tts"
TEST_OBSERVABILITY_INVALID_CAPABILITY_RAW: str = "video"
TEST_OBSERVABILITY_LOG_LEVEL_ERROR_RAW: str = "error"
TEST_OBSERVABILITY_TOKEN_COUNT_MODE_NONE_RAW: str = "NONE"
TEST_LOGGER_MIDDLEWARE_CONFIG: str = (
    "ai_api_unified.middleware.middleware_config"
)


def _set_configuration(
    middleware_config: MiddlewareConfig, list_entries: list[MiddlewareEntryModel]
) -> None:
    """
    Replaces the internal configuration with test-controlled middleware entries.

    Args:
        middleware_config: MiddlewareConfig instance under test.
        list_entries: Typed middleware entries to inject for this test case.

    Returns:
        None
    """
    middleware_config._configuration = MiddlewareConfigurationModel(
        list_middleware=list_entries
    )


def test_get_middleware_settings_returns_none_for_empty_settings() -> None:
    """
    Ensures an enabled middleware component with empty settings is treated as disabled.

    Args:
        None

    Returns:
        None
    """
    middleware_config: MiddlewareConfig = MiddlewareConfig()
    _set_configuration(
        middleware_config,
        [
            MiddlewareEntryModel(
                name=PII_REDACTION,
                enabled=True,
                settings={},
            )
        ],
    )

    middleware_settings: PiiRedactionSettingsModel | None = (
        middleware_config.get_middleware_settings(PII_REDACTION)
    )
    assert middleware_settings is None


def test_get_middleware_settings_returns_none_for_missing_settings() -> None:
    """
    Ensures an enabled middleware component without settings is treated as disabled.

    Args:
        None

    Returns:
        None
    """
    middleware_config: MiddlewareConfig = MiddlewareConfig()
    _set_configuration(
        middleware_config,
        [
            MiddlewareEntryModel(
                name=PII_REDACTION,
                enabled=True,
            )
        ],
    )

    middleware_settings: PiiRedactionSettingsModel | None = (
        middleware_config.get_middleware_settings(PII_REDACTION)
    )
    assert middleware_settings is None


def test_get_middleware_settings_returns_none_for_invalid_settings_type() -> None:
    """
    Ensures an enabled middleware component with non-dictionary settings is treated as disabled.

    Args:
        None

    Returns:
        None
    """
    middleware_config: MiddlewareConfig = MiddlewareConfig()
    _set_configuration(
        middleware_config,
        [
            MiddlewareEntryModel(
                name=PII_REDACTION,
                enabled=True,
                settings=["invalid"],
            )
        ],
    )

    middleware_settings: PiiRedactionSettingsModel | None = (
        middleware_config.get_middleware_settings(PII_REDACTION)
    )
    assert middleware_settings is None


def test_get_middleware_settings_returns_typed_model_for_valid_enabled_component() -> (
    None
):
    """
    Ensures a valid enabled middleware component returns a typed settings model.

    Args:
        None

    Returns:
        None
    """
    middleware_config: MiddlewareConfig = MiddlewareConfig()
    _set_configuration(
        middleware_config,
        [
            MiddlewareEntryModel(
                name=PII_REDACTION,
                enabled=True,
                settings={DIRECTION_KEY: TEST_DIRECTION_INPUT_ONLY},
            )
        ],
    )

    middleware_settings: PiiRedactionSettingsModel | None = (
        middleware_config.get_middleware_settings(PII_REDACTION)
    )
    assert middleware_settings is not None
    assert middleware_settings.direction == TEST_DIRECTION_INPUT_ONLY
    assert middleware_settings.strict_mode is False


def test_get_middleware_settings_coerces_settings_into_normalized_types() -> None:
    """
    Ensures Pydantic normalization applies coercion and defaults for the PII settings model.

    Args:
        None

    Returns:
        None
    """
    middleware_config: MiddlewareConfig = MiddlewareConfig()
    _set_configuration(
        middleware_config,
        [
            MiddlewareEntryModel(
                name=PII_REDACTION,
                enabled="true",
                settings={
                    DIRECTION_KEY: TEST_DIRECTION_INPUT_OUTPUT_RAW,
                    STRICT_MODE_KEY: TEST_BOOL_TRUE_RAW,
                    DETECTION_PROFILE_KEY: TEST_DETECTION_PROFILE_HIGH_ACCURACY_RAW,
                    LANGUAGE_KEY: TEST_LANGUAGE_RAW,
                    COUNTRY_SCOPE_KEY: TEST_COUNTRY_SCOPE_RAW,
                    ADDRESS_DETECTION_ENABLED_KEY: TEST_ADDRESS_DETECTION_ENABLED_RAW,
                    ADDRESS_DETECTION_PROVIDER_KEY: TEST_ADDRESS_DETECTION_PROVIDER_RAW,
                    SPAN_CONFLICT_POLICY_KEY: TEST_SPAN_CONFLICT_POLICY_RAW,
                    ALLOWED_ENTITIES_KEY: TEST_ALLOWED_ENTITY_NAME_RAW,
                    ENTITY_LABEL_MAP_KEY: {
                        TEST_CUSTOM_LABEL_KEY_NAME_RAW: TEST_CUSTOM_LABEL_VALUE_FULL_NAME
                    },
                },
            )
        ],
    )

    middleware_settings: PiiRedactionSettingsModel | None = (
        middleware_config.get_middleware_settings(PII_REDACTION)
    )
    assert middleware_settings is not None
    assert middleware_settings.direction == TEST_DIRECTION_INPUT_OUTPUT
    assert middleware_settings.strict_mode is True
    assert (
        middleware_settings.detection_profile
        == TEST_DETECTION_PROFILE_HIGH_ACCURACY_NORMALIZED
    )
    assert middleware_settings.language == TEST_LANGUAGE_NORMALIZED
    assert middleware_settings.country_scope == TEST_COUNTRY_SCOPE_NORMALIZED
    assert middleware_settings.address_detection_enabled is True
    assert (
        middleware_settings.address_detection_provider
        == TEST_ADDRESS_DETECTION_PROVIDER_NORMALIZED
    )
    assert (
        middleware_settings.span_conflict_policy == TEST_SPAN_CONFLICT_POLICY_NORMALIZED
    )
    assert middleware_settings.allowed_entities == [TEST_ALLOWED_ENTITY_NAME_NORMALIZED]
    assert middleware_settings.entity_label_map == {
        TEST_ALLOWED_ENTITY_NAME_NORMALIZED: TEST_CUSTOM_LABEL_VALUE_FULL_NAME
    }


def test_get_middleware_settings_warns_on_invalid_direction_and_detection_profile(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """
    Ensures invalid direction and detection_profile values emit warnings and fall back.

    Args:
        caplog: Pytest fixture used to capture logger output for assertions.

    Returns:
        None
    """
    caplog.set_level(logging.WARNING, logger=TEST_LOGGER_MIDDLEWARE_CONFIG)
    middleware_config: MiddlewareConfig = MiddlewareConfig()
    _set_configuration(
        middleware_config,
        [
            MiddlewareEntryModel(
                name=PII_REDACTION,
                enabled=True,
                settings={
                    DIRECTION_KEY: TEST_DIRECTION_INVALID_RAW,
                    DETECTION_PROFILE_KEY: TEST_DETECTION_PROFILE_INVALID_RAW,
                },
            )
        ],
    )

    middleware_settings: PiiRedactionSettingsModel | None = (
        middleware_config.get_middleware_settings(PII_REDACTION)
    )
    assert middleware_settings is not None
    assert middleware_settings.direction == TEST_DIRECTION_INPUT_ONLY
    assert (
        middleware_settings.detection_profile
        == TEST_DETECTION_PROFILE_BALANCED_NORMALIZED
    )
    list_str_logged_messages: list[str] = []
    # Loop through captured log records and collect fully rendered log message text.
    for log_record in caplog.records:
        list_str_logged_messages.append(log_record.getMessage())
    assert any(
        PII_SETTINGS_INVALID_VALUE_WARNING_MESSAGE
        % (DIRECTION_KEY, TEST_DIRECTION_INVALID_RAW)
        in str_logged_message
        for str_logged_message in list_str_logged_messages
    )
    assert any(
        PII_SETTINGS_INVALID_VALUE_WARNING_MESSAGE
        % (DETECTION_PROFILE_KEY, TEST_DETECTION_PROFILE_INVALID_RAW)
        in str_logged_message
        for str_logged_message in list_str_logged_messages
    )


def test_get_middleware_settings_warns_on_invalid_allowed_entities_type(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """
    Ensures invalid allowed_entities type emits warning and falls back to empty list.

    Args:
        caplog: Pytest fixture used to capture logger output for assertions.

    Returns:
        None
    """
    caplog.set_level(logging.WARNING, logger=TEST_LOGGER_MIDDLEWARE_CONFIG)
    middleware_config: MiddlewareConfig = MiddlewareConfig()
    _set_configuration(
        middleware_config,
        [
            MiddlewareEntryModel(
                name=PII_REDACTION,
                enabled=True,
                settings={ALLOWED_ENTITIES_KEY: TEST_ALLOWED_ENTITIES_INVALID_TYPE_RAW},
            )
        ],
    )

    middleware_settings: PiiRedactionSettingsModel | None = (
        middleware_config.get_middleware_settings(PII_REDACTION)
    )
    assert middleware_settings is not None
    assert middleware_settings.allowed_entities == []
    list_str_logged_messages: list[str] = []
    # Loop through captured log records and collect fully rendered log message text.
    for log_record in caplog.records:
        list_str_logged_messages.append(log_record.getMessage())
    assert any(
        PII_ALLOWED_ENTITIES_INVALID_TYPE_WARNING_MESSAGE
        % (ALLOWED_ENTITIES_KEY, int.__name__)
        in str_logged_message
        for str_logged_message in list_str_logged_messages
    )


def test_get_middleware_settings_warns_on_dropped_allowed_entities_values(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """
    Ensures unsupported allowed_entities values are dropped with warning logs.

    Args:
        caplog: Pytest fixture used to capture logger output for assertions.

    Returns:
        None
    """
    caplog.set_level(logging.WARNING, logger=TEST_LOGGER_MIDDLEWARE_CONFIG)
    middleware_config: MiddlewareConfig = MiddlewareConfig()
    _set_configuration(
        middleware_config,
        [
            MiddlewareEntryModel(
                name=PII_REDACTION,
                enabled=True,
                settings={
                    ALLOWED_ENTITIES_KEY: [
                        TEST_ALLOWED_ENTITY_NAME_RAW,
                        TEST_ALLOWED_ENTITY_INVALID_TOKEN_RAW,
                    ]
                },
            )
        ],
    )

    middleware_settings: PiiRedactionSettingsModel | None = (
        middleware_config.get_middleware_settings(PII_REDACTION)
    )
    assert middleware_settings is not None
    assert middleware_settings.allowed_entities == [TEST_ALLOWED_ENTITY_NAME_NORMALIZED]
    list_str_logged_messages: list[str] = []
    # Loop through captured log records and collect fully rendered log message text.
    for log_record in caplog.records:
        list_str_logged_messages.append(log_record.getMessage())
    assert any(
        PII_ALLOWED_ENTITIES_DROPPED_VALUES_WARNING_MESSAGE
        % (ALLOWED_ENTITIES_KEY, [TEST_ALLOWED_ENTITY_INVALID_TOKEN_RAW])
        in str_logged_message
        for str_logged_message in list_str_logged_messages
    )


def test_get_middleware_settings_warns_on_invalid_entity_label_map_type(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """
    Ensures invalid entity_label_map type emits warning and falls back to defaults.

    Args:
        caplog: Pytest fixture used to capture logger output for assertions.

    Returns:
        None
    """
    caplog.set_level(logging.WARNING, logger=TEST_LOGGER_MIDDLEWARE_CONFIG)
    middleware_config: MiddlewareConfig = MiddlewareConfig()
    _set_configuration(
        middleware_config,
        [
            MiddlewareEntryModel(
                name=PII_REDACTION,
                enabled=True,
                settings={ENTITY_LABEL_MAP_KEY: TEST_ENTITY_LABEL_MAP_INVALID_TYPE_RAW},
            )
        ],
    )

    middleware_settings: PiiRedactionSettingsModel | None = (
        middleware_config.get_middleware_settings(PII_REDACTION)
    )
    assert middleware_settings is not None
    assert middleware_settings.entity_label_map == {
        "NAME": "NAME",
        "PHONE": "PHONE",
        "EMAIL": "EMAIL",
        "SSN": "SSN",
        "ADDRESS": "ADDRESS",
        "DOB": "DOB",
        "CC_LAST4": "CC_LAST4",
    }
    list_str_logged_messages: list[str] = []
    # Loop through captured log records and collect fully rendered log message text.
    for log_record in caplog.records:
        list_str_logged_messages.append(log_record.getMessage())
    assert any(
        PII_ENTITY_LABEL_MAP_INVALID_TYPE_WARNING_MESSAGE
        % (ENTITY_LABEL_MAP_KEY, str.__name__)
        in str_logged_message
        for str_logged_message in list_str_logged_messages
    )


def test_get_middleware_settings_warns_on_dropped_entity_label_map_keys(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """
    Ensures unsupported entity_label_map keys are dropped with warning logs.

    Args:
        caplog: Pytest fixture used to capture logger output for assertions.

    Returns:
        None
    """
    caplog.set_level(logging.WARNING, logger=TEST_LOGGER_MIDDLEWARE_CONFIG)
    middleware_config: MiddlewareConfig = MiddlewareConfig()
    _set_configuration(
        middleware_config,
        [
            MiddlewareEntryModel(
                name=PII_REDACTION,
                enabled=True,
                settings={
                    ENTITY_LABEL_MAP_KEY: {
                        TEST_CUSTOM_LABEL_KEY_NAME_RAW: TEST_CUSTOM_LABEL_VALUE_FULL_NAME,
                        TEST_ENTITY_LABEL_MAP_DROPPED_KEY_RAW: TEST_CUSTOM_LABEL_VALUE_FULL_NAME,
                    }
                },
            )
        ],
    )

    middleware_settings: PiiRedactionSettingsModel | None = (
        middleware_config.get_middleware_settings(PII_REDACTION)
    )
    assert middleware_settings is not None
    assert middleware_settings.entity_label_map == {
        TEST_ALLOWED_ENTITY_NAME_NORMALIZED: TEST_CUSTOM_LABEL_VALUE_FULL_NAME
    }
    list_str_logged_messages: list[str] = []
    # Loop through captured log records and collect fully rendered log message text.
    for log_record in caplog.records:
        list_str_logged_messages.append(log_record.getMessage())
    assert any(
        PII_ENTITY_LABEL_MAP_DROPPED_KEYS_WARNING_MESSAGE
        % (ENTITY_LABEL_MAP_KEY, [TEST_ENTITY_LABEL_MAP_DROPPED_KEY_RAW])
        in str_logged_message
        for str_logged_message in list_str_logged_messages
    )


def test_get_middleware_settings_applies_defaults_for_missing_optional_fields() -> None:
    """
    Ensures omitted optional fields are populated with defaults by the typed settings model.

    Args:
        None

    Returns:
        None
    """
    middleware_config: MiddlewareConfig = MiddlewareConfig()
    _set_configuration(
        middleware_config,
        [
            MiddlewareEntryModel(
                name=PII_REDACTION,
                enabled=True,
                settings={DIRECTION_KEY: TEST_DIRECTION_OUTPUT_ONLY},
            )
        ],
    )

    middleware_settings: PiiRedactionSettingsModel | None = (
        middleware_config.get_middleware_settings(PII_REDACTION)
    )
    assert middleware_settings is not None
    assert middleware_settings.direction == TEST_DIRECTION_OUTPUT_ONLY
    assert middleware_settings.strict_mode is False
    assert (
        middleware_settings.detection_profile
        == TEST_DETECTION_PROFILE_BALANCED_NORMALIZED
    )
    assert middleware_settings.language == TEST_LANGUAGE_NORMALIZED
    assert middleware_settings.country_scope == TEST_COUNTRY_SCOPE_NORMALIZED
    assert middleware_settings.address_detection_enabled is True
    assert (
        middleware_settings.address_detection_provider
        == TEST_ADDRESS_DETECTION_PROVIDER_NORMALIZED
    )
    assert (
        middleware_settings.span_conflict_policy == TEST_SPAN_CONFLICT_POLICY_NORMALIZED
    )
    assert middleware_settings.allowed_entities == []


def test_get_middleware_settings_ignores_noncanonical_category_tokens() -> None:
    """
    Ensures non-canonical category aliases are ignored in allow-list and label-map normalization.

    Args:
        None

    Returns:
        None
    """
    middleware_config: MiddlewareConfig = MiddlewareConfig()
    _set_configuration(
        middleware_config,
        [
            MiddlewareEntryModel(
                name=PII_REDACTION,
                enabled=True,
                settings={
                    ALLOWED_ENTITIES_KEY: [
                        TEST_ALLOWED_ENTITY_NAME_RAW,
                        TEST_CUSTOM_LABEL_KEY_INVALID_PERSON_RAW,
                    ],
                    ENTITY_LABEL_MAP_KEY: {
                        TEST_CUSTOM_LABEL_KEY_NAME_RAW: TEST_CUSTOM_LABEL_VALUE_FULL_NAME,
                        TEST_CUSTOM_LABEL_KEY_INVALID_PERSON_RAW: TEST_CUSTOM_LABEL_VALUE_FULL_NAME,
                    },
                },
            )
        ],
    )

    middleware_settings: PiiRedactionSettingsModel | None = (
        middleware_config.get_middleware_settings(PII_REDACTION)
    )
    assert middleware_settings is not None
    assert middleware_settings.allowed_entities == [TEST_ALLOWED_ENTITY_NAME_NORMALIZED]
    assert middleware_settings.entity_label_map == {
        TEST_ALLOWED_ENTITY_NAME_NORMALIZED: TEST_CUSTOM_LABEL_VALUE_FULL_NAME
    }


def test_get_middleware_settings_applies_default_redaction_recognizers_when_missing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """
    Ensures redaction recognizer extension settings default correctly when omitted.

    Args:
        None

    Returns:
        None
    """
    caplog.set_level(logging.INFO, logger=TEST_LOGGER_MIDDLEWARE_CONFIG)
    middleware_config: MiddlewareConfig = MiddlewareConfig()
    _set_configuration(
        middleware_config,
        [
            MiddlewareEntryModel(
                name=PII_REDACTION,
                enabled=True,
                settings={DIRECTION_KEY: TEST_DIRECTION_INPUT_ONLY},
            )
        ],
    )

    middleware_settings: PiiRedactionSettingsModel | None = (
        middleware_config.get_middleware_settings(PII_REDACTION)
    )
    default_redaction_recognizers_settings: RedactionRecognizersSettingsModel = (
        RedactionRecognizersSettingsModel()
    )
    assert middleware_settings is not None
    assert (
        middleware_settings.redaction_recognizers.model_dump()
        == default_redaction_recognizers_settings.model_dump()
    )
    assert (
        TEST_RECOGNIZER_SSN_CONTEXT_TERM_TOKEN
        in middleware_settings.redaction_recognizers.ssn_last4.context_terms
    )
    assert (
        TEST_RECOGNIZER_SSN_NEGATIVE_CONTEXT_TERM_TOKEN
        in middleware_settings.redaction_recognizers.ssn_last4.negative_context_terms
    )
    assert (
        TEST_RECOGNIZER_CC_CONTEXT_TERM_TOKEN
        in middleware_settings.redaction_recognizers.cc_last4.context_terms
    )
    assert (
        TEST_RECOGNIZER_CC_BRAND_CONTEXT_TERM_TOKEN
        in middleware_settings.redaction_recognizers.cc_last4.context_terms
    )
    list_str_logged_messages: list[str] = []
    # Loop through captured log records and collect fully rendered log message text.
    for log_record in caplog.records:
        list_str_logged_messages.append(log_record.getMessage())
    assert any(
        PII_RECOGNIZERS_MISSING_INFO_MESSAGE % REDACTION_RECOGNIZERS_KEY
        in str_logged_message
        for str_logged_message in list_str_logged_messages
    )


def test_get_middleware_settings_normalizes_partial_redaction_recognizers_config() -> (
    None
):
    """
    Ensures partial recognizer extension config merges with defaults and normalizes values.

    Args:
        None

    Returns:
        None
    """
    middleware_config: MiddlewareConfig = MiddlewareConfig()
    _set_configuration(
        middleware_config,
        [
            MiddlewareEntryModel(
                name=PII_REDACTION,
                enabled=True,
                settings={
                    REDACTION_RECOGNIZERS_KEY: {
                        PROXIMITY_WINDOW_CHARS_KEY: TEST_RECOGNIZER_PROXIMITY_WINDOW_INJECTED,
                        RECOGNIZER_SSN_LAST4_KEY: {
                            RECOGNIZER_ENABLED_KEY: TEST_RECOGNIZER_ENABLED_FALSE_RAW,
                            RECOGNIZER_CONFIDENCE_THRESHOLD_KEY: TEST_RECOGNIZER_CONFIDENCE_THRESHOLD_RAW,
                            RECOGNIZER_CONTEXT_TERMS_KEY: TEST_RECOGNIZER_CONTEXT_TERM_INJECTED,
                        },
                    }
                },
            )
        ],
    )

    middleware_settings: PiiRedactionSettingsModel | None = (
        middleware_config.get_middleware_settings(PII_REDACTION)
    )
    default_redaction_recognizers_settings: RedactionRecognizersSettingsModel = (
        RedactionRecognizersSettingsModel()
    )
    assert middleware_settings is not None
    assert (
        middleware_settings.redaction_recognizers.proximity_window_chars
        == TEST_RECOGNIZER_PROXIMITY_WINDOW_INJECTED
    )
    assert middleware_settings.redaction_recognizers.ssn_last4.enabled is False
    assert (
        middleware_settings.redaction_recognizers.ssn_last4.confidence_threshold
        == TEST_RECOGNIZER_CONFIDENCE_THRESHOLD_EXPECTED
    )
    assert middleware_settings.redaction_recognizers.ssn_last4.context_terms == [
        TEST_RECOGNIZER_CONTEXT_TERM_INJECTED
    ]
    assert (
        middleware_settings.redaction_recognizers.cc_last4.model_dump()
        == default_redaction_recognizers_settings.cc_last4.model_dump()
    )
    assert (
        TEST_RECOGNIZER_SSN_NEGATIVE_CONTEXT_TERM_TOKEN
        in middleware_settings.redaction_recognizers.ssn_last4.negative_context_terms
    )
    assert (
        TEST_RECOGNIZER_CC_CONTEXT_TERM_TOKEN
        in middleware_settings.redaction_recognizers.cc_last4.context_terms
    )
    assert (
        TEST_RECOGNIZER_CC_BRAND_CONTEXT_TERM_TOKEN
        in middleware_settings.redaction_recognizers.cc_last4.context_terms
    )
    assert (
        middleware_settings.redaction_recognizers.dob.model_dump()
        == default_redaction_recognizers_settings.dob.model_dump()
    )


def test_get_middleware_settings_filters_nonstring_context_terms_in_recognizer_lists() -> (
    None
):
    """
    Ensures context-term list normalization keeps only non-empty string values.

    Args:
        None

    Returns:
        None
    """
    middleware_config: MiddlewareConfig = MiddlewareConfig()
    _set_configuration(
        middleware_config,
        [
            MiddlewareEntryModel(
                name=PII_REDACTION,
                enabled=True,
                settings={
                    REDACTION_RECOGNIZERS_KEY: {
                        RECOGNIZER_SSN_LAST4_KEY: {
                            RECOGNIZER_CONTEXT_TERMS_KEY: LIST_MIXED_RECOGNIZER_CONTEXT_TERMS_RAW
                        }
                    }
                },
            )
        ],
    )

    middleware_settings: PiiRedactionSettingsModel | None = (
        middleware_config.get_middleware_settings(PII_REDACTION)
    )
    assert middleware_settings is not None
    assert middleware_settings.redaction_recognizers.ssn_last4.context_terms == [
        TEST_RECOGNIZER_CONTEXT_TERM_NORMALIZED
    ]


def test_get_middleware_settings_falls_back_for_invalid_redaction_recognizers_config(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """
    Ensures invalid recognizer extension values safely fall back to typed defaults.

    Args:
        None

    Returns:
        None
    """
    caplog.set_level(logging.WARNING, logger=TEST_LOGGER_MIDDLEWARE_CONFIG)
    middleware_config: MiddlewareConfig = MiddlewareConfig()
    _set_configuration(
        middleware_config,
        [
            MiddlewareEntryModel(
                name=PII_REDACTION,
                enabled=True,
                settings={
                    REDACTION_RECOGNIZERS_KEY: {
                        PROXIMITY_WINDOW_CHARS_KEY: TEST_RECOGNIZER_INVALID_WINDOW_RAW,
                        RECOGNIZER_SSN_LAST4_KEY: {
                            RECOGNIZER_ENABLED_KEY: DICT_RECOGNIZER_INVALID_ENABLED_RAW
                        },
                        RECOGNIZER_CC_LAST4_KEY: TEST_RECOGNIZER_INVALID_CC_RULE_RAW,
                        RECOGNIZER_DOB_KEY: {
                            RECOGNIZER_CONFIDENCE_THRESHOLD_KEY: TEST_RECOGNIZER_INVALID_CONFIDENCE_THRESHOLD_RAW,
                            RECOGNIZER_CONTEXT_TERMS_KEY: TEST_RECOGNIZER_INVALID_CONTEXT_TERMS_RAW,
                        },
                    }
                },
            )
        ],
    )

    middleware_settings: PiiRedactionSettingsModel | None = (
        middleware_config.get_middleware_settings(PII_REDACTION)
    )
    default_redaction_recognizers_settings: RedactionRecognizersSettingsModel = (
        RedactionRecognizersSettingsModel()
    )
    assert middleware_settings is not None
    assert (
        middleware_settings.redaction_recognizers.model_dump()
        == default_redaction_recognizers_settings.model_dump()
    )
    list_str_logged_messages: list[str] = []
    # Loop through captured log records and collect fully rendered log message text.
    for log_record in caplog.records:
        list_str_logged_messages.append(log_record.getMessage())
    assert any(
        f"`{REDACTION_RECOGNIZERS_KEY}.{RECOGNIZER_SSN_LAST4_KEY}` "
        in str_logged_message
        and "failed validation; using defaults." in str_logged_message
        for str_logged_message in list_str_logged_messages
    )
    assert any(
        PII_RECOGNIZERS_INVALID_VALUE_WARNING_MESSAGE
        % (
            REDACTION_RECOGNIZERS_KEY,
            PROXIMITY_WINDOW_CHARS_KEY,
            TEST_RECOGNIZER_INVALID_WINDOW_RAW,
        )
        in str_logged_message
        for str_logged_message in list_str_logged_messages
    )
    assert any(
        PII_RECOGNIZER_RULE_INVALID_TYPE_WARNING_MESSAGE
        % (
            REDACTION_RECOGNIZERS_KEY,
            RECOGNIZER_CC_LAST4_KEY,
            str.__name__,
        )
        in str_logged_message
        for str_logged_message in list_str_logged_messages
    )


def test_load_configuration_reads_yaml_file_and_normalizes_settings(
    monkeypatch: Any, tmp_path: Path
) -> None:
    """
    Ensures YAML file loading path works and returns normalized typed PII settings.

    Args:
        monkeypatch: Pytest fixture used to control environment variables for this test process.
        tmp_path: Temporary filesystem path fixture used to write a test YAML file.

    Returns:
        None
    """
    str_yaml_payload: str = dedent(
        f"""
        middleware:
          - name: {PII_REDACTION}
            enabled: true
            settings:
              direction: {TEST_DIRECTION_INPUT_OUTPUT_RAW}
              strict_mode: "{TEST_BOOL_TRUE_RAW}"
              detection_profile: {TEST_DETECTION_PROFILE_HIGH_ACCURACY_RAW}
              language: {TEST_LANGUAGE_RAW}
              country_scope: {TEST_COUNTRY_SCOPE_RAW}
        """
    ).strip()
    yaml_file: Path = tmp_path / TEST_YAML_FILE_MIDDLEWARE_CONFIG
    yaml_file.write_text(str_yaml_payload, encoding="utf-8")

    monkeypatch.setenv(ENV_CONFIGURATION_PATH, str(yaml_file))

    middleware_config: MiddlewareConfig = MiddlewareConfig()
    middleware_settings: PiiRedactionSettingsModel | None = (
        middleware_config.get_middleware_settings(PII_REDACTION)
    )
    assert middleware_settings is not None
    assert middleware_settings.direction == TEST_DIRECTION_INPUT_OUTPUT
    assert middleware_settings.strict_mode is True
    assert (
        middleware_settings.detection_profile
        == TEST_DETECTION_PROFILE_HIGH_ACCURACY_NORMALIZED
    )
    assert middleware_settings.language == TEST_LANGUAGE_NORMALIZED
    assert middleware_settings.country_scope == TEST_COUNTRY_SCOPE_NORMALIZED

    # Normal return on successful assertions for YAML normalization behavior.
    return None


def test_load_configuration_reads_yaml_and_normalizes_redaction_recognizers(
    monkeypatch: Any, tmp_path: Path
) -> None:
    """
    Ensures YAML load path normalizes the redaction_recognizers contract.

    Args:
        monkeypatch: Pytest fixture used to control environment variables for this test process.
        tmp_path: Temporary filesystem path fixture used to write a test YAML file.

    Returns:
        None
    """
    str_yaml_payload: str = dedent(
        f"""
        middleware:
          - name: {PII_REDACTION}
            enabled: true
            settings:
              direction: {TEST_DIRECTION_INPUT_ONLY}
              {REDACTION_RECOGNIZERS_KEY}:
                {PROXIMITY_WINDOW_CHARS_KEY}: "{TEST_RECOGNIZER_PROXIMITY_WINDOW_YAML_RAW}"
                {RECOGNIZER_SSN_LAST4_KEY}:
                  {RECOGNIZER_ENABLED_KEY}: "{TEST_RECOGNIZER_ENABLED_FALSE_RAW}"
                  {RECOGNIZER_CONFIDENCE_THRESHOLD_KEY}: "{TEST_RECOGNIZER_CONFIDENCE_THRESHOLD_RAW}"
                  {RECOGNIZER_CONTEXT_TERMS_KEY}: "{TEST_RECOGNIZER_CONTEXT_TERM_YAML}"
        """
    ).strip()
    yaml_file: Path = tmp_path / TEST_YAML_FILE_MIDDLEWARE_RECOGNIZERS
    yaml_file.write_text(str_yaml_payload, encoding="utf-8")

    monkeypatch.setenv(ENV_CONFIGURATION_PATH, str(yaml_file))

    middleware_config: MiddlewareConfig = MiddlewareConfig()
    middleware_settings: PiiRedactionSettingsModel | None = (
        middleware_config.get_middleware_settings(PII_REDACTION)
    )
    default_redaction_recognizers_settings: RedactionRecognizersSettingsModel = (
        RedactionRecognizersSettingsModel()
    )
    assert middleware_settings is not None
    assert (
        middleware_settings.redaction_recognizers.proximity_window_chars
        == TEST_RECOGNIZER_PROXIMITY_WINDOW_YAML_EXPECTED
    )
    assert middleware_settings.redaction_recognizers.ssn_last4.enabled is False
    assert (
        middleware_settings.redaction_recognizers.ssn_last4.confidence_threshold
        == TEST_RECOGNIZER_CONFIDENCE_THRESHOLD_EXPECTED
    )
    assert middleware_settings.redaction_recognizers.ssn_last4.context_terms == [
        TEST_RECOGNIZER_CONTEXT_TERM_YAML
    ]
    assert (
        middleware_settings.redaction_recognizers.cc_last4.model_dump()
        == default_redaction_recognizers_settings.cc_last4.model_dump()
    )
    assert (
        middleware_settings.redaction_recognizers.dob.model_dump()
        == default_redaction_recognizers_settings.dob.model_dump()
    )
    # Normal return on successful assertions for YAML recognizer normalization.
    return None


def test_load_configuration_falls_back_for_invalid_yaml_redaction_recognizers(
    monkeypatch: Any, tmp_path: Path
) -> None:
    """
    Ensures invalid YAML recognizer extension values safely fall back to defaults.

    Args:
        monkeypatch: Pytest fixture used to control environment variables for this test process.
        tmp_path: Temporary filesystem path fixture used to write a test YAML file.

    Returns:
        None
    """
    str_yaml_payload: str = dedent(
        f"""
        middleware:
          - name: {PII_REDACTION}
            enabled: true
            settings:
              {REDACTION_RECOGNIZERS_KEY}:
                {PROXIMITY_WINDOW_CHARS_KEY}: "{TEST_RECOGNIZER_INVALID_WINDOW_RAW}"
                {RECOGNIZER_CC_LAST4_KEY}: "{TEST_RECOGNIZER_INVALID_CC_RULE_RAW}"
                {RECOGNIZER_DOB_KEY}:
                  {RECOGNIZER_CONFIDENCE_THRESHOLD_KEY}: "{TEST_RECOGNIZER_INVALID_CONFIDENCE_THRESHOLD_RAW}"
                  {RECOGNIZER_CONTEXT_TERMS_KEY}: {TEST_RECOGNIZER_INVALID_CONTEXT_TERMS_RAW}
        """
    ).strip()
    yaml_file: Path = tmp_path / TEST_YAML_FILE_MIDDLEWARE_RECOGNIZERS_INVALID
    yaml_file.write_text(str_yaml_payload, encoding="utf-8")

    monkeypatch.setenv(ENV_CONFIGURATION_PATH, str(yaml_file))

    middleware_config: MiddlewareConfig = MiddlewareConfig()
    middleware_settings: PiiRedactionSettingsModel | None = (
        middleware_config.get_middleware_settings(PII_REDACTION)
    )
    default_redaction_recognizers_settings: RedactionRecognizersSettingsModel = (
        RedactionRecognizersSettingsModel()
    )
    assert middleware_settings is not None
    assert (
        middleware_settings.redaction_recognizers.model_dump()
        == default_redaction_recognizers_settings.model_dump()
    )
    # Normal return on successful assertions for invalid YAML recognizer fallback.
    return None


def test_load_configuration_reads_yaml_and_normalizes_observability_settings(
    monkeypatch: Any, tmp_path: Path
) -> None:
    """
    Ensures YAML file loading path works and returns normalized typed observability settings.

    Args:
        monkeypatch: Pytest fixture used to control environment variables for this test process.
        tmp_path: Temporary filesystem path fixture used to write a test YAML file.

    Returns:
        None
    """
    str_yaml_payload: str = dedent(
        f"""
        middleware:
          - name: {OBSERVABILITY}
            enabled: true
            settings:
              direction: {TEST_DIRECTION_INPUT_OUTPUT_RAW}
              {CAPABILITIES_KEY}:
                - {TEST_OBSERVABILITY_CAPABILITY_IMAGES_RAW}
                - {TEST_OBSERVABILITY_CAPABILITY_TTS_RAW}
              {LOG_LEVEL_KEY}: {TEST_OBSERVABILITY_LOG_LEVEL_ERROR_RAW}
              {TOKEN_COUNT_MODE_KEY}: {TEST_OBSERVABILITY_TOKEN_COUNT_MODE_NONE_RAW}
              {EMIT_ERROR_EVENTS_KEY}: false
        """
    ).strip()
    yaml_file: Path = tmp_path / TEST_YAML_FILE_OBSERVABILITY_CONFIG
    yaml_file.write_text(str_yaml_payload, encoding="utf-8")

    monkeypatch.setenv(ENV_CONFIGURATION_PATH, str(yaml_file))

    middleware_config: MiddlewareConfig = MiddlewareConfig()
    middleware_settings: ObservabilitySettingsModel | None = (
        middleware_config.get_observability_settings()
    )
    assert middleware_settings is not None
    assert middleware_settings.direction == TEST_DIRECTION_INPUT_OUTPUT
    assert middleware_settings.capabilities == ["images", "tts"]
    assert middleware_settings.log_level == "ERROR"
    assert middleware_settings.token_count_mode == "none"
    assert middleware_settings.emit_error_events is False
    # Normal return on successful assertions for observability YAML normalization behavior.
    return None


def test_load_configuration_observability_yaml_drops_unsupported_capabilities(
    monkeypatch: Any,
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
) -> None:
    """
    Ensures invalid observability capability values from YAML are dropped with a warning.

    Args:
        monkeypatch: Pytest fixture used to control environment variables for this test process.
        caplog: Pytest fixture used to capture logger output for assertions.
        tmp_path: Temporary filesystem path fixture used to write a test YAML file.

    Returns:
        None
    """
    str_yaml_payload: str = dedent(
        f"""
        middleware:
          - name: {OBSERVABILITY}
            enabled: true
            settings:
              {CAPABILITIES_KEY}:
                - {TEST_OBSERVABILITY_CAPABILITY_TTS_RAW}
                - {TEST_OBSERVABILITY_INVALID_CAPABILITY_RAW}
        """
    ).strip()
    yaml_file: Path = tmp_path / TEST_YAML_FILE_OBSERVABILITY_INVALID
    yaml_file.write_text(str_yaml_payload, encoding="utf-8")

    monkeypatch.setenv(ENV_CONFIGURATION_PATH, str(yaml_file))
    caplog.set_level(logging.WARNING, logger=TEST_LOGGER_MIDDLEWARE_CONFIG)

    middleware_config: MiddlewareConfig = MiddlewareConfig()
    middleware_settings: ObservabilitySettingsModel | None = (
        middleware_config.get_observability_settings()
    )
    assert middleware_settings is not None
    assert middleware_settings.capabilities == ["tts"]
    list_str_logged_messages: list[str] = []
    # Loop through captured log records and collect fully rendered log message text.
    for log_record in caplog.records:
        list_str_logged_messages.append(log_record.getMessage())
    assert any(
        OBSERVABILITY_CAPABILITIES_DROPPED_VALUES_WARNING_MESSAGE
        % (CAPABILITIES_KEY, [TEST_OBSERVABILITY_INVALID_CAPABILITY_RAW])
        in str_logged_message
        for str_logged_message in list_str_logged_messages
    )
    # Normal return on successful assertions for observability YAML fallback behavior.
    return None


def test_load_configuration_treats_non_dictionary_yaml_root_as_disabled(
    monkeypatch: Any, tmp_path: Path
) -> None:
    """
    Ensures non-dictionary YAML roots are rejected and middleware remains disabled.

    Args:
        monkeypatch: Pytest fixture used to control environment variables for this test process.
        tmp_path: Temporary filesystem path fixture used to write a test YAML file.

    Returns:
        None
    """
    yaml_file: Path = tmp_path / TEST_YAML_FILE_MIDDLEWARE_INVALID_ROOT
    yaml_file.write_text("- not-a-dictionary-root\n", encoding="utf-8")

    monkeypatch.setenv(ENV_CONFIGURATION_PATH, str(yaml_file))

    middleware_config: MiddlewareConfig = MiddlewareConfig()
    middleware_settings: PiiRedactionSettingsModel | None = (
        middleware_config.get_middleware_settings(PII_REDACTION)
    )
    assert middleware_settings is None
    # Normal return on successful assertions for invalid root fallback behavior.
    return None


def test_load_configuration_treats_non_list_middleware_key_as_disabled(
    monkeypatch: Any, tmp_path: Path
) -> None:
    """
    Ensures a non-list `middleware` value is treated as disabled.

    Args:
        monkeypatch: Pytest fixture used to control environment variables for this test process.
        tmp_path: Temporary filesystem path fixture used to write a test YAML file.

    Returns:
        None
    """
    str_yaml_payload: str = dedent(
        """
        middleware:
          name: pii_redaction
        """
    ).strip()
    yaml_file: Path = tmp_path / TEST_YAML_FILE_MIDDLEWARE_NOT_LIST
    yaml_file.write_text(str_yaml_payload, encoding="utf-8")

    monkeypatch.setenv(ENV_CONFIGURATION_PATH, str(yaml_file))

    middleware_config: MiddlewareConfig = MiddlewareConfig()
    middleware_settings: PiiRedactionSettingsModel | None = (
        middleware_config.get_middleware_settings(PII_REDACTION)
    )
    assert middleware_settings is None
    # Normal return on successful assertions for invalid middleware-list behavior.
    return None


def test_load_configuration_returns_disabled_when_yaml_path_missing(
    monkeypatch: Any,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """
    Ensures middleware is disabled when no YAML configuration path is provided.

    Args:
        monkeypatch: Pytest fixture used to control environment variables for this test process.

    Returns:
        None
    """
    caplog.set_level(logging.WARNING, logger=TEST_LOGGER_MIDDLEWARE_CONFIG)
    monkeypatch.delenv(ENV_CONFIGURATION_PATH, raising=False)
    middleware_config: MiddlewareConfig = MiddlewareConfig()
    middleware_settings: PiiRedactionSettingsModel | None = (
        middleware_config.get_middleware_settings(PII_REDACTION)
    )
    assert middleware_settings is None
    list_str_logged_messages: list[str] = []
    # Loop through captured log records and collect fully rendered log message text.
    for log_record in caplog.records:
        list_str_logged_messages.append(log_record.getMessage())
    assert list_str_logged_messages == []
    # Normal return on successful assertions for missing-YAML disabled behavior.
    return None


def test_load_configuration_warns_when_yaml_path_is_set_but_file_missing(
    monkeypatch: Any,
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
) -> None:
    """
    Ensures a configured but missing YAML path emits a warning and middleware stays disabled.

    Args:
        monkeypatch: Pytest fixture used to control environment variables for this test process.
        caplog: Pytest fixture used to capture logger output for assertions.
        tmp_path: Temporary filesystem path fixture used to build a missing path.

    Returns:
        None
    """
    caplog.set_level(logging.WARNING, logger=TEST_LOGGER_MIDDLEWARE_CONFIG)
    yaml_file_missing: Path = tmp_path / "middleware_missing.yaml"
    monkeypatch.setenv(ENV_CONFIGURATION_PATH, str(yaml_file_missing))

    middleware_config: MiddlewareConfig = MiddlewareConfig()
    middleware_settings: PiiRedactionSettingsModel | None = (
        middleware_config.get_middleware_settings(PII_REDACTION)
    )
    assert middleware_settings is None
    list_str_logged_messages: list[str] = []
    # Loop through captured log records and collect fully rendered log message text.
    for log_record in caplog.records:
        list_str_logged_messages.append(log_record.getMessage())
    assert any(
        MIDDLEWARE_PATH_SET_BUT_MISSING_WARNING_MESSAGE
        % (ENV_CONFIGURATION_PATH, str(yaml_file_missing))
        in str_logged_message
        for str_logged_message in list_str_logged_messages
    )
    # Normal return on successful assertions for missing-path warning behavior.
    return None


def test_load_configuration_warns_when_yaml_file_is_empty(
    monkeypatch: Any,
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
) -> None:
    """
    Ensures an empty YAML config file emits a warning and middleware stays disabled.

    Args:
        monkeypatch: Pytest fixture used to control environment variables for this test process.
        caplog: Pytest fixture used to capture logger output for assertions.
        tmp_path: Temporary filesystem path fixture used to write the empty YAML file.

    Returns:
        None
    """
    caplog.set_level(logging.WARNING, logger=TEST_LOGGER_MIDDLEWARE_CONFIG)
    yaml_file: Path = tmp_path / TEST_YAML_FILE_MIDDLEWARE_EMPTY
    yaml_file.write_text("", encoding="utf-8")
    monkeypatch.setenv(ENV_CONFIGURATION_PATH, str(yaml_file))

    middleware_config: MiddlewareConfig = MiddlewareConfig()
    middleware_settings: PiiRedactionSettingsModel | None = (
        middleware_config.get_middleware_settings(PII_REDACTION)
    )
    assert middleware_settings is None
    list_str_logged_messages: list[str] = []
    # Loop through captured log records and collect fully rendered log message text.
    for log_record in caplog.records:
        list_str_logged_messages.append(log_record.getMessage())
    assert any(
        MIDDLEWARE_EMPTY_FILE_WARNING_MESSAGE % str(yaml_file) in str_logged_message
        for str_logged_message in list_str_logged_messages
    )
    # Normal return on successful assertions for empty-yaml warning behavior.
    return None


def test_load_configuration_treats_invalid_yaml_as_disabled(
    monkeypatch: Any, tmp_path: Path
) -> None:
    """
    Ensures YAML parse failures fall back to disabled middleware behavior.

    Args:
        monkeypatch: Pytest fixture used to control environment variables for this test process.
        tmp_path: Temporary filesystem path fixture used to write a test YAML file.

    Returns:
        None
    """
    yaml_file: Path = tmp_path / TEST_YAML_FILE_MIDDLEWARE_INVALID_SYNTAX
    yaml_file.write_text("middleware: [\n", encoding="utf-8")

    monkeypatch.setenv(ENV_CONFIGURATION_PATH, str(yaml_file))

    middleware_config: MiddlewareConfig = MiddlewareConfig()
    middleware_settings: PiiRedactionSettingsModel | None = (
        middleware_config.get_middleware_settings(PII_REDACTION)
    )
    assert middleware_settings is None
    # Normal return on successful assertions for YAML-parse-error fallback.
    return None
