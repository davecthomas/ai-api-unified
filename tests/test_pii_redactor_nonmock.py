from __future__ import annotations

import logging
import os
import secrets
from typing import Any

import pytest
from ai_api_unified.middleware.impl.base_redactor import (
    BaseRedactorLayer,
    RedactionResult,
)
from ai_api_unified.middleware.middleware_config import (
    PiiRedactionSettingsModel,
)
from ai_api_unified.middleware.pii_redactor import _get_configured_redactor
from ai_api_unified.middleware.redaction_exceptions import (
    PiiRedactionDependencyUnavailableError,
    PiiRedactionRuntimeError,
)

# Use standard pytest live logging output (configured in pytest.ini).
console_logger: logging.Logger = logging.getLogger(__name__)
console_logger.setLevel(logging.INFO)

# Ensure tldextract cache writes stay inside a writable test path.
TLDEXTRACT_CACHE_DIRECTORY: str = "/tmp/ai_api_unified_tldextract_cache"
os.environ.setdefault("TLDEXTRACT_CACHE", TLDEXTRACT_CACHE_DIRECTORY)
os.makedirs(TLDEXTRACT_CACHE_DIRECTORY, exist_ok=True)

# Centralized constants pool to prevent hardcoded fat-finger errors in test assertions.
# SSNs specifically require valid block prefixes like '149' for Presidio to catch them properly.
LIST_STR_SSNS: list[str] = [
    "135-79-2468",
    "246-81-3579",
    "318-52-6407",
    "427-63-9150",
    "539-74-1286",
    "641-85-2397",
    "752-96-3408",
    "863-17-4519",
    "974-28-5620",
    "585-39-6731",
]

LIST_STR_EMAILS: list[str] = [
    "olivia.carter@example.com",
    "ethan.brooks@hotmail.com",
    "maya.patel@sample.co",
    "noah.bennett@demo.net",
    "sofia.ramirez@aol.com",
    "liam.foster@gmail.com",
    "ava.nguyen@inbox.test",
    "caleb.turner@acme-mail.com",
    "zoe.mitchell@yahoo.com",
    "isaac.reed@github.com",
]

LIST_STR_PHONES: list[str] = [
    "(212) 555-0184",
    "646-555-7730",
    "917.555.2049",
    "303 555 4412",
    "555-0199",
    "+44 20 7946 0958",
    "+1 (415) 555-3007",
    "020 7946 1234",
    "+61 2 9374 4000",
    "1-800-555-1212",
]

LIST_STR_NAMES: list[str] = [
    "Olivia Carter",
    "Ethan Brooks",
    "Maya Patel",
    "Noah Bennett",
    "Sofia Ramirez",
    "Liam Foster",
    "Ava Nguyen",
    "Caleb Turner",
    "Dave Mitchell",
    "Isaac Reed",
]

LIST_STR_ADDRESSES: list[str] = [
    "742 Willow Creek Lane, Apt 3B, Madison, WI 53703",  # complete
    "1189 Cedar Hill Road, Boise, ID",  # missing ZIP
    "56 Pine Meadow Court, Raleigh, 27609",  # missing state
    "2407 Elmwood Drive, Unit 12",  # street + unit only
    "931 Harbor View Terrace, Sacramento",  # missing state and ZIP
    "Apartment 4B, 123 Main Street, Denver, CO",  # missing ZIP
    "88 North Maple Ave",  # street only
    "500 Westlake Blvd, Suite 210, Austin",  # missing state and ZIP
    "12B Riverfront Lofts, Portland, OR 97205",  # non-standard unit-first format
    "Corner of Elm St and 5th Ave, Springfield",  # intersection-style location
]

DETECTION_PROFILE_KEY: str = "detection_profile"
DETECTION_PROFILE_HIGH_ACCURACY: str = "high_accuracy"
DETECTION_PROFILE_BALANCED: str = "balanced"
DETECTION_PROFILE_LOW_MEMORY: str = "low_memory"
INVALID_DETECTION_PROFILE_RAW: str = "invalid-profile-token"
BOOLEAN_TRUTHINESS_TEST_VALUE: list[int] = [1, 2, 3]
PRESIDIO_REDACTOR_LOGGER_NAME: str = (
    "ai_api_unified.middleware.impl._presidio_redactor"
)
MATRIX_PHONE: str = "1-800-555-1212"
MATRIX_SSN: str = "585-39-6731"
MATRIX_ENTITY_PHONE: str = "PHONE"
MATRIX_ENTITY_SSN: str = "SSN"
MATRIX_ENTITY_DOB: str = "DOB"
MATRIX_ENTITY_CC_LAST4: str = "CC_LAST4"
TEXT_CONTEXTUAL_SSN_LAST4: str = "Please verify SSN ending in 6789 for onboarding."
TEXT_CONTEXTUAL_CC_LAST4: str = "Please keep Visa ending in 4242 on file for billing."
TEXT_SHORT_CONTEXTUAL_CC_LAST4: str = "Visa ending in 4242 is on file."
TEXT_SINGLE_TERM_CONTEXTUAL_CC_LAST4: str = "Please keep visa 4242 on file."
TEXT_SOCIAL_SECURITY_CARD_LAST4: str = (
    "Social security card ending in 6789 was verified."
)
TEXT_CONTEXTUAL_DOB_ISO: str = "Please verify DOB 1991-12-31 before onboarding."
TEXT_CONTEXTUAL_DOB_MONTH_NAME: str = (
    "Please verify date of birth January 22, 1988 before onboarding."
)
TEXT_CONTEXTUAL_DOB_DMY: str = "The applicant was born 31-12-1991 in Boston."
TEXT_NEGATIVE_DOB_INVOICE_DATE: str = "Invoice date 1991-12-31 was archived."
TEXT_MIXED_RECORD_ALL_PII: str = (
    "Test mixed record: Claire Foster, date of birth 1992-06-18, "
    "phone (646) 555-0121, email claire.foster@demo-company.net, "
    "Visa card ending in 1129, SSN last 4 is 5542."
)
TEXT_MIXED_RECORD_WITH_USADDRESS_FIELD_LABELS: str = (
    "Record for Thomas Nguyen, DOB 2000-01-29, SSN last 4 4418, "
    "email thomas.nguyen@qa-mail.dev, address 255 Harbor St, Miami, FL 33132."
)
TEXT_NONCONTEXTUAL_FOUR_DIGIT_NUMBER: str = (
    "Please verify order number 6789 for shipping."
)
EXPECTED_SSN_LAST4_TOKEN: str = "6789"
EXPECTED_REDACTED_SSN_TOKEN: str = "[REDACTED:SSN]"
EXPECTED_DOB_TOKEN_ISO: str = "1991-12-31"
EXPECTED_DOB_TOKEN_MONTH_NAME: str = "January 22, 1988"
EXPECTED_DOB_TOKEN_DMY: str = "31-12-1991"
EXPECTED_REDACTED_DOB_TOKEN: str = "[REDACTED:DOB]"
EXPECTED_CC_LAST4_TOKEN: str = "4242"
EXPECTED_REDACTED_CC_LAST4_TOKEN: str = "[REDACTED:CC_LAST4]"
EXPECTED_CUSTOM_CC_LAST4_TOKEN: str = "[REDACTED:CARD_TAIL]"
EXPECTED_CUSTOM_DOB_TOKEN: str = "[REDACTED:BIRTH_DATE]"
EXPECTED_MIXED_RECORD_REDACTION_COUNT: int = 6
EXPECTED_REDACTED_ADDRESS_TOKEN: str = "[REDACTED:ADDRESS]"
EXPECTED_USADDRESS_FIELD_LABEL_REDACTION_COUNT: int = 5
EXPECTED_USADDRESS_FIELD_LABEL_STREET_FRAGMENT: str = "255 Harbor St"
EXPECTED_USADDRESS_FIELD_LABEL_CITY_FRAGMENT: str = "Miami"
EXPECTED_USADDRESS_FIELD_LABEL_STATE_ZIP_FRAGMENT: str = "FL 33132"

LIST_DICT_REGEX_CONFIG_VARIATIONS: list[dict[str, Any]] = [
    {
        "str_case_id": "regex_defaults_redact_phone_and_ssn",
        "dict_setting_overrides": {},
        "list_str_expected_contains": [
            "[REDACTED:PHONE]",
            "[REDACTED:SSN]",
        ],
        "list_str_expected_not_contains": [
            MATRIX_PHONE,
            MATRIX_SSN,
        ],
    },
    {
        "str_case_id": "regex_allowed_entities_keeps_phone_only",
        "dict_setting_overrides": {
            "allowed_entities": [MATRIX_ENTITY_PHONE],
        },
        "list_str_expected_contains": [
            MATRIX_PHONE,
            "[REDACTED:SSN]",
        ],
        "list_str_expected_not_contains": [
            MATRIX_SSN,
            "[REDACTED:PHONE]",
        ],
    },
    {
        "str_case_id": "regex_allowed_entities_keeps_phone_and_ssn",
        "dict_setting_overrides": {
            "allowed_entities": [
                MATRIX_ENTITY_PHONE,
                MATRIX_ENTITY_SSN,
            ]
        },
        "list_str_expected_contains": [
            MATRIX_PHONE,
            MATRIX_SSN,
        ],
        "list_str_expected_not_contains": [
            "[REDACTED:PHONE]",
            "[REDACTED:SSN]",
        ],
    },
    {
        "str_case_id": "regex_full_label_mapping_rewrites_phone_and_ssn_labels",
        "dict_setting_overrides": {
            "entity_label_map": {
                MATRIX_ENTITY_PHONE: "CALLBACK",
                MATRIX_ENTITY_SSN: "TAX_ID",
            }
        },
        "list_str_expected_contains": [
            "[REDACTED:CALLBACK]",
            "[REDACTED:TAX_ID]",
        ],
        "list_str_expected_not_contains": [
            MATRIX_PHONE,
            MATRIX_SSN,
        ],
    },
    {
        "str_case_id": "regex_partial_entity_map_updates_label_without_changing_scope",
        "dict_setting_overrides": {"entity_label_map": {MATRIX_ENTITY_SSN: "TAX_ID"}},
        "list_str_expected_contains": [
            "[REDACTED:PHONE]",
            "[REDACTED:TAX_ID]",
        ],
        "list_str_expected_not_contains": [
            "[REDACTED:SSN]",
            MATRIX_PHONE,
            MATRIX_SSN,
        ],
    },
    {
        "str_case_id": "regex_default_redaction_label_overrides_prefix",
        "dict_setting_overrides": {"default_redaction_label": "SECRET"},
        "list_str_expected_contains": [
            "[SECRET:PHONE]",
            "[SECRET:SSN]",
        ],
        "list_str_expected_not_contains": [
            MATRIX_PHONE,
            MATRIX_SSN,
        ],
    },
]


def _build_redactor_middleware_settings(
    str_detection_profile: str,
    dict_setting_overrides: dict[str, Any] | None = None,
) -> PiiRedactionSettingsModel:
    """
    Builds and validates a typed middleware settings model for redactor construction.

    Args:
        str_detection_profile: The desired middleware detection profile key value for the redactor instance.
        dict_setting_overrides: Optional settings merged over the profile baseline.

    Returns:
        A validated PiiRedactionSettingsModel instance.
    """
    dict_middleware_settings: dict[str, Any] = {
        DETECTION_PROFILE_KEY: str_detection_profile
    }
    if dict_setting_overrides is not None:
        dict_middleware_settings.update(dict_setting_overrides)
    middleware_settings: PiiRedactionSettingsModel = (
        PiiRedactionSettingsModel.model_validate(dict_middleware_settings)
    )
    # Normal return with validated middleware settings for redactor construction.
    return middleware_settings


def _create_redactor(
    middleware_settings: PiiRedactionSettingsModel, bool_strict_mode: bool = False
) -> BaseRedactorLayer:
    """
    Creates a configured redactor through middleware factory flow using typed settings.

    Args:
        middleware_settings: Typed middleware settings model for pii_redaction.
        bool_strict_mode: Strict-mode behavior for dependency failures.

    Returns:
        A constructed BaseRedactorLayer implementation.
    """
    # Normal return with redactor created via middleware factory boundary.
    return _get_configured_redactor(
        configuration=middleware_settings,
        bool_strict_mode=bool_strict_mode,
    )


def _assert_runtime_dependencies_available() -> None:
    """
    Skips this module when optional middleware redaction dependencies are missing.

    Args:
        None

    Returns:
        None
    """
    middleware_settings: PiiRedactionSettingsModel = (
        _build_redactor_middleware_settings(DETECTION_PROFILE_LOW_MEMORY)
    )
    try:
        _create_redactor(middleware_settings, bool_strict_mode=True)
    except PiiRedactionDependencyUnavailableError:
        pytest.skip(
            "Optional PII redaction dependencies are not installed.",
            allow_module_level=True,
        )
    # Normal return once optional dependencies are confirmed.
    return None


_assert_runtime_dependencies_available()


def test_presidio_redactor_warns_on_invalid_detection_profile_fallback(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """
    Ensures direct Presidio redactor construction warns when detection profile is invalid.

    Args:
        caplog: Pytest fixture used to capture logger output for assertions.

    Returns:
        None
    """
    from ai_api_unified.middleware.impl._presidio_redactor import (
        DETECTION_PROFILE_BALANCED as PRESIDIO_DETECTION_PROFILE_BALANCED,
    )
    from ai_api_unified.middleware.impl._presidio_redactor import (
        DETECTION_PROFILE_KEY as PRESIDIO_DETECTION_PROFILE_KEY,
    )
    from ai_api_unified.middleware.impl._presidio_redactor import (
        INVALID_DETECTION_PROFILE_FALLBACK_WARNING_MESSAGE,
    )
    from ai_api_unified.middleware.impl._presidio_redactor import PiiRedactor

    caplog.set_level(logging.WARNING, logger=PRESIDIO_REDACTOR_LOGGER_NAME)
    redactor: PiiRedactor = PiiRedactor(
        dict_config={PRESIDIO_DETECTION_PROFILE_KEY: INVALID_DETECTION_PROFILE_RAW}
    )

    assert redactor.str_detection_profile == PRESIDIO_DETECTION_PROFILE_BALANCED
    list_str_logged_messages: list[str] = []
    # Loop through captured log records and collect fully rendered log message text.
    for log_record in caplog.records:
        list_str_logged_messages.append(log_record.getMessage())
    assert any(
        INVALID_DETECTION_PROFILE_FALLBACK_WARNING_MESSAGE
        % (INVALID_DETECTION_PROFILE_RAW, PRESIDIO_DETECTION_PROFILE_BALANCED)
        in str_logged_message
        for str_logged_message in list_str_logged_messages
    )


def test_presidio_redactor_warns_on_truthiness_bool_coercion(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """
    Ensures direct Presidio redactor construction warns when bool coercion uses truthiness fallback.

    Args:
        caplog: Pytest fixture used to capture logger output for assertions.

    Returns:
        None
    """
    from ai_api_unified.middleware.impl._presidio_redactor import (
        ADDRESS_DETECTION_ENABLED_KEY as PRESIDIO_ADDRESS_DETECTION_ENABLED_KEY,
    )
    from ai_api_unified.middleware.impl._presidio_redactor import (
        BOOL_TRUTHINESS_FALLBACK_WARNING_MESSAGE,
    )
    from ai_api_unified.middleware.impl._presidio_redactor import PiiRedactor

    caplog.set_level(logging.WARNING, logger=PRESIDIO_REDACTOR_LOGGER_NAME)
    redactor: PiiRedactor = PiiRedactor(
        dict_config={
            PRESIDIO_ADDRESS_DETECTION_ENABLED_KEY: BOOLEAN_TRUTHINESS_TEST_VALUE
        }
    )

    assert redactor.bool_address_detection_enabled is True
    list_str_logged_messages: list[str] = []
    # Loop through captured log records and collect fully rendered log message text.
    for log_record in caplog.records:
        list_str_logged_messages.append(log_record.getMessage())
    assert any(
        BOOL_TRUTHINESS_FALLBACK_WARNING_MESSAGE
        % (PRESIDIO_ADDRESS_DETECTION_ENABLED_KEY, list.__name__)
        in str_logged_message
        for str_logged_message in list_str_logged_messages
    )


def _sanitize_and_log_categories(
    str_profile_name: str, redactor: BaseRedactorLayer, str_input: str
) -> RedactionResult:
    """
    Runs redaction and logs canonical category metadata from the base contract.

    Args:
        str_profile_name: Human-readable profile label for diagnostics.
        redactor: Base redactor interface implementation.
        str_input: Raw text to sanitize.

    Returns:
        RedactionResult including sanitized text and detected categories.
    """
    redaction_result: RedactionResult = redactor.sanitize_with_result(str_input)
    console_logger.info(
        "[%s] Detected Categories canonical=%s",
        str_profile_name,
        redaction_result.list_str_detected_categories,
    )
    # Normal return with redaction output for test assertions.
    return redaction_result


def _build_runtime_ready_redactor(
    str_detection_profile: str, dict_setting_overrides: dict[str, Any] | None = None
) -> BaseRedactorLayer:
    """
    Creates a redactor and verifies the profile can execute in the current runtime.

    Args:
        str_detection_profile: Detection profile value to instantiate.
        dict_setting_overrides: Optional settings overrides merged into profile settings.

    Returns:
        A runtime-ready BaseRedactorLayer.
    """
    middleware_settings: PiiRedactionSettingsModel = (
        _build_redactor_middleware_settings(
            str_detection_profile,
            dict_setting_overrides,
        )
    )
    redactor: BaseRedactorLayer = _create_redactor(middleware_settings)
    try:
        redactor.sanitize_with_result("Runtime warmup for redaction tests.")
    except PiiRedactionRuntimeError:
        pytest.skip(
            f"PII detection profile '{str_detection_profile}' is unavailable in this runtime."
        )
    # Normal return once runtime profile dependencies are confirmed.
    return redactor


@pytest.fixture(scope="module")
def redactor_large() -> BaseRedactorLayer:
    """
    Fixture to provide a BaseRedactorLayer using the high-accuracy detection profile.

    Args:
        None

    Returns:
        Runtime-ready large-profile redactor.
    """
    # Normal return for module-scoped large profile redactor.
    return _build_runtime_ready_redactor(DETECTION_PROFILE_HIGH_ACCURACY)


@pytest.fixture(scope="module")
def redactor_small() -> BaseRedactorLayer:
    """
    Fixture to provide a BaseRedactorLayer using the balanced detection profile.

    Args:
        None

    Returns:
        Runtime-ready small-profile redactor.
    """
    # Normal return for module-scoped small profile redactor.
    return _build_runtime_ready_redactor(DETECTION_PROFILE_BALANCED)


@pytest.fixture(scope="module")
def redactor_regex() -> BaseRedactorLayer:
    """
    Fixture to provide a BaseRedactorLayer using the low-memory detection profile.

    Args:
        None

    Returns:
        Runtime-ready regex-profile redactor.
    """
    # Normal return for module-scoped regex profile redactor.
    return _build_runtime_ready_redactor(DETECTION_PROFILE_LOW_MEMORY)


def test_deterministic_pii_across_all_profiles() -> None:
    """
    Emails, Phones, and SSNs redact consistently across all profile types.
    """
    str_email: str = secrets.choice(LIST_STR_EMAILS)
    str_phone: str = MATRIX_PHONE
    str_ssn: str = secrets.choice(LIST_STR_SSNS)
    str_input: str = f"Contact {str_email} or {str_phone} or {str_ssn}."

    redactor_large_no_address_detection: BaseRedactorLayer = (
        _build_runtime_ready_redactor(
            DETECTION_PROFILE_HIGH_ACCURACY,
            {"address_detection_enabled": False},
        )
    )
    redactor_small_no_address_detection: BaseRedactorLayer = (
        _build_runtime_ready_redactor(
            DETECTION_PROFILE_BALANCED,
            {"address_detection_enabled": False},
        )
    )
    redactor_regex_no_address_detection: BaseRedactorLayer = (
        _build_runtime_ready_redactor(
            DETECTION_PROFILE_LOW_MEMORY,
            {"address_detection_enabled": False},
        )
    )

    # Loop over deterministic profile variants with address parser disabled to
    # isolate deterministic regex-based entity behavior.
    for str_profile, redactor in [
        ("LARGE", redactor_large_no_address_detection),
        ("SMALL", redactor_small_no_address_detection),
        ("REGEX", redactor_regex_no_address_detection),
    ]:
        console_logger.info("[%s DETERMINISTIC] Before: %s", str_profile, str_input)
        redaction_result: RedactionResult = _sanitize_and_log_categories(
            f"{str_profile} DETERMINISTIC", redactor, str_input
        )
        str_output: str = redaction_result.str_sanitized_text
        console_logger.info("[%s DETERMINISTIC] After: %s", str_profile, str_output)

        assert "[REDACTED:EMAIL]" in str_output
        assert str_email not in str_output
        assert "[REDACTED:PHONE]" in str_output
        assert str_phone not in str_output
        assert "[REDACTED:SSN]" in str_output
        assert str_ssn not in str_output
        assert "EMAIL" in redaction_result.list_str_detected_categories
        assert "PHONE" in redaction_result.list_str_detected_categories
        assert "SSN" in redaction_result.list_str_detected_categories


def test_contextual_ner_large_profile(redactor_large: BaseRedactorLayer) -> None:
    """
    The high-accuracy detection profile redacts configured US address variants and names.
    """
    # Loop through configured address samples to validate contextual NER coverage.
    for str_address in LIST_STR_ADDRESSES:
        str_name: str = secrets.choice(LIST_STR_NAMES)
        str_input: str = f"Hello, my name is {str_name} and I live at {str_address}."

        console_logger.info("[LARGE NER] Before: %s", str_input)
        redaction_result: RedactionResult = _sanitize_and_log_categories(
            "LARGE NER", redactor_large, str_input
        )
        str_output: str = redaction_result.str_sanitized_text
        console_logger.info("[LARGE NER] After: %s", str_output)

        assert "[REDACTED:NAME]" in str_output
        assert str_name not in str_output
        assert "[REDACTED:ADDRESS]" in str_output
        assert str_address not in str_output


def test_contextual_ner_regex_profile_tradeoff(
    redactor_regex: BaseRedactorLayer,
) -> None:
    """
    Regex profile keeps contextual names and addresses unchanged by design.
    """
    str_name: str = secrets.choice(LIST_STR_NAMES)
    str_address: str = secrets.choice(LIST_STR_ADDRESSES)
    str_input: str = f"Hello, my name is {str_name} and I live at {str_address}."

    console_logger.info("[REGEX NER TRADE-OFF] Before: %s", str_input)
    redaction_result: RedactionResult = _sanitize_and_log_categories(
        "REGEX NER TRADE-OFF", redactor_regex, str_input
    )
    str_output: str = redaction_result.str_sanitized_text
    console_logger.info("[REGEX NER TRADE-OFF] After: %s", str_output)

    assert "[REDACTED:NAME]" not in str_output
    assert str_name in str_output
    assert "[REDACTED:ADDRESS]" not in str_output
    assert str_address in str_output


def test_usaddress_span_precedence_over_person_fragment(
    redactor_large: BaseRedactorLayer,
) -> None:
    """
    Ensures full address redaction wins over overlapping person-like fragments.
    """
    str_input: str = (
        "Please update Ava Nguyen's profile with address: "
        "742 Willow Creek Lane, Apt 3B, Madison, WI 53703."
    )

    redaction_result: RedactionResult = _sanitize_and_log_categories(
        "USADDRESS PRECEDENCE", redactor_large, str_input
    )
    str_output: str = redaction_result.str_sanitized_text
    console_logger.info("[USADDRESS PRECEDENCE] After: %s", str_output)

    assert "[REDACTED:ADDRESS]" in str_output
    assert "742 Willow Creek Lane" not in str_output
    assert "Willow Creek Lane" not in str_output
    assert "[REDACTED:NAME]" in str_output


def test_redaction_with_allowed_entities() -> None:
    """
    Configuring canonical allow-list categories passes those categories through.
    """
    str_email: str = secrets.choice(LIST_STR_EMAILS)
    str_ssn: str = secrets.choice(LIST_STR_SSNS)

    middleware_settings: PiiRedactionSettingsModel = (
        _build_redactor_middleware_settings(
            DETECTION_PROFILE_LOW_MEMORY,
            {"allowed_entities": ["EMAIL"]},
        )
    )
    redactor: BaseRedactorLayer = _create_redactor(middleware_settings)
    str_input: str = f"Send it to {str_email} or call me back. SSN: {str_ssn}"
    str_output: str = redactor.sanitize_with_result(str_input).str_sanitized_text

    assert str_email in str_output
    assert str_ssn not in str_output
    assert "[REDACTED:SSN]" in str_output


def test_redaction_empty_input(redactor_regex: BaseRedactorLayer) -> None:
    """
    Empty input returns empty output with no categories.
    """
    redaction_result: RedactionResult = redactor_regex.sanitize_with_result("")
    assert redaction_result.str_sanitized_text == ""
    assert redaction_result.list_str_detected_categories == []


def test_redaction_whitespace_input(redactor_large: BaseRedactorLayer) -> None:
    """
    Whitespace-only input returns unchanged and no categories.
    """
    str_input: str = "   \n \t "
    redaction_result: RedactionResult = redactor_large.sanitize_with_result(str_input)
    assert redaction_result.str_sanitized_text == str_input
    assert redaction_result.list_str_detected_categories == []


def test_redaction_no_pii(redactor_small: BaseRedactorLayer) -> None:
    """
    Normal text with no PII passes through unchanged.
    """
    str_input: str = "We should get Mexican food today."
    redaction_result: RedactionResult = redactor_small.sanitize_with_result(str_input)
    assert redaction_result.str_sanitized_text == str_input
    assert redaction_result.list_str_detected_categories == []


def test_redaction_multiple_ssns(redactor_regex: BaseRedactorLayer) -> None:
    """
    Inputs containing multiple SSNs redact each SSN occurrence.
    """
    str_ssn_one: str = LIST_STR_SSNS[0]
    str_ssn_two: str = LIST_STR_SSNS[1]

    str_input: str = f"Records say {str_ssn_one} and {str_ssn_two}."
    str_output: str = redactor_regex.sanitize_with_result(str_input).str_sanitized_text
    assert str_ssn_one not in str_output
    assert str_ssn_two not in str_output
    assert str_output.count("[REDACTED:SSN]") == 2


def test_redaction_detects_contextual_ssn_last4(
    redactor_small: BaseRedactorLayer,
) -> None:
    """
    Balanced profile redacts SSN last-4 candidates only when local SSN context is present.
    """
    redaction_result: RedactionResult = redactor_small.sanitize_with_result(
        TEXT_CONTEXTUAL_SSN_LAST4
    )
    str_output: str = redaction_result.str_sanitized_text
    assert EXPECTED_SSN_LAST4_TOKEN not in str_output
    assert EXPECTED_REDACTED_SSN_TOKEN in str_output
    assert "SSN" in redaction_result.list_str_detected_categories


def test_redaction_ignores_noncontextual_four_digit_numbers(
    redactor_small: BaseRedactorLayer,
) -> None:
    """
    Balanced profile leaves generic 4-digit numbers unchanged when SSN context is absent.
    """
    middleware_settings: PiiRedactionSettingsModel = (
        _build_redactor_middleware_settings(
            DETECTION_PROFILE_BALANCED,
            {"address_detection_enabled": False},
        )
    )
    redactor: BaseRedactorLayer = _create_redactor(middleware_settings)
    redaction_result: RedactionResult = redactor.sanitize_with_result(
        TEXT_NONCONTEXTUAL_FOUR_DIGIT_NUMBER
    )
    str_output: str = redaction_result.str_sanitized_text
    assert EXPECTED_SSN_LAST4_TOKEN in str_output
    assert EXPECTED_REDACTED_SSN_TOKEN not in str_output
    assert redaction_result.list_str_detected_categories == []


def test_allowed_entities_ssn_passes_through_contextual_ssn_last4() -> None:
    """
    Canonical SSN allow-list pass-through also bypasses contextual SSN last-4 redaction.
    """
    middleware_settings: PiiRedactionSettingsModel = (
        _build_redactor_middleware_settings(
            DETECTION_PROFILE_BALANCED,
            {"allowed_entities": [MATRIX_ENTITY_SSN]},
        )
    )
    redactor: BaseRedactorLayer = _create_redactor(middleware_settings)
    redaction_result: RedactionResult = redactor.sanitize_with_result(
        TEXT_CONTEXTUAL_SSN_LAST4
    )
    str_output: str = redaction_result.str_sanitized_text
    assert EXPECTED_SSN_LAST4_TOKEN in str_output
    assert EXPECTED_REDACTED_SSN_TOKEN not in str_output
    assert redaction_result.list_str_detected_categories == []


def test_redaction_detects_contextual_cc_last4(
    redactor_small: BaseRedactorLayer,
) -> None:
    """
    Balanced profile redacts card last-4 candidates only when local card context is present.
    """
    redaction_result: RedactionResult = redactor_small.sanitize_with_result(
        TEXT_CONTEXTUAL_CC_LAST4
    )
    str_output: str = redaction_result.str_sanitized_text
    assert EXPECTED_CC_LAST4_TOKEN not in str_output
    assert EXPECTED_REDACTED_CC_LAST4_TOKEN in str_output
    assert MATRIX_ENTITY_CC_LAST4 in redaction_result.list_str_detected_categories


def test_redaction_detects_short_contextual_cc_last4_without_address_override(
    redactor_small: BaseRedactorLayer,
) -> None:
    """
    Balanced profile keeps short card-tail phrasing classified as CC last-4, not ADDRESS.
    """
    redaction_result: RedactionResult = redactor_small.sanitize_with_result(
        TEXT_SHORT_CONTEXTUAL_CC_LAST4
    )
    str_output: str = redaction_result.str_sanitized_text
    assert EXPECTED_CC_LAST4_TOKEN not in str_output
    assert EXPECTED_REDACTED_CC_LAST4_TOKEN in str_output
    assert "[REDACTED:ADDRESS]" not in str_output
    assert MATRIX_ENTITY_CC_LAST4 in redaction_result.list_str_detected_categories
    assert "ADDRESS" not in redaction_result.list_str_detected_categories


def test_allowed_entities_cc_last4_passes_through_contextual_card_last4() -> None:
    """
    Canonical card-tail allow-list pass-through bypasses contextual CC last-4 redaction.
    """
    middleware_settings: PiiRedactionSettingsModel = (
        _build_redactor_middleware_settings(
            DETECTION_PROFILE_BALANCED,
            {"allowed_entities": [MATRIX_ENTITY_CC_LAST4]},
        )
    )
    redactor: BaseRedactorLayer = _create_redactor(middleware_settings)
    redaction_result: RedactionResult = redactor.sanitize_with_result(
        TEXT_CONTEXTUAL_CC_LAST4
    )
    str_output: str = redaction_result.str_sanitized_text
    assert EXPECTED_CC_LAST4_TOKEN in str_output
    assert EXPECTED_REDACTED_CC_LAST4_TOKEN not in str_output
    assert redaction_result.list_str_detected_categories == []


def test_allowed_entities_cc_last4_passes_through_short_contextual_card_last4() -> None:
    """
    Canonical card-tail allow-list also prevents parser-backed ADDRESS override on short phrasing.
    """
    middleware_settings: PiiRedactionSettingsModel = (
        _build_redactor_middleware_settings(
            DETECTION_PROFILE_BALANCED,
            {"allowed_entities": [MATRIX_ENTITY_CC_LAST4]},
        )
    )
    redactor: BaseRedactorLayer = _create_redactor(middleware_settings)
    redaction_result: RedactionResult = redactor.sanitize_with_result(
        TEXT_SHORT_CONTEXTUAL_CC_LAST4
    )
    str_output: str = redaction_result.str_sanitized_text
    assert EXPECTED_CC_LAST4_TOKEN in str_output
    assert EXPECTED_REDACTED_CC_LAST4_TOKEN not in str_output
    assert "[REDACTED:ADDRESS]" not in str_output
    assert redaction_result.list_str_detected_categories == []


def test_entity_label_map_cc_last4_rewrites_output_label() -> None:
    """
    Balanced profile supports canonical CC last-4 mapping overrides.
    """
    middleware_settings: PiiRedactionSettingsModel = (
        _build_redactor_middleware_settings(
            DETECTION_PROFILE_BALANCED,
            {"entity_label_map": {MATRIX_ENTITY_CC_LAST4: "CARD_TAIL"}},
        )
    )
    redactor: BaseRedactorLayer = _create_redactor(middleware_settings)
    redaction_result: RedactionResult = redactor.sanitize_with_result(
        TEXT_CONTEXTUAL_CC_LAST4
    )
    str_output: str = redaction_result.str_sanitized_text
    assert EXPECTED_CC_LAST4_TOKEN not in str_output
    assert EXPECTED_CUSTOM_CC_LAST4_TOKEN in str_output


def test_redaction_detects_single_term_configured_cc_last4_without_address_override() -> (
    None
):
    """
    Balanced profile honors single-term CC context configuration and still avoids ADDRESS override.
    """
    middleware_settings: PiiRedactionSettingsModel = (
        _build_redactor_middleware_settings(
            DETECTION_PROFILE_BALANCED,
            {
                "redaction_recognizers": {
                    "cc_last4": {
                        "context_terms": ["visa"],
                    }
                }
            },
        )
    )
    redactor: BaseRedactorLayer = _create_redactor(middleware_settings)
    redaction_result: RedactionResult = redactor.sanitize_with_result(
        TEXT_SINGLE_TERM_CONTEXTUAL_CC_LAST4
    )
    str_output: str = redaction_result.str_sanitized_text
    assert EXPECTED_CC_LAST4_TOKEN not in str_output
    assert EXPECTED_REDACTED_CC_LAST4_TOKEN in str_output
    assert "[REDACTED:ADDRESS]" not in str_output
    assert MATRIX_ENTITY_CC_LAST4 in redaction_result.list_str_detected_categories
    assert "ADDRESS" not in redaction_result.list_str_detected_categories


def test_redaction_detects_social_security_card_last4_without_address_override(
    redactor_small: BaseRedactorLayer,
) -> None:
    """
    Balanced profile keeps social-security-card phrasing classified as SSN, not ADDRESS.
    """
    redaction_result: RedactionResult = redactor_small.sanitize_with_result(
        TEXT_SOCIAL_SECURITY_CARD_LAST4
    )
    str_output: str = redaction_result.str_sanitized_text
    assert EXPECTED_SSN_LAST4_TOKEN not in str_output
    assert EXPECTED_REDACTED_SSN_TOKEN in str_output
    assert "[REDACTED:ADDRESS]" not in str_output
    assert MATRIX_ENTITY_SSN in redaction_result.list_str_detected_categories
    assert "ADDRESS" not in redaction_result.list_str_detected_categories


def test_redaction_detects_contextual_dob_iso(
    redactor_small: BaseRedactorLayer,
) -> None:
    """
    Balanced profile redacts DOB dates when birth-date context is present.
    """
    redaction_result: RedactionResult = redactor_small.sanitize_with_result(
        TEXT_CONTEXTUAL_DOB_ISO
    )
    str_output: str = redaction_result.str_sanitized_text
    assert EXPECTED_DOB_TOKEN_ISO not in str_output
    assert EXPECTED_REDACTED_DOB_TOKEN in str_output
    assert MATRIX_ENTITY_DOB in redaction_result.list_str_detected_categories


def test_redaction_detects_contextual_dob_dmy(
    redactor_small: BaseRedactorLayer,
) -> None:
    """
    Balanced profile redacts DD-MM-YYYY DOB dates when birth-date context is present.
    """
    redaction_result: RedactionResult = redactor_small.sanitize_with_result(
        TEXT_CONTEXTUAL_DOB_DMY
    )
    str_output: str = redaction_result.str_sanitized_text
    assert EXPECTED_DOB_TOKEN_DMY not in str_output
    assert EXPECTED_REDACTED_DOB_TOKEN in str_output
    assert MATRIX_ENTITY_DOB in redaction_result.list_str_detected_categories


def test_redaction_detects_contextual_dob_month_name(
    redactor_small: BaseRedactorLayer,
) -> None:
    """
    Balanced profile redacts month-name DOB dates when birth-date context is present.
    """
    redaction_result: RedactionResult = redactor_small.sanitize_with_result(
        TEXT_CONTEXTUAL_DOB_MONTH_NAME
    )
    str_output: str = redaction_result.str_sanitized_text
    assert EXPECTED_DOB_TOKEN_MONTH_NAME not in str_output
    assert EXPECTED_REDACTED_DOB_TOKEN in str_output
    assert MATRIX_ENTITY_DOB in redaction_result.list_str_detected_categories


def test_redaction_ignores_non_dob_date_context(
    redactor_small: BaseRedactorLayer,
) -> None:
    """
    Balanced profile leaves generic business dates unchanged when DOB context is absent.
    """
    redaction_result: RedactionResult = redactor_small.sanitize_with_result(
        TEXT_NEGATIVE_DOB_INVOICE_DATE
    )
    str_output: str = redaction_result.str_sanitized_text
    assert EXPECTED_DOB_TOKEN_ISO in str_output
    assert EXPECTED_REDACTED_DOB_TOKEN not in str_output
    assert redaction_result.list_str_detected_categories == []


def test_allowed_entities_dob_passes_through_contextual_dob() -> None:
    """
    Canonical DOB allow-list pass-through bypasses contextual DOB redaction.
    """
    middleware_settings: PiiRedactionSettingsModel = (
        _build_redactor_middleware_settings(
            DETECTION_PROFILE_BALANCED,
            {"allowed_entities": [MATRIX_ENTITY_DOB]},
        )
    )
    redactor: BaseRedactorLayer = _create_redactor(middleware_settings)
    redaction_result: RedactionResult = redactor.sanitize_with_result(
        TEXT_CONTEXTUAL_DOB_ISO
    )
    str_output: str = redaction_result.str_sanitized_text
    assert EXPECTED_DOB_TOKEN_ISO in str_output
    assert EXPECTED_REDACTED_DOB_TOKEN not in str_output
    assert redaction_result.list_str_detected_categories == []


def test_entity_label_map_dob_rewrites_output_label() -> None:
    """
    Balanced profile supports canonical DOB label overrides.
    """
    middleware_settings: PiiRedactionSettingsModel = (
        _build_redactor_middleware_settings(
            DETECTION_PROFILE_BALANCED,
            {"entity_label_map": {MATRIX_ENTITY_DOB: "BIRTH_DATE"}},
        )
    )
    redactor: BaseRedactorLayer = _create_redactor(middleware_settings)
    redaction_result: RedactionResult = redactor.sanitize_with_result(
        TEXT_CONTEXTUAL_DOB_ISO
    )
    str_output: str = redaction_result.str_sanitized_text
    assert EXPECTED_DOB_TOKEN_ISO not in str_output
    assert EXPECTED_CUSTOM_DOB_TOKEN in str_output


def test_redaction_mixed_pii_complex_punctuation(
    redactor_regex: BaseRedactorLayer,
) -> None:
    """
    Regex profile handles mixed punctuation around deterministic PII patterns.
    """
    str_email: str = secrets.choice(LIST_STR_EMAILS)
    str_ssn: str = secrets.choice(LIST_STR_SSNS)

    str_input: str = f"Please contact ({str_email}) regarding ID: {str_ssn} for #12345"
    str_output: str = redactor_regex.sanitize_with_result(str_input).str_sanitized_text
    assert str_email not in str_output
    assert "[REDACTED:EMAIL]" in str_output
    assert f"ID: {str_ssn}" not in str_output
    assert "[REDACTED:SSN]" in str_output


def test_redaction_balanced_profile_custom_name_mapping() -> None:
    """
    Balanced detection profile supports canonical NAME mapping overrides.
    """
    str_name: str = secrets.choice(LIST_STR_NAMES)
    middleware_settings: PiiRedactionSettingsModel = (
        _build_redactor_middleware_settings(
            DETECTION_PROFILE_BALANCED,
            {"entity_label_map": {"NAME": "CUSTOMER_NAME"}},
        )
    )
    redactor: BaseRedactorLayer = _create_redactor(middleware_settings)
    str_input: str = f"I am speaking with {str_name} right now."
    str_output: str = redactor.sanitize_with_result(str_input).str_sanitized_text
    assert str_name not in str_output
    assert "[REDACTED:CUSTOMER_NAME]" in str_output


@pytest.mark.parametrize(
    "dict_case",
    LIST_DICT_REGEX_CONFIG_VARIATIONS,
    ids=[dict_case["str_case_id"] for dict_case in LIST_DICT_REGEX_CONFIG_VARIATIONS],
)
def test_regex_config_variation_matrix(dict_case: dict[str, Any]) -> None:
    """
    Validates regex-profile configuration variations against deterministic PII input.

    Args:
        dict_case: A single variation case including settings overrides and expected output tokens.

    Returns:
        None
    """
    middleware_settings: PiiRedactionSettingsModel = (
        _build_redactor_middleware_settings(
            DETECTION_PROFILE_LOW_MEMORY,
            dict_case["dict_setting_overrides"],
        )
    )
    redactor: BaseRedactorLayer = _create_redactor(middleware_settings)
    str_input: str = f"Phone={MATRIX_PHONE}; SSN={MATRIX_SSN}."
    str_output: str = redactor.sanitize_with_result(str_input).str_sanitized_text

    # Loop through required tokens to verify each expected marker/raw value exists in output.
    for str_expected_contains in dict_case["list_str_expected_contains"]:
        assert str_expected_contains in str_output

    # Loop through forbidden tokens to verify each disallowed marker/raw value is absent.
    for str_expected_not_contains in dict_case["list_str_expected_not_contains"]:
        assert str_expected_not_contains not in str_output


def test_redaction_result_contract_exposes_canonical_categories(
    redactor_regex: BaseRedactorLayer,
) -> None:
    """
    Validates the new generic redaction result contract category fields.

    Args:
        redactor_regex: Runtime-ready regex profile redactor fixture.

    Returns:
        None
    """
    str_input: str = (
        "Reach me at olivia.carter@example.com, +1 (415) 555-3007, SSN 585-39-6731"
    )
    redaction_result: RedactionResult = redactor_regex.sanitize_with_result(str_input)

    assert "EMAIL" in redaction_result.list_str_detected_categories
    assert "PHONE" in redaction_result.list_str_detected_categories
    assert "SSN" in redaction_result.list_str_detected_categories
    assert redaction_result.int_redaction_count == 3


def test_redaction_balanced_profile_detects_all_supported_entities_in_mixed_record(
    redactor_small: BaseRedactorLayer,
) -> None:
    """
    Ensures a mixed balanced-profile record redacts all supported entity families together.

    Args:
        redactor_small: Runtime-ready balanced-profile redactor fixture.

    Returns:
        None for normal test completion.
    """
    redaction_result: RedactionResult = redactor_small.sanitize_with_result(
        TEXT_MIXED_RECORD_ALL_PII
    )
    str_output: str = redaction_result.str_sanitized_text

    assert "Claire Foster" not in str_output
    assert "1992-06-18" not in str_output
    assert "(646) 555-0121" not in str_output
    assert "claire.foster@demo-company.net" not in str_output
    assert "1129" not in str_output
    assert "5542" not in str_output
    assert "[REDACTED:NAME]" in str_output
    assert EXPECTED_REDACTED_DOB_TOKEN in str_output
    assert "[REDACTED:PHONE]" in str_output
    assert "[REDACTED:EMAIL]" in str_output
    assert EXPECTED_REDACTED_CC_LAST4_TOKEN in str_output
    assert EXPECTED_REDACTED_SSN_TOKEN in str_output
    assert "ADDRESS" not in redaction_result.list_str_detected_categories
    assert "NAME" in redaction_result.list_str_detected_categories
    assert MATRIX_ENTITY_DOB in redaction_result.list_str_detected_categories
    assert MATRIX_ENTITY_PHONE in redaction_result.list_str_detected_categories
    assert "EMAIL" in redaction_result.list_str_detected_categories
    assert MATRIX_ENTITY_CC_LAST4 in redaction_result.list_str_detected_categories
    assert MATRIX_ENTITY_SSN in redaction_result.list_str_detected_categories
    assert redaction_result.int_redaction_count == EXPECTED_MIXED_RECORD_REDACTION_COUNT


def test_redaction_balanced_profile_redacts_full_address_after_structured_field_labels(
    redactor_small: BaseRedactorLayer,
) -> None:
    """
    Ensures parser-backed address redaction stays aligned to the real address in mixed records.

    Args:
        redactor_small: Runtime-ready balanced-profile redactor fixture.

    Returns:
        None for normal test completion.
    """
    redaction_result: RedactionResult = redactor_small.sanitize_with_result(
        TEXT_MIXED_RECORD_WITH_USADDRESS_FIELD_LABELS
    )
    str_output: str = redaction_result.str_sanitized_text

    assert "Thomas Nguyen" not in str_output
    assert "2000-01-29" not in str_output
    assert "4418" not in str_output
    assert "thomas.nguyen@qa-mail.dev" not in str_output
    assert EXPECTED_USADDRESS_FIELD_LABEL_STREET_FRAGMENT not in str_output
    assert EXPECTED_USADDRESS_FIELD_LABEL_CITY_FRAGMENT not in str_output
    assert EXPECTED_USADDRESS_FIELD_LABEL_STATE_ZIP_FRAGMENT not in str_output
    assert "[REDACTED:NAME]" in str_output
    assert EXPECTED_REDACTED_DOB_TOKEN in str_output
    assert EXPECTED_REDACTED_SSN_TOKEN in str_output
    assert "[REDACTED:EMAIL]" in str_output
    assert f"address {EXPECTED_REDACTED_ADDRESS_TOKEN}" in str_output
    assert MATRIX_ENTITY_DOB in redaction_result.list_str_detected_categories
    assert MATRIX_ENTITY_SSN in redaction_result.list_str_detected_categories
    assert "EMAIL" in redaction_result.list_str_detected_categories
    assert "ADDRESS" in redaction_result.list_str_detected_categories
    assert (
        redaction_result.int_redaction_count
        == EXPECTED_USADDRESS_FIELD_LABEL_REDACTION_COUNT
    )
