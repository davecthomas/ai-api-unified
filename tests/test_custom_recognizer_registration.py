# ruff: noqa: E402

from __future__ import annotations

import pytest

pytest.importorskip("presidio_analyzer")

from presidio_analyzer import AnalyzerEngine, RecognizerResult

from ai_api_unified.middleware.impl._presidio_redactor import (
    PROFILE_SMALL,
    PiiRedactor,
    get_cached_analyzer_engine,
)
from ai_api_unified.middleware.impl.custom_recognizer_factory import (
    DICT_STR_RECOGNIZER_RULE_NAME_TO_RECOGNIZER_NAME,
    PROVIDER_ENTITY_CREDIT_CARD_LAST4,
    PROVIDER_ENTITY_DOB_DATE,
    PROVIDER_ENTITY_US_SSN_LAST4,
    RecognizerRegistrationConfig,
    RECOGNIZER_RULE_KEY_CC_LAST4,
    RECOGNIZER_RULE_KEY_DOB,
    RECOGNIZER_RULE_KEY_SSN_LAST4,
)

TEST_LANGUAGE: str = "en"
TEST_SIGNATURE_ENABLED: str = "phase_c_custom_recognizers_enabled"
TEST_SIGNATURE_DISABLED: str = "phase_c_custom_recognizers_disabled"
TEST_SIGNATURE_SSN_POSITIVE_MATCH: str = "phase_d_ssn_last4_positive_match"
TEST_SIGNATURE_SSN_NEGATIVE_MATCH: str = "phase_d_ssn_last4_negative_match"
TEST_SIGNATURE_CC_POSITIVE_MATCH: str = "phase_e_cc_last4_positive_match"
TEST_SIGNATURE_CC_NEGATIVE_MATCH: str = "phase_e_cc_last4_negative_match"
TEST_SIGNATURE_DOB_POSITIVE_MATCH: str = "phase_f_dob_positive_match"
TEST_SIGNATURE_DOB_NEGATIVE_MATCH: str = "phase_f_dob_negative_match"
TUPLE_STR_ENABLED_RULE_KEYS: tuple[str, ...] = (
    RECOGNIZER_RULE_KEY_SSN_LAST4,
    RECOGNIZER_RULE_KEY_CC_LAST4,
    RECOGNIZER_RULE_KEY_DOB,
)
TUPLE_STR_SSN_ENABLED_RULE_KEY: tuple[str, ...] = (RECOGNIZER_RULE_KEY_SSN_LAST4,)
TUPLE_STR_CC_ENABLED_RULE_KEY: tuple[str, ...] = (RECOGNIZER_RULE_KEY_CC_LAST4,)
TUPLE_STR_DOB_ENABLED_RULE_KEY: tuple[str, ...] = (RECOGNIZER_RULE_KEY_DOB,)
TEST_PROXIMITY_WINDOW_CHARS: int = 28
TEST_CONFIDENCE_THRESHOLD_SSN: float = 0.75
TEST_CONFIDENCE_THRESHOLD_DEFAULT: float = 0.8
TEST_CONFIDENCE_THRESHOLD_ZERO: float = 0.0
TEST_CONFIDENCE_THRESHOLD_CC: float = 0.8
TEST_CONFIDENCE_THRESHOLD_DOB: float = 0.8
TUPLE_STR_SSN_CONTEXT_TERMS: tuple[str, ...] = (
    "ssn",
    "social security",
)
TUPLE_STR_SSN_NEGATIVE_CONTEXT_TERMS: tuple[str, ...] = (
    "cc",
    "credit card",
    "debit",
    "visa",
    "mastercard",
    "master card",
    "amex",
    "american express",
    "discover",
)
TUPLE_STR_EMPTY_CONTEXT_TERMS: tuple[str, ...] = ()
TUPLE_STR_CC_CONTEXT_TERMS: tuple[str, ...] = (
    "cc",
    "card",
    "credit card",
    "debit",
    "visa",
    "mastercard",
    "master card",
    "amex",
    "american express",
    "discover",
    "last 4",
    "ending in",
)
TUPLE_STR_CC_NEGATIVE_CONTEXT_TERMS: tuple[str, ...] = (
    "ssn",
    "social security",
    "expires",
    "expiration",
    "expiry",
    "exp",
)
TUPLE_STR_DOB_CONTEXT_TERMS: tuple[str, ...] = (
    "dob",
    "date of birth",
    "born",
    "birth date",
)
TUPLE_STR_DOB_CONTEXT_TERMS_SINGLE_BORN: tuple[str, ...] = ("born",)
TUPLE_STR_DOB_NEGATIVE_CONTEXT_TERMS: tuple[str, ...] = (
    "invoice date",
    "appointment date",
    "due date",
)
TEXT_CONTEXTUAL_SSN_LAST4: str = "The customer SSN ending in 6789 was verified."
TEXT_NONCONTEXTUAL_FOUR_DIGITS: str = "Order number 6789 has shipped."
TEXT_CC_CONTEXTUAL_LAST4: str = "Visa ending in 4242 is on file."
TEXT_CARD_STYLE_LAST4: str = "Card ending in 4242 is on file."
TEXT_VISA_ONLY_CONTEXTUAL_CC_LAST4: str = "Visa 4242 is on file."
TEXT_ENDING_IN_ONLY_CONTEXTUAL_CC_LAST4: str = "Please verify ending in 4242."
TEXT_MASKED_CC_CONTEXTUAL_LAST4: str = (
    "The stored Visa credit card is xxxx-4242 for billing."
)
TEXT_CC_NONCONTEXTUAL_YEAR: str = "The card expires in 2024."
TEXT_SOCIAL_SECURITY_CARD_LAST4: str = (
    "Social security card ending in 6789 was verified."
)
TEXT_CONTEXTUAL_SSN_LAST4_WITH_SUCCESS_WORD: str = (
    "The customer SSN ending in 6789 was a success."
)
TEXT_CONTEXTUAL_DOB_ISO: str = "DOB 1991-12-31 was verified."
TEXT_CONTEXTUAL_DOB_SLASH: str = "Date of birth: 01/22/1988."
TEXT_CONTEXTUAL_DOB_DMY: str = "born 31-12-1991 in Boston."
TEXT_CONTEXTUAL_DOB_MONTH_NAME: str = "Date of birth: January 22, 1988."
TEXT_NEGATIVE_DOB_INVOICE_DATE: str = "Invoice date 1991-12-31 was archived."
EXPECTED_CONTEXTUAL_SSN_TOKEN: str = "6789"
EXPECTED_CONTEXTUAL_CC_TOKEN: str = "4242"
EXPECTED_CONTEXTUAL_DOB_ISO_TOKEN: str = "1991-12-31"
EXPECTED_CONTEXTUAL_DOB_SLASH_TOKEN: str = "01/22/1988"
EXPECTED_CONTEXTUAL_DOB_DMY_TOKEN: str = "31-12-1991"
EXPECTED_CONTEXTUAL_DOB_MONTH_NAME_TOKEN: str = "January 22, 1988"
TEXT_CONTEXTUAL_SSN_LAST4_ZERO_THRESHOLD: str = "SSN ending in 1122 was verified."
TEXT_CONTEXTUAL_SSN_LAST4_HIGH_THRESHOLD: str = "SSN ending in 3344 was verified."
TEXT_CONTEXTUAL_CC_LAST4_HIGH_THRESHOLD: str = "Visa ending in 4455 is on file."
TEXT_CONTEXTUAL_DOB_HIGH_THRESHOLD: str = "DOB 1991-12-31 was verified."
TEST_PROXIMITY_WINDOW_CHARS_RAW: str = "44"
TEST_CONFIDENCE_THRESHOLD_RAW: str = "0.66"
TEST_CONFIDENCE_THRESHOLD_NORMALIZED: float = 0.66
TEST_PROXIMITY_WINDOW_CHARS_NORMALIZED: int = 44
TEST_CONTEXT_TERM_DIRECT_INPUT: str = "ssn ending"
TEST_CONFIDENCE_THRESHOLD_ONE: float = 1.0
TUPLE_STR_CC_CONTEXT_TERMS_SINGLE_VISA: tuple[str, ...] = ("visa",)
TUPLE_STR_CC_CONTEXT_TERMS_SINGLE_ENDING_IN: tuple[str, ...] = ("ending in",)
CONFIG_REDACTION_RECOGNIZERS_FIELD: str = "redaction_recognizers"
CONFIG_ENABLED_FIELD: str = "enabled"
CONFIG_CONTEXT_TERMS_FIELD: str = "context_terms"


def _build_registration_configs(
    tuple_str_enabled_rule_names: tuple[str, ...],
    float_ssn_confidence_threshold: float = TEST_CONFIDENCE_THRESHOLD_SSN,
    float_cc_confidence_threshold: float = TEST_CONFIDENCE_THRESHOLD_CC,
    tuple_str_cc_context_terms: tuple[str, ...] = TUPLE_STR_CC_CONTEXT_TERMS,
    float_dob_confidence_threshold: float = TEST_CONFIDENCE_THRESHOLD_DOB,
    tuple_str_dob_context_terms: tuple[str, ...] = TUPLE_STR_DOB_CONTEXT_TERMS,
) -> tuple[RecognizerRegistrationConfig, ...]:
    """
    Builds hashable runtime recognizer registration configs for tests.

    Args:
        tuple_str_enabled_rule_names: Ordered tuple of enabled recognizer rule names.
            float_ssn_confidence_threshold: SSN threshold override for targeted tests.
            float_cc_confidence_threshold: CC threshold override for targeted tests.
            tuple_str_cc_context_terms: CC positive context terms used for targeted tests.
            float_dob_confidence_threshold: DOB threshold override for targeted tests.
            tuple_str_dob_context_terms: DOB positive context terms used for targeted tests.

    Returns:
        Ordered tuple of RecognizerRegistrationConfig records.
    """
    list_recognizer_registration_configs: list[RecognizerRegistrationConfig] = []
    # Loop through enabled rule names and build one config payload per rule.
    for str_enabled_rule_name in tuple_str_enabled_rule_names:
        float_confidence_threshold: float = TEST_CONFIDENCE_THRESHOLD_DEFAULT
        tuple_str_context_terms: tuple[str, ...] = TUPLE_STR_EMPTY_CONTEXT_TERMS
        tuple_str_negative_context_terms: tuple[str, ...] = (
            TUPLE_STR_EMPTY_CONTEXT_TERMS
        )
        if str_enabled_rule_name == RECOGNIZER_RULE_KEY_SSN_LAST4:
            float_confidence_threshold = float_ssn_confidence_threshold
            tuple_str_context_terms = TUPLE_STR_SSN_CONTEXT_TERMS
            tuple_str_negative_context_terms = TUPLE_STR_SSN_NEGATIVE_CONTEXT_TERMS
        if str_enabled_rule_name == RECOGNIZER_RULE_KEY_CC_LAST4:
            float_confidence_threshold = float_cc_confidence_threshold
            tuple_str_context_terms = tuple_str_cc_context_terms
            tuple_str_negative_context_terms = TUPLE_STR_CC_NEGATIVE_CONTEXT_TERMS
        if str_enabled_rule_name == RECOGNIZER_RULE_KEY_DOB:
            float_confidence_threshold = float_dob_confidence_threshold
            tuple_str_context_terms = tuple_str_dob_context_terms
            tuple_str_negative_context_terms = TUPLE_STR_DOB_NEGATIVE_CONTEXT_TERMS
        recognizer_registration_config: RecognizerRegistrationConfig = (
            RecognizerRegistrationConfig(
                str_recognizer_rule_name=str_enabled_rule_name,
                float_confidence_threshold=float_confidence_threshold,
                int_proximity_window_chars=TEST_PROXIMITY_WINDOW_CHARS,
                int_min_positive_context_matches=1,
                tuple_str_context_terms=tuple_str_context_terms,
                tuple_str_negative_context_terms=tuple_str_negative_context_terms,
            )
        )
        list_recognizer_registration_configs.append(recognizer_registration_config)
    tuple_recognizer_registration_configs: tuple[RecognizerRegistrationConfig, ...] = (
        tuple(list_recognizer_registration_configs)
    )
    # Normal return with recognizer registration configs for cache-builder tests.
    return tuple_recognizer_registration_configs


def _get_recognizer_name_set(analyzer_engine: AnalyzerEngine) -> set[str]:
    """
    Collects recognizer names from an analyzer engine registry.

    Args:
        analyzer_engine: Analyzer engine instance whose registry should be inspected.

    Returns:
        Set of recognizer names currently registered in the engine registry.
    """
    set_str_recognizer_names: set[str] = {
        recognizer.name for recognizer in analyzer_engine.registry.recognizers
    }
    # Normal return with recognizer names from the registry snapshot.
    return set_str_recognizer_names


def test_get_cached_analyzer_engine_registers_custom_recognizers_when_enabled() -> None:
    """
    Ensures cache-built analyzer engines register custom recognizers for enabled rules.

    Args:
        None

    Returns:
        None
    """
    get_cached_analyzer_engine.cache_clear()
    analyzer_engine: AnalyzerEngine = get_cached_analyzer_engine(
        str_language=TEST_LANGUAGE,
        str_profile=PROFILE_SMALL,
        str_engine_cache_identity_signature=TEST_SIGNATURE_ENABLED,
        tuple_custom_recognizer_registration_configs=_build_registration_configs(
            TUPLE_STR_ENABLED_RULE_KEYS
        ),
    )
    set_str_recognizer_names: set[str] = _get_recognizer_name_set(analyzer_engine)
    set_str_expected_custom_recognizer_names: set[str] = {
        DICT_STR_RECOGNIZER_RULE_NAME_TO_RECOGNIZER_NAME[str_rule_key]
        for str_rule_key in TUPLE_STR_ENABLED_RULE_KEYS
    }
    assert set_str_expected_custom_recognizer_names.issubset(set_str_recognizer_names)
    # Normal return on successful assertions for enabled custom recognizer registration.
    return None


def test_get_cached_analyzer_engine_does_not_register_custom_recognizers_when_disabled() -> (
    None
):
    """
    Ensures analyzer engines skip custom recognizer registration when no rules are enabled.

    Args:
        None

    Returns:
        None
    """
    get_cached_analyzer_engine.cache_clear()
    analyzer_engine: AnalyzerEngine = get_cached_analyzer_engine(
        str_language=TEST_LANGUAGE,
        str_profile=PROFILE_SMALL,
        str_engine_cache_identity_signature=TEST_SIGNATURE_DISABLED,
        tuple_custom_recognizer_registration_configs=(),
    )
    set_str_recognizer_names: set[str] = _get_recognizer_name_set(analyzer_engine)
    set_str_expected_custom_recognizer_names: set[str] = {
        DICT_STR_RECOGNIZER_RULE_NAME_TO_RECOGNIZER_NAME[str_rule_key]
        for str_rule_key in TUPLE_STR_ENABLED_RULE_KEYS
    }
    assert set_str_expected_custom_recognizer_names.isdisjoint(set_str_recognizer_names)
    # Normal return on successful assertions for disabled custom recognizer registration.
    return None


def test_ssn_last4_custom_recognizer_detects_contextual_spans() -> None:
    """
    Ensures Phase-D SSN last-4 recognizer detects spans when SSN context is present.

    Args:
        None

    Returns:
        None
    """
    get_cached_analyzer_engine.cache_clear()
    analyzer_engine: AnalyzerEngine = get_cached_analyzer_engine(
        str_language=TEST_LANGUAGE,
        str_profile=PROFILE_SMALL,
        str_engine_cache_identity_signature=TEST_SIGNATURE_SSN_POSITIVE_MATCH,
        tuple_custom_recognizer_registration_configs=_build_registration_configs(
            TUPLE_STR_SSN_ENABLED_RULE_KEY
        ),
    )
    list_recognizer_results: list[RecognizerResult] = analyzer_engine.analyze(
        text=TEXT_CONTEXTUAL_SSN_LAST4,
        entities=[PROVIDER_ENTITY_US_SSN_LAST4],
        language=TEST_LANGUAGE,
    )
    assert len(list_recognizer_results) == 1
    recognizer_result: RecognizerResult = list_recognizer_results[0]
    assert (
        TEXT_CONTEXTUAL_SSN_LAST4[recognizer_result.start : recognizer_result.end]
        == EXPECTED_CONTEXTUAL_SSN_TOKEN
    )
    # Normal return on successful assertions for SSN contextual detection.
    return None


def test_ssn_last4_custom_recognizer_ignores_noncontextual_spans() -> None:
    """
    Ensures Phase-D SSN last-4 recognizer ignores 4-digit spans without SSN context.

    Args:
        None

    Returns:
        None
    """
    get_cached_analyzer_engine.cache_clear()
    analyzer_engine: AnalyzerEngine = get_cached_analyzer_engine(
        str_language=TEST_LANGUAGE,
        str_profile=PROFILE_SMALL,
        str_engine_cache_identity_signature=TEST_SIGNATURE_SSN_NEGATIVE_MATCH,
        tuple_custom_recognizer_registration_configs=_build_registration_configs(
            TUPLE_STR_SSN_ENABLED_RULE_KEY
        ),
    )
    list_recognizer_results: list[RecognizerResult] = analyzer_engine.analyze(
        text=TEXT_NONCONTEXTUAL_FOUR_DIGITS,
        entities=[PROVIDER_ENTITY_US_SSN_LAST4],
        language=TEST_LANGUAGE,
    )
    assert list_recognizer_results == []
    # Normal return on successful assertions for non-context SSN suppression.
    return None


def test_ssn_last4_custom_recognizer_ignores_card_style_context() -> None:
    """
    Ensures SSN last-4 recognizer rejects card-style phrases that use generic last-4 wording.

    Args:
        None

    Returns:
        None
    """
    get_cached_analyzer_engine.cache_clear()
    analyzer_engine: AnalyzerEngine = get_cached_analyzer_engine(
        str_language=TEST_LANGUAGE,
        str_profile=PROFILE_SMALL,
        str_engine_cache_identity_signature="phase_d_ssn_last4_card_style_negative",
        tuple_custom_recognizer_registration_configs=_build_registration_configs(
            TUPLE_STR_SSN_ENABLED_RULE_KEY
        ),
    )
    list_recognizer_results: list[RecognizerResult] = analyzer_engine.analyze(
        text=TEXT_CARD_STYLE_LAST4,
        entities=[PROVIDER_ENTITY_US_SSN_LAST4],
        language=TEST_LANGUAGE,
    )
    assert list_recognizer_results == []
    # Normal return on successful assertions for card-style SSN suppression.
    return None


def test_ssn_last4_custom_recognizer_ignores_short_term_substrings_inside_words() -> (
    None
):
    """
    Ensures short negative context terms do not match inside unrelated words such as success.

    Args:
        None

    Returns:
        None
    """
    get_cached_analyzer_engine.cache_clear()
    analyzer_engine: AnalyzerEngine = get_cached_analyzer_engine(
        str_language=TEST_LANGUAGE,
        str_profile=PROFILE_SMALL,
        str_engine_cache_identity_signature="phase_d_ssn_last4_success_word_positive",
        tuple_custom_recognizer_registration_configs=_build_registration_configs(
            TUPLE_STR_SSN_ENABLED_RULE_KEY
        ),
    )
    list_recognizer_results: list[RecognizerResult] = analyzer_engine.analyze(
        text=TEXT_CONTEXTUAL_SSN_LAST4_WITH_SUCCESS_WORD,
        entities=[PROVIDER_ENTITY_US_SSN_LAST4],
        language=TEST_LANGUAGE,
    )
    assert len(list_recognizer_results) == 1
    # Normal return on successful assertions for boundary-aware context matching.
    return None


def test_ssn_last4_custom_recognizer_allows_social_security_card_phrase() -> None:
    """
    Ensures SSN last-4 detection still works for social-security phrasing containing card.

    Args:
        None

    Returns:
        None
    """
    get_cached_analyzer_engine.cache_clear()
    analyzer_engine: AnalyzerEngine = get_cached_analyzer_engine(
        str_language=TEST_LANGUAGE,
        str_profile=PROFILE_SMALL,
        str_engine_cache_identity_signature="phase_d_ssn_last4_social_security_card_positive",
        tuple_custom_recognizer_registration_configs=_build_registration_configs(
            TUPLE_STR_SSN_ENABLED_RULE_KEY
        ),
    )
    list_recognizer_results: list[RecognizerResult] = analyzer_engine.analyze(
        text=TEXT_SOCIAL_SECURITY_CARD_LAST4,
        entities=[PROVIDER_ENTITY_US_SSN_LAST4],
        language=TEST_LANGUAGE,
    )
    assert len(list_recognizer_results) == 1
    # Normal return on successful assertions for social-security card phrasing.
    return None


def test_cc_last4_custom_recognizer_detects_contextual_spans() -> None:
    """
    Ensures card last-4 recognizer detects spans when payment-card context is present.

    Args:
        None

    Returns:
        None
    """
    get_cached_analyzer_engine.cache_clear()
    analyzer_engine: AnalyzerEngine = get_cached_analyzer_engine(
        str_language=TEST_LANGUAGE,
        str_profile=PROFILE_SMALL,
        str_engine_cache_identity_signature=TEST_SIGNATURE_CC_POSITIVE_MATCH,
        tuple_custom_recognizer_registration_configs=_build_registration_configs(
            TUPLE_STR_CC_ENABLED_RULE_KEY
        ),
    )
    list_recognizer_results: list[RecognizerResult] = analyzer_engine.analyze(
        text=TEXT_CC_CONTEXTUAL_LAST4,
        entities=[PROVIDER_ENTITY_CREDIT_CARD_LAST4],
        language=TEST_LANGUAGE,
    )
    assert len(list_recognizer_results) == 1
    recognizer_result: RecognizerResult = list_recognizer_results[0]
    assert (
        TEXT_CC_CONTEXTUAL_LAST4[recognizer_result.start : recognizer_result.end]
        == EXPECTED_CONTEXTUAL_CC_TOKEN
    )
    # Normal return on successful assertions for contextual CC detection.
    return None


def test_cc_last4_custom_recognizer_detects_single_term_visa_context() -> None:
    """
    Ensures card last-4 recognizer still works when config narrows context terms to one anchor.

    Args:
        None

    Returns:
        None
    """
    get_cached_analyzer_engine.cache_clear()
    analyzer_engine: AnalyzerEngine = get_cached_analyzer_engine(
        str_language=TEST_LANGUAGE,
        str_profile=PROFILE_SMALL,
        str_engine_cache_identity_signature="phase_e_cc_last4_single_term_visa_positive",
        tuple_custom_recognizer_registration_configs=_build_registration_configs(
            TUPLE_STR_CC_ENABLED_RULE_KEY,
            tuple_str_cc_context_terms=TUPLE_STR_CC_CONTEXT_TERMS_SINGLE_VISA,
        ),
    )
    list_recognizer_results: list[RecognizerResult] = analyzer_engine.analyze(
        text=TEXT_VISA_ONLY_CONTEXTUAL_CC_LAST4,
        entities=[PROVIDER_ENTITY_CREDIT_CARD_LAST4],
        language=TEST_LANGUAGE,
    )
    assert len(list_recognizer_results) == 1
    assert (
        TEXT_VISA_ONLY_CONTEXTUAL_CC_LAST4[
            list_recognizer_results[0].start : list_recognizer_results[0].end
        ]
        == EXPECTED_CONTEXTUAL_CC_TOKEN
    )
    # Normal return on successful assertions for single-term visa context.
    return None


def test_cc_last4_custom_recognizer_detects_single_term_ending_in_context() -> None:
    """
    Ensures card last-4 recognizer still works when config keeps only `ending in`.

    Args:
        None

    Returns:
        None
    """
    get_cached_analyzer_engine.cache_clear()
    analyzer_engine: AnalyzerEngine = get_cached_analyzer_engine(
        str_language=TEST_LANGUAGE,
        str_profile=PROFILE_SMALL,
        str_engine_cache_identity_signature="phase_e_cc_last4_single_term_ending_in_positive",
        tuple_custom_recognizer_registration_configs=_build_registration_configs(
            TUPLE_STR_CC_ENABLED_RULE_KEY,
            tuple_str_cc_context_terms=TUPLE_STR_CC_CONTEXT_TERMS_SINGLE_ENDING_IN,
        ),
    )
    list_recognizer_results: list[RecognizerResult] = analyzer_engine.analyze(
        text=TEXT_ENDING_IN_ONLY_CONTEXTUAL_CC_LAST4,
        entities=[PROVIDER_ENTITY_CREDIT_CARD_LAST4],
        language=TEST_LANGUAGE,
    )
    assert len(list_recognizer_results) == 1
    assert (
        TEXT_ENDING_IN_ONLY_CONTEXTUAL_CC_LAST4[
            list_recognizer_results[0].start : list_recognizer_results[0].end
        ]
        == EXPECTED_CONTEXTUAL_CC_TOKEN
    )
    # Normal return on successful assertions for single-term ending-in context.
    return None


def test_cc_last4_custom_recognizer_detects_masked_tail_spans() -> None:
    """
    Ensures card last-4 recognizer detects tails adjacent to masking text.

    Args:
        None

    Returns:
        None
    """
    get_cached_analyzer_engine.cache_clear()
    analyzer_engine: AnalyzerEngine = get_cached_analyzer_engine(
        str_language=TEST_LANGUAGE,
        str_profile=PROFILE_SMALL,
        str_engine_cache_identity_signature="phase_e_cc_last4_masked_positive",
        tuple_custom_recognizer_registration_configs=_build_registration_configs(
            TUPLE_STR_CC_ENABLED_RULE_KEY
        ),
    )
    list_recognizer_results: list[RecognizerResult] = analyzer_engine.analyze(
        text=TEXT_MASKED_CC_CONTEXTUAL_LAST4,
        entities=[PROVIDER_ENTITY_CREDIT_CARD_LAST4],
        language=TEST_LANGUAGE,
    )
    assert len(list_recognizer_results) == 1
    assert (
        TEXT_MASKED_CC_CONTEXTUAL_LAST4[
            list_recognizer_results[0].start : list_recognizer_results[0].end
        ]
        == EXPECTED_CONTEXTUAL_CC_TOKEN
    )
    # Normal return on successful assertions for masked-tail CC detection.
    return None


def test_cc_last4_custom_recognizer_ignores_noncontextual_spans() -> None:
    """
    Ensures card last-4 recognizer ignores generic 4-digit spans without card context.

    Args:
        None

    Returns:
        None
    """
    get_cached_analyzer_engine.cache_clear()
    analyzer_engine: AnalyzerEngine = get_cached_analyzer_engine(
        str_language=TEST_LANGUAGE,
        str_profile=PROFILE_SMALL,
        str_engine_cache_identity_signature=TEST_SIGNATURE_CC_NEGATIVE_MATCH,
        tuple_custom_recognizer_registration_configs=_build_registration_configs(
            TUPLE_STR_CC_ENABLED_RULE_KEY
        ),
    )
    list_recognizer_results: list[RecognizerResult] = analyzer_engine.analyze(
        text=TEXT_NONCONTEXTUAL_FOUR_DIGITS,
        entities=[PROVIDER_ENTITY_CREDIT_CARD_LAST4],
        language=TEST_LANGUAGE,
    )
    assert list_recognizer_results == []
    # Normal return on successful assertions for non-context CC suppression.
    return None


def test_cc_last4_custom_recognizer_ignores_generic_card_year_text() -> None:
    """
    Ensures generic card-related year text does not redact an unrelated 4-digit value.

    Args:
        None

    Returns:
        None
    """
    get_cached_analyzer_engine.cache_clear()
    analyzer_engine: AnalyzerEngine = get_cached_analyzer_engine(
        str_language=TEST_LANGUAGE,
        str_profile=PROFILE_SMALL,
        str_engine_cache_identity_signature="phase_e_cc_last4_card_year_negative",
        tuple_custom_recognizer_registration_configs=_build_registration_configs(
            TUPLE_STR_CC_ENABLED_RULE_KEY
        ),
    )
    list_recognizer_results: list[RecognizerResult] = analyzer_engine.analyze(
        text=TEXT_CC_NONCONTEXTUAL_YEAR,
        entities=[PROVIDER_ENTITY_CREDIT_CARD_LAST4],
        language=TEST_LANGUAGE,
    )
    assert list_recognizer_results == []
    # Normal return on successful assertions for generic card-year suppression.
    return None


def test_cc_last4_custom_recognizer_ignores_social_security_card_phrase() -> None:
    """
    Ensures card last-4 recognizer rejects social-security phrasing containing card.

    Args:
        None

    Returns:
        None
    """
    get_cached_analyzer_engine.cache_clear()
    analyzer_engine: AnalyzerEngine = get_cached_analyzer_engine(
        str_language=TEST_LANGUAGE,
        str_profile=PROFILE_SMALL,
        str_engine_cache_identity_signature="phase_e_cc_last4_social_security_negative",
        tuple_custom_recognizer_registration_configs=_build_registration_configs(
            TUPLE_STR_CC_ENABLED_RULE_KEY
        ),
    )
    list_recognizer_results: list[RecognizerResult] = analyzer_engine.analyze(
        text=TEXT_SOCIAL_SECURITY_CARD_LAST4,
        entities=[PROVIDER_ENTITY_CREDIT_CARD_LAST4],
        language=TEST_LANGUAGE,
    )
    assert list_recognizer_results == []
    # Normal return on successful assertions for SSN-vs-CC disambiguation.
    return None


def test_ssn_last4_contextual_detection_works_with_zero_threshold() -> None:
    """
    Ensures SSN last-4 contextual detection still works when threshold is explicitly 0.0.

    Args:
        None

    Returns:
        None
    """
    get_cached_analyzer_engine.cache_clear()
    analyzer_engine: AnalyzerEngine = get_cached_analyzer_engine(
        str_language=TEST_LANGUAGE,
        str_profile=PROFILE_SMALL,
        str_engine_cache_identity_signature="phase_d_ssn_last4_zero_threshold",
        tuple_custom_recognizer_registration_configs=_build_registration_configs(
            TUPLE_STR_SSN_ENABLED_RULE_KEY,
            float_ssn_confidence_threshold=TEST_CONFIDENCE_THRESHOLD_ZERO,
        ),
    )
    list_recognizer_results: list[RecognizerResult] = analyzer_engine.analyze(
        text=TEXT_CONTEXTUAL_SSN_LAST4_ZERO_THRESHOLD,
        entities=[PROVIDER_ENTITY_US_SSN_LAST4],
        language=TEST_LANGUAGE,
    )
    assert len(list_recognizer_results) == 1
    # Normal return on successful assertions for zero-threshold contextual detection.
    return None


def test_ssn_last4_contextual_detection_works_with_max_threshold() -> None:
    """
    Ensures strict contextual SSN last-4 matches are not disabled by a threshold of 1.0.

    Args:
        None

    Returns:
        None
    """
    get_cached_analyzer_engine.cache_clear()
    analyzer_engine: AnalyzerEngine = get_cached_analyzer_engine(
        str_language=TEST_LANGUAGE,
        str_profile=PROFILE_SMALL,
        str_engine_cache_identity_signature="phase_d_ssn_last4_max_threshold",
        tuple_custom_recognizer_registration_configs=_build_registration_configs(
            TUPLE_STR_SSN_ENABLED_RULE_KEY,
            float_ssn_confidence_threshold=TEST_CONFIDENCE_THRESHOLD_ONE,
        ),
    )
    list_recognizer_results: list[RecognizerResult] = analyzer_engine.analyze(
        text=TEXT_CONTEXTUAL_SSN_LAST4_HIGH_THRESHOLD,
        entities=[PROVIDER_ENTITY_US_SSN_LAST4],
        language=TEST_LANGUAGE,
    )
    assert len(list_recognizer_results) == 1
    assert list_recognizer_results[0].score == TEST_CONFIDENCE_THRESHOLD_ONE
    # Normal return on successful assertions for max-threshold contextual detection.
    return None


def test_cc_last4_contextual_detection_works_with_max_threshold() -> None:
    """
    Ensures strict contextual card last-4 matches are not disabled by a threshold of 1.0.

    Args:
        None

    Returns:
        None
    """
    get_cached_analyzer_engine.cache_clear()
    analyzer_engine: AnalyzerEngine = get_cached_analyzer_engine(
        str_language=TEST_LANGUAGE,
        str_profile=PROFILE_SMALL,
        str_engine_cache_identity_signature="phase_e_cc_last4_max_threshold",
        tuple_custom_recognizer_registration_configs=_build_registration_configs(
            TUPLE_STR_CC_ENABLED_RULE_KEY,
            float_cc_confidence_threshold=TEST_CONFIDENCE_THRESHOLD_ONE,
        ),
    )
    list_recognizer_results: list[RecognizerResult] = analyzer_engine.analyze(
        text=TEXT_CONTEXTUAL_CC_LAST4_HIGH_THRESHOLD,
        entities=[PROVIDER_ENTITY_CREDIT_CARD_LAST4],
        language=TEST_LANGUAGE,
    )
    assert len(list_recognizer_results) == 1
    assert list_recognizer_results[0].score == TEST_CONFIDENCE_THRESHOLD_ONE
    # Normal return on successful assertions for max-threshold card detection.
    return None


def test_dob_custom_recognizer_detects_iso_contextual_span() -> None:
    """
    Ensures DOB recognizer detects ISO-style birth dates with DOB context.

    Args:
        None

    Returns:
        None
    """
    get_cached_analyzer_engine.cache_clear()
    analyzer_engine: AnalyzerEngine = get_cached_analyzer_engine(
        str_language=TEST_LANGUAGE,
        str_profile=PROFILE_SMALL,
        str_engine_cache_identity_signature=TEST_SIGNATURE_DOB_POSITIVE_MATCH,
        tuple_custom_recognizer_registration_configs=_build_registration_configs(
            TUPLE_STR_DOB_ENABLED_RULE_KEY
        ),
    )
    list_recognizer_results: list[RecognizerResult] = analyzer_engine.analyze(
        text=TEXT_CONTEXTUAL_DOB_ISO,
        entities=[PROVIDER_ENTITY_DOB_DATE],
        language=TEST_LANGUAGE,
    )
    assert len(list_recognizer_results) == 1
    assert (
        TEXT_CONTEXTUAL_DOB_ISO[
            list_recognizer_results[0].start : list_recognizer_results[0].end
        ]
        == EXPECTED_CONTEXTUAL_DOB_ISO_TOKEN
    )
    # Normal return on successful assertions for ISO DOB detection.
    return None


def test_dob_custom_recognizer_detects_slash_contextual_span() -> None:
    """
    Ensures DOB recognizer detects slash-style birth dates with DOB context.

    Args:
        None

    Returns:
        None
    """
    get_cached_analyzer_engine.cache_clear()
    analyzer_engine: AnalyzerEngine = get_cached_analyzer_engine(
        str_language=TEST_LANGUAGE,
        str_profile=PROFILE_SMALL,
        str_engine_cache_identity_signature="phase_f_dob_slash_positive",
        tuple_custom_recognizer_registration_configs=_build_registration_configs(
            TUPLE_STR_DOB_ENABLED_RULE_KEY
        ),
    )
    list_recognizer_results: list[RecognizerResult] = analyzer_engine.analyze(
        text=TEXT_CONTEXTUAL_DOB_SLASH,
        entities=[PROVIDER_ENTITY_DOB_DATE],
        language=TEST_LANGUAGE,
    )
    assert len(list_recognizer_results) == 1
    assert (
        TEXT_CONTEXTUAL_DOB_SLASH[
            list_recognizer_results[0].start : list_recognizer_results[0].end
        ]
        == EXPECTED_CONTEXTUAL_DOB_SLASH_TOKEN
    )
    # Normal return on successful assertions for slash DOB detection.
    return None


def test_dob_custom_recognizer_detects_month_name_contextual_span() -> None:
    """
    Ensures DOB recognizer detects month-name birth dates with DOB context.

    Args:
        None

    Returns:
        None
    """
    get_cached_analyzer_engine.cache_clear()
    analyzer_engine: AnalyzerEngine = get_cached_analyzer_engine(
        str_language=TEST_LANGUAGE,
        str_profile=PROFILE_SMALL,
        str_engine_cache_identity_signature="phase_f_dob_month_name_positive",
        tuple_custom_recognizer_registration_configs=_build_registration_configs(
            TUPLE_STR_DOB_ENABLED_RULE_KEY
        ),
    )
    list_recognizer_results: list[RecognizerResult] = analyzer_engine.analyze(
        text=TEXT_CONTEXTUAL_DOB_MONTH_NAME,
        entities=[PROVIDER_ENTITY_DOB_DATE],
        language=TEST_LANGUAGE,
    )
    assert len(list_recognizer_results) == 1
    assert (
        TEXT_CONTEXTUAL_DOB_MONTH_NAME[
            list_recognizer_results[0].start : list_recognizer_results[0].end
        ]
        == EXPECTED_CONTEXTUAL_DOB_MONTH_NAME_TOKEN
    )
    # Normal return on successful assertions for month-name DOB detection.
    return None


def test_dob_custom_recognizer_detects_dmy_contextual_span() -> None:
    """
    Ensures DOB recognizer detects DD-MM-YYYY birth dates with DOB context.

    Args:
        None

    Returns:
        None
    """
    get_cached_analyzer_engine.cache_clear()
    analyzer_engine: AnalyzerEngine = get_cached_analyzer_engine(
        str_language=TEST_LANGUAGE,
        str_profile=PROFILE_SMALL,
        str_engine_cache_identity_signature="phase_f_dob_dmy_positive",
        tuple_custom_recognizer_registration_configs=_build_registration_configs(
            TUPLE_STR_DOB_ENABLED_RULE_KEY,
            tuple_str_dob_context_terms=TUPLE_STR_DOB_CONTEXT_TERMS_SINGLE_BORN,
        ),
    )
    list_recognizer_results: list[RecognizerResult] = analyzer_engine.analyze(
        text=TEXT_CONTEXTUAL_DOB_DMY,
        entities=[PROVIDER_ENTITY_DOB_DATE],
        language=TEST_LANGUAGE,
    )
    assert len(list_recognizer_results) == 1
    assert (
        TEXT_CONTEXTUAL_DOB_DMY[
            list_recognizer_results[0].start : list_recognizer_results[0].end
        ]
        == EXPECTED_CONTEXTUAL_DOB_DMY_TOKEN
    )
    # Normal return on successful assertions for DD-MM-YYYY DOB detection.
    return None


def test_dob_custom_recognizer_ignores_non_dob_date_context() -> None:
    """
    Ensures DOB recognizer rejects generic business-date context near a valid date pattern.

    Args:
        None

    Returns:
        None
    """
    get_cached_analyzer_engine.cache_clear()
    analyzer_engine: AnalyzerEngine = get_cached_analyzer_engine(
        str_language=TEST_LANGUAGE,
        str_profile=PROFILE_SMALL,
        str_engine_cache_identity_signature=TEST_SIGNATURE_DOB_NEGATIVE_MATCH,
        tuple_custom_recognizer_registration_configs=_build_registration_configs(
            TUPLE_STR_DOB_ENABLED_RULE_KEY
        ),
    )
    list_recognizer_results: list[RecognizerResult] = analyzer_engine.analyze(
        text=TEXT_NEGATIVE_DOB_INVOICE_DATE,
        entities=[PROVIDER_ENTITY_DOB_DATE],
        language=TEST_LANGUAGE,
    )
    assert list_recognizer_results == []
    # Normal return on successful assertions for non-DOB date suppression.
    return None


def test_dob_contextual_detection_works_with_max_threshold() -> None:
    """
    Ensures strict contextual DOB matches are not disabled by a threshold of 1.0.

    Args:
        None

    Returns:
        None
    """
    get_cached_analyzer_engine.cache_clear()
    analyzer_engine: AnalyzerEngine = get_cached_analyzer_engine(
        str_language=TEST_LANGUAGE,
        str_profile=PROFILE_SMALL,
        str_engine_cache_identity_signature="phase_f_dob_max_threshold",
        tuple_custom_recognizer_registration_configs=_build_registration_configs(
            TUPLE_STR_DOB_ENABLED_RULE_KEY,
            float_dob_confidence_threshold=TEST_CONFIDENCE_THRESHOLD_ONE,
        ),
    )
    list_recognizer_results: list[RecognizerResult] = analyzer_engine.analyze(
        text=TEXT_CONTEXTUAL_DOB_HIGH_THRESHOLD,
        entities=[PROVIDER_ENTITY_DOB_DATE],
        language=TEST_LANGUAGE,
    )
    assert len(list_recognizer_results) == 1
    assert list_recognizer_results[0].score == TEST_CONFIDENCE_THRESHOLD_ONE
    # Normal return on successful assertions for max-threshold DOB detection.
    return None


def test_pii_redactor_applies_default_recognizer_settings_for_missing_block() -> None:
    """
    Ensures direct PiiRedactor construction uses default recognizer settings when omitted.

    Args:
        None

    Returns:
        None
    """
    redactor: PiiRedactor = PiiRedactor(dict_config={"detection_profile": "low_memory"})
    assert redactor.tuple_str_enabled_custom_recognizer_rules == (
        RECOGNIZER_RULE_KEY_SSN_LAST4,
        RECOGNIZER_RULE_KEY_CC_LAST4,
        RECOGNIZER_RULE_KEY_DOB,
    )
    dict_cc_rule_settings: dict[str, object] = dict(
        redactor.dict_redaction_recognizers_settings[RECOGNIZER_RULE_KEY_CC_LAST4]
    )
    dict_dob_rule_settings: dict[str, object] = dict(
        redactor.dict_redaction_recognizers_settings[RECOGNIZER_RULE_KEY_DOB]
    )
    assert "cc" in dict_cc_rule_settings["context_terms"]
    assert dict_cc_rule_settings["negative_context_terms"] == list(
        TUPLE_STR_CC_NEGATIVE_CONTEXT_TERMS
    )
    assert "dob" in dict_dob_rule_settings["context_terms"]
    assert dict_dob_rule_settings["negative_context_terms"] == list(
        TUPLE_STR_DOB_NEGATIVE_CONTEXT_TERMS
    )
    # Normal return on successful assertions for direct-runtime default normalization.
    return None


def test_pii_redactor_resolves_enabled_custom_recognizer_rule_keys_from_config() -> (
    None
):
    """
    Ensures PiiRedactor config wiring resolves enabled custom recognizer rules for registration.

    Args:
        None

    Returns:
        None
    """
    redactor: PiiRedactor = PiiRedactor(
        dict_config={
            "detection_profile": "low_memory",
            CONFIG_REDACTION_RECOGNIZERS_FIELD: {
                RECOGNIZER_RULE_KEY_SSN_LAST4: {CONFIG_ENABLED_FIELD: True},
                RECOGNIZER_RULE_KEY_CC_LAST4: {CONFIG_ENABLED_FIELD: False},
                RECOGNIZER_RULE_KEY_DOB: {CONFIG_ENABLED_FIELD: True},
            },
        }
    )
    assert redactor.tuple_str_enabled_custom_recognizer_rules == (
        RECOGNIZER_RULE_KEY_SSN_LAST4,
        RECOGNIZER_RULE_KEY_DOB,
    )
    tuple_str_registration_rule_names: tuple[str, ...] = tuple(
        recognizer_registration_config.str_recognizer_rule_name
        for recognizer_registration_config in redactor.tuple_custom_recognizer_registration_configs
    )
    assert tuple_str_registration_rule_names == (
        RECOGNIZER_RULE_KEY_SSN_LAST4,
        RECOGNIZER_RULE_KEY_DOB,
    )
    ssn_registration_config: RecognizerRegistrationConfig = (
        redactor.tuple_custom_recognizer_registration_configs[0]
    )
    assert ssn_registration_config.float_confidence_threshold == 0.75
    assert ssn_registration_config.int_proximity_window_chars == 28
    assert ssn_registration_config.tuple_str_context_terms == (
        "ssn",
        "social security",
    )
    assert ssn_registration_config.tuple_str_negative_context_terms == (
        "cc",
        "credit card",
        "debit",
        "visa",
        "mastercard",
        "master card",
        "amex",
        "american express",
        "discover",
    )
    # Normal return on successful assertions for config-to-runtime recognizer settings.
    return None


def test_pii_redactor_normalizes_direct_runtime_recognizer_settings_from_strings() -> (
    None
):
    """
    Ensures direct PiiRedactor construction normalizes recognizer settings like the YAML path.

    Args:
        None

    Returns:
        None
    """
    redactor: PiiRedactor = PiiRedactor(
        dict_config={
            "detection_profile": "low_memory",
            CONFIG_REDACTION_RECOGNIZERS_FIELD: {
                "proximity_window_chars": TEST_PROXIMITY_WINDOW_CHARS_RAW,
                RECOGNIZER_RULE_KEY_SSN_LAST4: {
                    CONFIG_ENABLED_FIELD: "true",
                    "confidence_threshold": TEST_CONFIDENCE_THRESHOLD_RAW,
                    "context_terms": TEST_CONTEXT_TERM_DIRECT_INPUT,
                },
            },
        }
    )
    ssn_registration_config: RecognizerRegistrationConfig = (
        redactor.tuple_custom_recognizer_registration_configs[0]
    )
    assert ssn_registration_config.float_confidence_threshold == (
        TEST_CONFIDENCE_THRESHOLD_NORMALIZED
    )
    assert (
        ssn_registration_config.int_proximity_window_chars
        == TEST_PROXIMITY_WINDOW_CHARS_NORMALIZED
    )
    assert ssn_registration_config.tuple_str_context_terms == (
        TEST_CONTEXT_TERM_DIRECT_INPUT,
    )
    # Normal return on successful assertions for direct-runtime normalization.
    return None


def test_pii_redactor_preserves_explicitly_empty_context_terms() -> None:
    """
    Ensures an explicitly empty context_terms list is preserved rather than replaced by defaults.

    Args:
        None

    Returns:
        None
    """
    redactor: PiiRedactor = PiiRedactor(
        dict_config={
            "detection_profile": "low_memory",
            CONFIG_REDACTION_RECOGNIZERS_FIELD: {
                RECOGNIZER_RULE_KEY_SSN_LAST4: {
                    CONFIG_ENABLED_FIELD: True,
                    CONFIG_CONTEXT_TERMS_FIELD: [],
                },
            },
        }
    )
    ssn_registration_config: RecognizerRegistrationConfig = (
        redactor.tuple_custom_recognizer_registration_configs[0]
    )
    assert ssn_registration_config.tuple_str_context_terms == ()
    # Normal return on successful assertions for explicit empty context preservation.
    return None
