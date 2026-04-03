"""Tests for Presidio log-noise filtering helpers."""

from __future__ import annotations

import logging

from ai_api_unified.middleware.impl.presidio_log_control import (
    PRESIDIO_ANALYZER_LOGGER_NAME,
    PRESIDIO_CREATED_NLP_ENGINE_INFO_SUBSTRING,
    PRESIDIO_DEFAULT_REGISTRY_INFO_SUBSTRING,
    PRESIDIO_ENTITY_RECOGNIZER_WARNING_SUBSTRING,
    PRESIDIO_LANGUAGE_MISMATCH_WARNING_SUBSTRING,
    PRESIDIO_LOADED_RECOGNIZER_INFO_SUBSTRING,
    PRESIDIO_SKIP_CONTEXT_EXTRACTION_INFO_SUBSTRING,
    PRESIDIO_USING_DEVICE_INFO_SUBSTRING,
    PresidioKnownWarningNoiseFilter,
)

ALLOWED_LOGGER_NAME: str = "some-other-logger"
ALLOWED_WARNING_MESSAGE: str = "A different warning that should remain visible."
ALLOWED_INFO_MESSAGE: str = "Initialized Presidio Analyzer with profile."


def test_presidio_noise_filter_suppresses_known_language_mismatch_warning() -> None:
    """
    Ensures the known non-actionable Presidio language mismatch warning is filtered.

    Args:
        None

    Returns:
        None for normal test completion.
    """
    filter_presidio_noise: PresidioKnownWarningNoiseFilter = (
        PresidioKnownWarningNoiseFilter()
    )
    log_record: logging.LogRecord = logging.LogRecord(
        name=PRESIDIO_ANALYZER_LOGGER_NAME,
        level=logging.WARNING,
        pathname=__file__,
        lineno=1,
        msg=PRESIDIO_LANGUAGE_MISMATCH_WARNING_SUBSTRING,
        args=(),
        exc_info=None,
    )

    assert filter_presidio_noise.filter(log_record) is False
    # Normal return on successful assertions.
    return None


def test_presidio_noise_filter_allows_unrelated_warning_message() -> None:
    """
    Ensures warnings not matching the known noisy Presidio pattern are preserved.

    Args:
        None

    Returns:
        None for normal test completion.
    """
    filter_presidio_noise: PresidioKnownWarningNoiseFilter = (
        PresidioKnownWarningNoiseFilter()
    )
    log_record: logging.LogRecord = logging.LogRecord(
        name=ALLOWED_LOGGER_NAME,
        level=logging.WARNING,
        pathname=__file__,
        lineno=1,
        msg=ALLOWED_WARNING_MESSAGE,
        args=(),
        exc_info=None,
    )

    assert filter_presidio_noise.filter(log_record) is True
    # Normal return on successful assertions.
    return None


def test_presidio_noise_filter_suppresses_loaded_recognizer_info() -> None:
    """
    Ensures recognizer-load INFO chatter from Presidio analyzer is filtered.

    Args:
        None

    Returns:
        None for normal test completion.
    """
    filter_presidio_noise: PresidioKnownWarningNoiseFilter = (
        PresidioKnownWarningNoiseFilter()
    )
    log_record: logging.LogRecord = logging.LogRecord(
        name=PRESIDIO_ANALYZER_LOGGER_NAME,
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg=PRESIDIO_LOADED_RECOGNIZER_INFO_SUBSTRING,
        args=(),
        exc_info=None,
    )

    assert filter_presidio_noise.filter(log_record) is False
    # Normal return on successful assertions.
    return None


def test_presidio_noise_filter_suppresses_using_device_info() -> None:
    """
    Ensures Presidio device-selection INFO chatter is filtered.

    Args:
        None

    Returns:
        None for normal test completion.
    """
    filter_presidio_noise: PresidioKnownWarningNoiseFilter = (
        PresidioKnownWarningNoiseFilter()
    )
    log_record: logging.LogRecord = logging.LogRecord(
        name=PRESIDIO_ANALYZER_LOGGER_NAME,
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg=PRESIDIO_USING_DEVICE_INFO_SUBSTRING,
        args=(),
        exc_info=None,
    )

    assert filter_presidio_noise.filter(log_record) is False
    # Normal return on successful assertions.
    return None


def test_presidio_noise_filter_suppresses_created_nlp_engine_info() -> None:
    """
    Ensures Presidio NLP-engine creation INFO chatter is filtered.

    Args:
        None

    Returns:
        None for normal test completion.
    """
    filter_presidio_noise: PresidioKnownWarningNoiseFilter = (
        PresidioKnownWarningNoiseFilter()
    )
    log_record: logging.LogRecord = logging.LogRecord(
        name=PRESIDIO_ANALYZER_LOGGER_NAME,
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg=PRESIDIO_CREATED_NLP_ENGINE_INFO_SUBSTRING,
        args=(),
        exc_info=None,
    )

    assert filter_presidio_noise.filter(log_record) is False
    # Normal return on successful assertions.
    return None


def test_presidio_noise_filter_suppresses_default_registry_info() -> None:
    """
    Ensures Presidio default-registry INFO chatter is filtered.

    Args:
        None

    Returns:
        None for normal test completion.
    """
    filter_presidio_noise: PresidioKnownWarningNoiseFilter = (
        PresidioKnownWarningNoiseFilter()
    )
    log_record: logging.LogRecord = logging.LogRecord(
        name=PRESIDIO_ANALYZER_LOGGER_NAME,
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg=PRESIDIO_DEFAULT_REGISTRY_INFO_SUBSTRING,
        args=(),
        exc_info=None,
    )

    assert filter_presidio_noise.filter(log_record) is False
    # Normal return on successful assertions.
    return None


def test_presidio_noise_filter_suppresses_skip_context_extraction_info() -> None:
    """
    Ensures regex-profile context-skip INFO chatter is filtered.

    Args:
        None

    Returns:
        None for normal test completion.
    """
    filter_presidio_noise: PresidioKnownWarningNoiseFilter = (
        PresidioKnownWarningNoiseFilter()
    )
    log_record: logging.LogRecord = logging.LogRecord(
        name=PRESIDIO_ANALYZER_LOGGER_NAME,
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg=PRESIDIO_SKIP_CONTEXT_EXTRACTION_INFO_SUBSTRING,
        args=(),
        exc_info=None,
    )

    assert filter_presidio_noise.filter(log_record) is False
    # Normal return on successful assertions.
    return None


def test_presidio_noise_filter_suppresses_missing_entity_recognizer_warning() -> None:
    """
    Ensures regex-profile Presidio entity recognizer warnings are filtered.

    Args:
        None

    Returns:
        None for normal test completion.
    """
    filter_presidio_noise: PresidioKnownWarningNoiseFilter = (
        PresidioKnownWarningNoiseFilter()
    )
    log_record: logging.LogRecord = logging.LogRecord(
        name=PRESIDIO_ANALYZER_LOGGER_NAME,
        level=logging.WARNING,
        pathname=__file__,
        lineno=1,
        msg=PRESIDIO_ENTITY_RECOGNIZER_WARNING_SUBSTRING,
        args=(),
        exc_info=None,
    )

    assert filter_presidio_noise.filter(log_record) is False
    # Normal return on successful assertions.
    return None


def test_presidio_noise_filter_allows_other_presidio_info_messages() -> None:
    """
    Ensures Presidio INFO records remain visible when not matching noisy patterns.

    Args:
        None

    Returns:
        None for normal test completion.
    """
    filter_presidio_noise: PresidioKnownWarningNoiseFilter = (
        PresidioKnownWarningNoiseFilter()
    )
    log_record: logging.LogRecord = logging.LogRecord(
        name=PRESIDIO_ANALYZER_LOGGER_NAME,
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg=ALLOWED_INFO_MESSAGE,
        args=(),
        exc_info=None,
    )

    assert filter_presidio_noise.filter(log_record) is True
    # Normal return on successful assertions.
    return None
