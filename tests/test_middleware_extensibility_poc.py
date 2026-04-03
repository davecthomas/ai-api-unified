"""Tests for middleware extensibility proof-of-concept behavior."""

from __future__ import annotations

import importlib.util

import pytest

BOOL_HAS_PRESIDIO_ANALYZER: bool = (
    importlib.util.find_spec("presidio_analyzer") is not None
)
BOOL_HAS_PRESIDIO_ANONYMIZER: bool = (
    importlib.util.find_spec("presidio_anonymizer") is not None
)

pytestmark = pytest.mark.skipif(
    not BOOL_HAS_PRESIDIO_ANALYZER or not BOOL_HAS_PRESIDIO_ANONYMIZER,
    reason="Presidio dependencies are required for POC tests.",
)

CONSTANT_TEST_TEXT_WITH_CONTEXT: str = "Customer SSN ending in 6789"
CONSTANT_TEST_TEXT_WITHOUT_CONTEXT: str = "Order 1234 shipped"
CONSTANT_TEST_TEXT_MIXED_CONTEXT: str = (
    "SSN ending in 6789. Unrelated order reference text appears far away before 1234."
)
CONSTANT_EXPECTED_REPLACEMENT_LABEL: str = "SSN_LAST4"
CONSTANT_FOUR_DIGITS_WITH_SSN_CONTEXT: str = "6789"
CONSTANT_EXPECTED_CONTEXTUAL_CANDIDATE_COUNT: int = 1


def test_redact_ssn_last4_with_context() -> None:
    """
    Verifies that SSN last-4 is redacted when explicit SSN context is present.

    Args:
        None

    Returns:
        None for normal test completion.
    """
    from ai_api_unified.middleware.impl.middleware_extensibility_poc import (
        CONSTANT_REDACTION_PREFIX,
        redact_ssn_last4_poc,
    )

    str_redacted_output: str = redact_ssn_last4_poc(
        str_text=CONSTANT_TEST_TEXT_WITH_CONTEXT,
    )
    str_expected_token: str = (
        f"[{CONSTANT_REDACTION_PREFIX}:{CONSTANT_EXPECTED_REPLACEMENT_LABEL}]"
    )

    assert str_expected_token in str_redacted_output
    assert CONSTANT_FOUR_DIGITS_WITH_SSN_CONTEXT not in str_redacted_output
    # Normal return on successful assertions.
    return None


def test_redact_ssn_last4_without_context() -> None:
    """
    Verifies that a generic 4-digit value is not redacted without SSN context.

    Args:
        None

    Returns:
        None for normal test completion.
    """
    from ai_api_unified.middleware.impl.middleware_extensibility_poc import (
        redact_ssn_last4_poc,
    )

    str_redacted_output: str = redact_ssn_last4_poc(
        str_text=CONSTANT_TEST_TEXT_WITHOUT_CONTEXT,
    )

    assert str_redacted_output == CONSTANT_TEST_TEXT_WITHOUT_CONTEXT
    # Normal return on successful assertions.
    return None


def test_analyze_ssn_last4_candidates_keeps_contextual_spans_only() -> None:
    """
    Verifies analyzer output includes only context-qualified SSN last-4 candidates.

    Args:
        None

    Returns:
        None for normal test completion.
    """
    from ai_api_unified.middleware.impl.middleware_extensibility_poc import (
        analyze_ssn_last4_candidates,
    )

    list_recognizer_results = analyze_ssn_last4_candidates(
        str_text=CONSTANT_TEST_TEXT_MIXED_CONTEXT,
    )

    assert len(list_recognizer_results) == CONSTANT_EXPECTED_CONTEXTUAL_CANDIDATE_COUNT
    recognizer_result = list_recognizer_results[0]
    assert (
        CONSTANT_TEST_TEXT_MIXED_CONTEXT[
            recognizer_result.start : recognizer_result.end
        ]
        == CONSTANT_FOUR_DIGITS_WITH_SSN_CONTEXT
    )
    # Normal return on successful assertions.
    return None
