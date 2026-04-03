"""
Proof-of-concept helpers for middleware extensibility experiments.

This module intentionally implements one hard-wired Presidio extension to validate
an early assumption before broader phased implementation work:

- detect and redact SSN last-4 values only when nearby SSN context is present.

This file is a POC utility and is not wired into the production middleware flow.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, Iterator, cast

from presidio_analyzer import (
    AnalyzerEngine,
    Pattern,
    PatternRecognizer,
    RecognizerResult,
)
from presidio_analyzer.nlp_engine import NlpArtifacts, NlpEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

from .presidio_log_control import configure_presidio_log_noise_filter

_LOGGER: logging.Logger = logging.getLogger(__name__)

# Install one-time Presidio warning-noise filtering for this process.
configure_presidio_log_noise_filter()

CONSTANT_DEFAULT_LANGUAGE: str = "en"
CONSTANT_SSN_LAST4_PROVIDER_ENTITY: str = "US_SSN_LAST4"
CONSTANT_SSN_LAST4_RECOGNIZER_NAME: str = "SsnLast4ContextRecognizer"
CONSTANT_SSN_LAST4_PATTERN_NAME: str = "ssn_last4_four_digits"
CONSTANT_SSN_LAST4_PATTERN_REGEX: str = r"\b\d{4}\b"
CONSTANT_SSN_LAST4_BASE_SCORE: float = 0.30
CONSTANT_SSN_LAST4_DEFAULT_PROXIMITY_WINDOW_CHARS: int = 28
CONSTANT_REDACTION_LABEL_DEFAULT: str = "SSN_LAST4"
CONSTANT_REDACTION_PREFIX: str = "REDACTED"
CONSTANT_OPERATOR_REPLACE: str = "replace"
CONSTANT_OPERATOR_NEW_VALUE_KEY: str = "new_value"
CONSTANT_OPERATOR_DEFAULT_KEY: str = "DEFAULT"

LIST_STR_SSN_CONTEXT_TERMS_DEFAULT: list[str] = [
    "ssn",
    "social security",
    "social security number",
    "last 4",
    "last four",
    "ending in",
]


class RegexOnlyNlpEngine(NlpEngine):
    """
    Minimal NLP engine implementation for regex-only Presidio analysis in this POC.

    The class returns empty NLP artifacts so Presidio can run pattern recognizers
    without loading spaCy models.
    """

    def is_loaded(self) -> bool:
        """
        Reports that the POC NLP engine is immediately ready.

        Args:
            None

        Returns:
            Always returns True because there is no underlying ML model to load.
        """
        # Normal return because this engine has no async/model load lifecycle.
        return True

    def load(self) -> None:
        """
        No-op model load for regex-only analyzer operation.

        Args:
            None

        Returns:
            None because no model initialization is required.
        """
        # Normal return from no-op load path.
        return None

    def is_stopword(self, word: str, language: str) -> bool:
        """
        Indicates stopword status for the provided token.

        Args:
            word: Raw token string.
            language: Language code requested by the analyzer.

        Returns:
            Always returns False because this engine performs no NLP semantics.
        """
        # Normal return because regex-only mode has no stopword dictionary.
        return False

    def is_punct(self, word: str, language: str) -> bool:
        """
        Indicates punctuation status for the provided token.

        Args:
            word: Raw token string.
            language: Language code requested by the analyzer.

        Returns:
            Always returns False because this engine performs no token NLP parsing.
        """
        # Normal return because regex-only mode does not tokenize punctuation semantics.
        return False

    def get_nlp(self, language: str | None = None) -> Any:
        """
        Returns the backing NLP model object for a language.

        Args:
            language: Optional language code requested by Presidio.

        Returns:
            Always returns None because there is no backing NLP model object.
        """
        # Normal return because regex-only mode has no backing NLP object.
        return None

    def get_supported_entities(self) -> list[str]:
        """
        Returns entities extracted by NLP features.

        Args:
            None

        Returns:
            Empty list because this engine contributes no NLP-extracted entities.
        """
        # Normal return because all detection comes from pattern recognizers.
        return []

    def get_supported_languages(self) -> list[str]:
        """
        Returns supported NLP languages.

        Args:
            None

        Returns:
            Empty list because this minimal engine is language-agnostic in practice.
        """
        # Normal return because this POC engine has no language pack dependency.
        return []

    def process_text(self, text: str, language: str) -> NlpArtifacts:
        """
        Creates empty NLP artifacts for one text sample.

        Args:
            text: Source text requested by Presidio.
            language: Language code requested by Presidio.

        Returns:
            Empty NlpArtifacts payload that satisfies Presidio engine contracts.
        """
        nlp_artifacts: NlpArtifacts = NlpArtifacts(
            entities=[],
            tokens=cast(Any, None),
            tokens_indices=[],
            lemmas=[],
            nlp_engine=self,
            language=language,
        )
        # Normal return with empty artifacts to support regex-only analysis.
        return nlp_artifacts

    def process_batch(
        self, texts: Iterable[str], language: str, *args: Any, **kwargs: Any
    ) -> Iterator[tuple[str, NlpArtifacts]]:
        """
        Creates empty NLP artifacts for each text sample in a batch.

        Args:
            texts: Iterable text payloads.
            language: Language code requested by Presidio.
            *args: Unused positional passthrough parameters.
            **kwargs: Unused keyword passthrough parameters.

        Returns:
            Iterator yielding tuples of source text and empty NlpArtifacts.
        """
        # Loop over batch items to emit one empty artifact bundle per text.
        for str_text in texts:
            yield str_text, self.process_text(str_text, language)


def _build_hard_wired_ssn_last4_recognizer() -> PatternRecognizer:
    """
    Builds a hard-wired Presidio pattern recognizer for SSN last-4 candidates.

    Args:
        None

    Returns:
        A PatternRecognizer configured with a generic 4-digit pattern plus SSN
        context hints.
    """
    pattern_ssn_last4: Pattern = Pattern(
        name=CONSTANT_SSN_LAST4_PATTERN_NAME,
        regex=CONSTANT_SSN_LAST4_PATTERN_REGEX,
        score=CONSTANT_SSN_LAST4_BASE_SCORE,
    )
    recognizer: PatternRecognizer = PatternRecognizer(
        supported_entity=CONSTANT_SSN_LAST4_PROVIDER_ENTITY,
        name=CONSTANT_SSN_LAST4_RECOGNIZER_NAME,
        patterns=[pattern_ssn_last4],
        context=LIST_STR_SSN_CONTEXT_TERMS_DEFAULT,
    )
    # Normal return with configured POC recognizer.
    return recognizer


def build_ssn_last4_poc_analyzer(
    str_language: str = CONSTANT_DEFAULT_LANGUAGE,
) -> AnalyzerEngine:
    """
    Builds a regex-only AnalyzerEngine and registers the SSN last-4 recognizer.

    Args:
        str_language: Language code used by Presidio analyzer operations.

    Returns:
        AnalyzerEngine instance with the hard-wired SSN last-4 recognizer registered.
    """
    analyzer_engine: AnalyzerEngine = AnalyzerEngine(
        nlp_engine=RegexOnlyNlpEngine(),
        supported_languages=[str_language],
    )
    recognizer_ssn_last4: PatternRecognizer = _build_hard_wired_ssn_last4_recognizer()
    analyzer_engine.registry.add_recognizer(recognizer_ssn_last4)
    # Normal return with analyzer containing hard-wired SSN last-4 recognizer.
    return analyzer_engine


def _contains_ssn_context_near_span(
    str_text: str,
    int_span_start: int,
    int_span_end: int,
    int_proximity_window_chars: int,
) -> bool:
    """
    Checks for SSN-related context near a detected numeric span.

    Args:
        str_text: Full source text.
        int_span_start: Inclusive start offset of candidate span.
        int_span_end: Exclusive end offset of candidate span.
        int_proximity_window_chars: Character window on each side of span.

    Returns:
        True when at least one SSN context term appears near the candidate span.
    """
    int_left_bound: int = max(0, int_span_start - int_proximity_window_chars)
    int_right_bound: int = min(len(str_text), int_span_end + int_proximity_window_chars)
    str_window_text_lower: str = str_text[int_left_bound:int_right_bound].lower()

    # Loop through context terms and accept the span on the first contextual hit.
    for str_context_term in LIST_STR_SSN_CONTEXT_TERMS_DEFAULT:
        if str_context_term in str_window_text_lower:
            # Early return because a required SSN context signal was found.
            return True

    # Normal return because no SSN context terms were found near this span.
    return False


def analyze_ssn_last4_candidates(
    str_text: str,
    int_proximity_window_chars: int = CONSTANT_SSN_LAST4_DEFAULT_PROXIMITY_WINDOW_CHARS,
) -> list[RecognizerResult]:
    """
    Detects SSN last-4 candidates and filters them using hard-wired proximity context.

    Args:
        str_text: Source text analyzed for SSN last-4 detection.
        int_proximity_window_chars: Character window used to enforce SSN context.

    Returns:
        List of context-qualified RecognizerResult values for SSN last-4 spans.
    """
    analyzer_engine: AnalyzerEngine = build_ssn_last4_poc_analyzer(
        str_language=CONSTANT_DEFAULT_LANGUAGE,
    )
    list_candidates: list[RecognizerResult] = analyzer_engine.analyze(
        text=str_text,
        entities=[CONSTANT_SSN_LAST4_PROVIDER_ENTITY],
        language=CONSTANT_DEFAULT_LANGUAGE,
        score_threshold=0.0,
    )

    list_context_qualified_results: list[RecognizerResult] = []

    # Loop over raw candidates and keep only candidates with nearby SSN context terms.
    for recognizer_result in list_candidates:
        bool_has_context: bool = _contains_ssn_context_near_span(
            str_text=str_text,
            int_span_start=recognizer_result.start,
            int_span_end=recognizer_result.end,
            int_proximity_window_chars=int_proximity_window_chars,
        )
        if bool_has_context:
            list_context_qualified_results.append(recognizer_result)

    # Normal return with candidates that passed proximity context filtering.
    return list_context_qualified_results


def redact_ssn_last4_poc(
    str_text: str,
    int_proximity_window_chars: int = CONSTANT_SSN_LAST4_DEFAULT_PROXIMITY_WINDOW_CHARS,
    str_replacement_label: str = CONSTANT_REDACTION_LABEL_DEFAULT,
) -> str:
    """
    Redacts SSN last-4 values in text using the hard-wired POC recognizer behavior.

    Args:
        str_text: Source text to sanitize.
        int_proximity_window_chars: Character window used for SSN context checks.
        str_replacement_label: Redaction entity label inserted in replacement tokens.

    Returns:
        Sanitized text string with SSN last-4 candidates replaced when context checks pass.
    """
    if not str_text:
        # Early return because empty input has nothing to redact.
        return str_text

    list_results: list[RecognizerResult] = analyze_ssn_last4_candidates(
        str_text=str_text,
        int_proximity_window_chars=int_proximity_window_chars,
    )
    if not list_results:
        # Early return because no context-qualified SSN last-4 spans were detected.
        return str_text

    str_replacement_token: str = (
        f"[{CONSTANT_REDACTION_PREFIX}:{str_replacement_label.strip().upper()}]"
    )
    dict_operators: dict[str, OperatorConfig] = {
        CONSTANT_OPERATOR_DEFAULT_KEY: OperatorConfig(
            CONSTANT_OPERATOR_REPLACE,
            {CONSTANT_OPERATOR_NEW_VALUE_KEY: str_replacement_token},
        )
    }

    anonymizer_engine: AnonymizerEngine = AnonymizerEngine()
    list_any_results: Any = list_results
    anonymizer_result: Any = anonymizer_engine.anonymize(
        text=str_text,
        analyzer_results=list_any_results,
        operators=dict_operators,
    )
    str_sanitized_text: str = str(anonymizer_result.text)
    _LOGGER.info(
        "POC SSN last-4 redaction replaced %s span(s).",
        len(list_results),
    )
    # Normal return with redacted text from hard-wired POC flow.
    return str_sanitized_text
