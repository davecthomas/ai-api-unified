from dataclasses import dataclass
import logging
import re
from typing import Any

from presidio_analyzer import (
    Pattern,
    PatternRecognizer,
    RecognizerRegistry,
    RecognizerResult,
)

logger: logging.Logger = logging.getLogger(__name__)

RECOGNIZER_RULE_KEY_SSN_LAST4: str = "ssn_last4"
RECOGNIZER_RULE_KEY_CC_LAST4: str = "cc_last4"
RECOGNIZER_RULE_KEY_DOB: str = "dob"
TUPLE_STR_RECOGNIZER_RULE_KEYS: tuple[str, ...] = (
    RECOGNIZER_RULE_KEY_SSN_LAST4,
    RECOGNIZER_RULE_KEY_CC_LAST4,
    RECOGNIZER_RULE_KEY_DOB,
)

PROVIDER_ENTITY_US_SSN_LAST4: str = "US_SSN_LAST4"
PROVIDER_ENTITY_CREDIT_CARD_LAST4: str = "CREDIT_CARD_LAST4"
PROVIDER_ENTITY_DOB_DATE: str = "DOB_DATE"

RECOGNIZER_NAME_US_SSN_LAST4_CONTEXT: str = "US_SSN_LAST4_ContextRecognizer"
RECOGNIZER_NAME_CREDIT_CARD_LAST4_CONTEXT: str = "CREDIT_CARD_LAST4_ContextRecognizer"
RECOGNIZER_NAME_DOB_DATE_CONTEXT: str = "DOB_DATE_ContextRecognizer"

SSN_LAST4_PATTERN_NAME: str = "ssn_last4_four_digits"
# Match standalone 4-digit spans that are not directly attached to digit/hyphen
# chains (for example, avoid trailing segments inside full phone/SSN formats).
SSN_LAST4_PATTERN_REGEX: str = r"(?<![\d-])\b\d{4}\b(?![-\d])"
# Keep pattern score strictly above zero so PatternRecognizer candidate generation
# does not silently drop matches when confidence_threshold is configured to 0.0.
SSN_LAST4_PATTERN_BASE_SCORE: float = 0.85
CC_LAST4_PATTERN_NAME: str = "cc_last4_tail_digits"
# Match any standalone 4-digit tail that is not part of a longer numeric chain.
# This intentionally allows tails adjacent to masking text such as "xxxx-4242"
# or "**** 4242" while relying on nearby card-specific context terms to avoid
# treating arbitrary 4-digit numbers as payment-card data.
CC_LAST4_PATTERN_REGEX: str = r"(?<!\d)\d{4}(?!\d)"
CC_LAST4_PATTERN_BASE_SCORE: float = 0.85
# Match a narrow DOB candidate set for this phase:
# - ISO date: 1991-12-31
# - US slash date: 01/22/1988
# - Common DMY hyphen date: 31-12-1991
# - Long month form: January 22, 1988
DOB_DATE_PATTERN_NAME: str = "dob_date_candidate"
DOB_DATE_PATTERN_REGEX: str = (
    r"(?<!\d)(?:"
    r"\d{4}-\d{2}-\d{2}"
    r"|"
    r"\d{2}/\d{2}/\d{4}"
    r"|"
    r"\d{2}-\d{2}-\d{4}"
    r"|"
    r"(?i:(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|"
    r"jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|"
    r"nov(?:ember)?|dec(?:ember)?))\s+\d{1,2},\s+\d{4}"
    r")(?!\d)"
)
DOB_DATE_PATTERN_BASE_SCORE: float = 0.85
STRICT_CONTEXTUAL_MATCH_CONFIDENCE_SCORE: float = 1.0
SET_STR_CONTEXT_WINDOW_RIGHT_DELIMITERS: set[str] = {",", ";", "\n", "\r", "."}

DICT_STR_RECOGNIZER_RULE_NAME_TO_RECOGNIZER_NAME: dict[str, str] = {
    RECOGNIZER_RULE_KEY_SSN_LAST4: RECOGNIZER_NAME_US_SSN_LAST4_CONTEXT,
    RECOGNIZER_RULE_KEY_CC_LAST4: RECOGNIZER_NAME_CREDIT_CARD_LAST4_CONTEXT,
    RECOGNIZER_RULE_KEY_DOB: RECOGNIZER_NAME_DOB_DATE_CONTEXT,
}


@dataclass(frozen=True)
class RecognizerRegistrationConfig:
    """
    Immutable runtime registration settings for one middleware recognizer rule.

    Args:
        str_recognizer_rule_name: Middleware recognizer rule name.
        float_confidence_threshold: Minimum score accepted for this rule.
        int_proximity_window_chars: Character window used for context-gating checks.
        int_min_positive_context_matches: Minimum number of positive context
                                          matches required near a candidate span.
        tuple_str_context_terms: Positive context terms that must appear near a candidate span.
        tuple_str_negative_context_terms: Negative context terms that reject a candidate span.

    Returns:
        Frozen dataclass payload used as a hashable cache-safe registration input.
    """

    str_recognizer_rule_name: str
    float_confidence_threshold: float
    int_proximity_window_chars: int
    int_min_positive_context_matches: int
    tuple_str_context_terms: tuple[str, ...]
    tuple_str_negative_context_terms: tuple[str, ...]


class ContextConstrainedPatternRecognizer(PatternRecognizer):
    """
    Presidio pattern recognizer with strict local context gating.

    Candidate spans are produced by regex, then filtered so matches are accepted
    only when required context terms appear near the span and no negative context
    terms are present near the span.

    Presidio recognizer docs:
    https://microsoft.github.io/presidio/analyzer/adding_recognizers/
    """

    def __init__(
        self,
        str_supported_entity: str,
        str_recognizer_name: str,
        str_language: str,
        str_pattern_name: str,
        str_pattern_regex: str,
        float_pattern_score: float,
        float_confidence_threshold: float,
        int_proximity_window_chars: int,
        int_min_positive_context_matches: int,
        tuple_str_context_terms: tuple[str, ...],
        tuple_str_negative_context_terms: tuple[str, ...],
    ) -> None:
        """
        Initializes a context-constrained recognizer over one regex pattern.

        Args:
            str_supported_entity: Provider entity emitted by this recognizer.
            str_recognizer_name: Recognizer name shown in analyzer registry/debug output.
            str_language: Language code used by Presidio analyzer calls.
            str_pattern_name: Presidio pattern identifier.
            str_pattern_regex: Regex used to generate candidate spans.
            float_pattern_score: Base score assigned to regex candidate spans.
            float_confidence_threshold: Minimum score accepted after context gating.
            int_proximity_window_chars: Character window used around each span.
            int_min_positive_context_matches: Minimum number of positive context
                                              matches required near span.
            tuple_str_context_terms: Positive context terms required near span.
            tuple_str_negative_context_terms: Negative context terms that reject span.

        Returns:
            None
        """
        pattern: Pattern = Pattern(
            name=str_pattern_name,
            regex=str_pattern_regex,
            score=float_pattern_score,
        )
        # Do not rely on Presidio context boosting for this recognizer.
        super().__init__(
            supported_entity=str_supported_entity,
            name=str_recognizer_name,
            supported_language=str_language,
            patterns=[pattern],
            context=[],
        )
        self.float_confidence_threshold: float = float_confidence_threshold
        self.int_proximity_window_chars: int = int_proximity_window_chars
        self.int_min_positive_context_matches: int = max(
            1, int_min_positive_context_matches
        )
        self.tuple_str_context_terms: tuple[str, ...] = tuple(
            str_term.lower() for str_term in tuple_str_context_terms
        )
        self.tuple_str_negative_context_terms: tuple[str, ...] = tuple(
            str_term.lower() for str_term in tuple_str_negative_context_terms
        )
        self._bool_negative_context_enabled: bool = (
            len(self.tuple_str_negative_context_terms) > 0
        )

    @staticmethod
    def _build_proximity_window_text_lower(
        str_text: str,
        int_span_start: int,
        int_span_end: int,
        int_proximity_window_chars: int,
    ) -> str:
        """
        Builds the lowercase proximity window text around one candidate span.

        Args:
            str_text: Full source text under analysis.
            int_span_start: Inclusive candidate start character offset.
            int_span_end: Exclusive candidate end character offset.
            int_proximity_window_chars: Character window on each side of span.

        Returns:
            Lowercase substring used for local context term checks.
        """
        int_left_bound: int = max(0, int_span_start - int_proximity_window_chars)
        int_right_bound: int = min(
            len(str_text), int_span_end + int_proximity_window_chars
        )
        int_candidate_end_to_window_end: int = int_right_bound - int_span_end
        # Keep negative and positive context local to the candidate clause by
        # stopping the right-side scan at the first obvious field delimiter.
        # This prevents a following field label such as ", SSN last 4 ..."
        # from suppressing an otherwise valid card-tail match in mixed records.
        for int_relative_idx in range(int_candidate_end_to_window_end):
            if (
                str_text[int_span_end + int_relative_idx]
                in SET_STR_CONTEXT_WINDOW_RIGHT_DELIMITERS
            ):
                int_right_bound = int_span_end + int_relative_idx
                break
        # Expand the raw character window out to token boundaries so configured
        # context phrases are not accidentally truncated mid-word at the window
        # edge (for example, "social security" losing its leading letters).
        while int_left_bound > 0 and str_text[int_left_bound - 1].isalnum():
            int_left_bound -= 1
        while int_right_bound < len(str_text) and str_text[int_right_bound].isalnum():
            int_right_bound += 1
        str_window_text_lower: str = str_text[int_left_bound:int_right_bound].lower()
        # Normal return with lowercased window text for context matching.
        return str_window_text_lower

    @staticmethod
    def _count_context_term_matches(
        str_window_text_lower: str,
        tuple_str_context_terms: tuple[str, ...],
    ) -> int:
        """
        Counts configured context terms found in the local span window.

        Args:
            str_window_text_lower: Lower-cased proximity window text.
            tuple_str_context_terms: Context terms to search for.

        Returns:
            Integer count of matched context terms in the local span window.
        """
        list_tuple_matched_spans: list[tuple[int, int]] = []
        int_match_count: int = 0
        tuple_str_terms_by_length_desc: tuple[str, ...] = tuple(
            sorted(tuple_str_context_terms, key=len, reverse=True)
        )
        # Loop through context terms in longest-first order and count only
        # non-overlapping phrase matches. This prevents nested phrases such as
        # "credit card" and "card" from inflating the match count for the same
        # local signal while still allowing independent anchors like "visa" and
        # "ending in" to count separately.
        for str_context_term in tuple_str_terms_by_length_desc:
            str_context_term_pattern: str = (
                rf"(?<![a-z0-9]){re.escape(str_context_term)}(?![a-z0-9])"
            )
            match_context_term: re.Match[str] | None = re.search(
                str_context_term_pattern,
                str_window_text_lower,
            )
            if match_context_term is None:
                continue
            bool_overlaps_existing_match: bool = False
            # Loop through previously accepted spans and reject nested overlaps.
            for int_match_start, int_match_end in list_tuple_matched_spans:
                if not (
                    match_context_term.end() <= int_match_start
                    or int_match_end <= match_context_term.start()
                ):
                    bool_overlaps_existing_match = True
                    break
            if bool_overlaps_existing_match:
                continue
            list_tuple_matched_spans.append(
                (match_context_term.start(), match_context_term.end())
            )
            int_match_count += 1
        # Normal return with the number of matched context terms in this window.
        return int_match_count

    @staticmethod
    def _build_contextual_confidence_score(
        recognizer_result: RecognizerResult,
        int_positive_context_match_count: int,
    ) -> float:
        """
        Builds the final confidence score for a strict context-qualified match.

        Args:
            recognizer_result: Regex candidate returned by Presidio pattern matching.
            int_positive_context_match_count: Number of matched positive context terms.

        Returns:
            Final confidence score for the accepted recognizer result.
        """
        if int_positive_context_match_count <= 0:
            # Early return because spans without positive context are never accepted.
            return recognizer_result.score
        # A match that survives strict positive and negative context gating is a
        # high-confidence detection for this phase. Promote the final score so
        # configured thresholds in the documented 0.0-1.0 range do not silently
        # disable an otherwise valid contextual SSN last-4 detection.
        return max(recognizer_result.score, STRICT_CONTEXTUAL_MATCH_CONFIDENCE_SCORE)

    def analyze(
        self,
        text: str,
        entities: list[str],
        nlp_artifacts: Any | None = None,
        regex_flags: int | None = None,
    ) -> list[RecognizerResult]:
        """
        Runs regex detection and keeps only spans passing strict local context checks.

        Args:
            text: Source text analyzed by Presidio.
            entities: Entity filter list requested by the analyzer.
            nlp_artifacts: Optional NLP artifacts provided by Presidio analyzer flow.
            regex_flags: Optional regex flags override used by Presidio internals.

        Returns:
            Filtered recognizer results after positive and negative context gating.
        """
        list_candidate_results: list[RecognizerResult] = super().analyze(
            text=text,
            entities=entities,
            nlp_artifacts=nlp_artifacts,
            regex_flags=regex_flags,
        )
        list_context_qualified_results: list[RecognizerResult] = []

        # Loop through regex candidates and keep only context-qualified spans.
        for recognizer_result in list_candidate_results:
            str_window_text_lower: str = self._build_proximity_window_text_lower(
                str_text=text,
                int_span_start=recognizer_result.start,
                int_span_end=recognizer_result.end,
                int_proximity_window_chars=self.int_proximity_window_chars,
            )
            int_positive_context_match_count: int = self._count_context_term_matches(
                str_window_text_lower=str_window_text_lower,
                tuple_str_context_terms=self.tuple_str_context_terms,
            )
            if int_positive_context_match_count < self.int_min_positive_context_matches:
                continue
            if self._bool_negative_context_enabled:
                int_negative_context_match_count: int = (
                    self._count_context_term_matches(
                        str_window_text_lower=str_window_text_lower,
                        tuple_str_context_terms=self.tuple_str_negative_context_terms,
                    )
                )
                if int_negative_context_match_count > 0:
                    continue
            recognizer_result.score = self._build_contextual_confidence_score(
                recognizer_result=recognizer_result,
                int_positive_context_match_count=int_positive_context_match_count,
            )
            if recognizer_result.score < self.float_confidence_threshold:
                continue
            list_context_qualified_results.append(recognizer_result)
        # Normal return with strict context-qualified recognizer results.
        return list_context_qualified_results


class CustomRecognizerFactory:
    """
    Builds custom Presidio recognizers for middleware extension rules.

    Current behavior:
    - `ssn_last4` builds a strict context-constrained recognizer.
    - `cc_last4` builds a strict context-constrained recognizer.
    - `dob` builds a strict context-constrained date recognizer.
    """

    @classmethod
    def list_pattern_recognizers_for_rules(
        cls,
        tuple_recognizer_registration_configs: tuple[RecognizerRegistrationConfig, ...],
        str_language: str,
    ) -> list[PatternRecognizer]:
        """
        Builds PatternRecognizer objects for the provided middleware rule names.

        Args:
            tuple_recognizer_registration_configs: Ordered tuple of enabled middleware
                                                   recognizer registration settings.
            str_language: Language code used by Presidio recognizers.

        Returns:
            List of PatternRecognizer instances for known rule names. Unknown
            rule names are ignored.
        """
        list_pattern_recognizers: list[PatternRecognizer] = []
        # Loop through enabled rule configs and build corresponding recognizers.
        for recognizer_registration_config in tuple_recognizer_registration_configs:
            pattern_recognizer: PatternRecognizer | None = (
                cls._build_pattern_recognizer_for_rule(
                    recognizer_registration_config=recognizer_registration_config,
                    str_language=str_language,
                )
            )
            if pattern_recognizer is None:
                continue
            list_pattern_recognizers.append(pattern_recognizer)
        # Normal return with recognizers for known enabled rule names.
        return list_pattern_recognizers

    @classmethod
    def register_pattern_recognizers(
        cls,
        recognizer_registry: RecognizerRegistry,
        tuple_recognizer_registration_configs: tuple[RecognizerRegistrationConfig, ...],
        str_language: str,
    ) -> tuple[str, ...]:
        """
        Registers custom recognizers in the provided Presidio registry.

        Args:
            recognizer_registry: Presidio recognizer registry instance bound to analyzer engine.
            tuple_recognizer_registration_configs: Ordered tuple of enabled middleware
                                                   recognizer registration settings.
            str_language: Language code used by Presidio recognizers.

        Returns:
            Immutable tuple of recognizer names registered for this analyzer
            instance. A tuple is used intentionally to communicate that this is
            a read-only registration snapshot for logging and tests, not a
            mutable working collection.
        """
        list_pattern_recognizers: list[PatternRecognizer] = (
            cls.list_pattern_recognizers_for_rules(
                tuple_recognizer_registration_configs=tuple_recognizer_registration_configs,
                str_language=str_language,
            )
        )
        list_str_registered_names: list[str] = []
        # Loop through built recognizers and register each in the analyzer registry.
        for pattern_recognizer in list_pattern_recognizers:
            recognizer_registry.add_recognizer(pattern_recognizer)
            list_str_registered_names.append(pattern_recognizer.name)
        # Convert to an immutable snapshot so callers treat registration output as read-only.
        tuple_str_registered_names: tuple[str, ...] = tuple(list_str_registered_names)
        # Normal return with registered recognizer names for diagnostics/tests.
        return tuple_str_registered_names

    @classmethod
    def _build_pattern_recognizer_for_rule(
        cls,
        recognizer_registration_config: RecognizerRegistrationConfig,
        str_language: str,
    ) -> PatternRecognizer | None:
        """
        Routes one middleware rule name to its corresponding recognizer instance.

        Args:
            recognizer_registration_config: Middleware recognizer registration settings.
            str_language: Language code used by Presidio recognizers.

        Returns:
            A PatternRecognizer for known rule names, or None for unknown
            rule names.
        """
        str_recognizer_rule_name: str = (
            recognizer_registration_config.str_recognizer_rule_name
        )
        if str_recognizer_rule_name == RECOGNIZER_RULE_KEY_SSN_LAST4:
            # Normal return for SSN last-4 recognizer build path.
            return cls._build_ssn_last4_context_constrained_recognizer(
                str_supported_entity=PROVIDER_ENTITY_US_SSN_LAST4,
                str_recognizer_name=RECOGNIZER_NAME_US_SSN_LAST4_CONTEXT,
                str_language=str_language,
                recognizer_registration_config=recognizer_registration_config,
            )
        if str_recognizer_rule_name == RECOGNIZER_RULE_KEY_CC_LAST4:
            # Normal return for card last-4 recognizer build path.
            return cls._build_cc_last4_context_constrained_recognizer(
                str_supported_entity=PROVIDER_ENTITY_CREDIT_CARD_LAST4,
                str_recognizer_name=RECOGNIZER_NAME_CREDIT_CARD_LAST4_CONTEXT,
                str_language=str_language,
                recognizer_registration_config=recognizer_registration_config,
            )
        if str_recognizer_rule_name == RECOGNIZER_RULE_KEY_DOB:
            # Normal return for DOB recognizer build path.
            return cls._build_dob_context_constrained_recognizer(
                str_supported_entity=PROVIDER_ENTITY_DOB_DATE,
                str_recognizer_name=RECOGNIZER_NAME_DOB_DATE_CONTEXT,
                str_language=str_language,
                recognizer_registration_config=recognizer_registration_config,
            )
        logger.debug(
            "Skipping unknown custom recognizer rule name during registration: %s",
            str_recognizer_rule_name,
        )
        # Early return because unknown rule names are ignored by registration plumbing.
        return None

    @staticmethod
    def _build_ssn_last4_context_constrained_recognizer(
        str_supported_entity: str,
        str_recognizer_name: str,
        str_language: str,
        recognizer_registration_config: RecognizerRegistrationConfig,
    ) -> PatternRecognizer:
        """
        Builds a strict context-constrained recognizer for SSN last-4 detection.

        Args:
            str_supported_entity: Provider entity type emitted by the recognizer.
            str_recognizer_name: Presidio registry recognizer name.
            str_language: Language code used by Presidio recognizers.
            recognizer_registration_config: Rule-level registration settings from middleware config.

        Returns:
            ContextConstrainedPatternRecognizer configured for SSN last-4 logic.
        """
        pattern_recognizer: ContextConstrainedPatternRecognizer = (
            ContextConstrainedPatternRecognizer(
                str_supported_entity=str_supported_entity,
                str_recognizer_name=str_recognizer_name,
                str_language=str_language,
                str_pattern_name=SSN_LAST4_PATTERN_NAME,
                str_pattern_regex=SSN_LAST4_PATTERN_REGEX,
                # Pattern score is intentionally decoupled from threshold because
                # Presidio drops zero-score regex candidates before contextual gates.
                float_pattern_score=SSN_LAST4_PATTERN_BASE_SCORE,
                float_confidence_threshold=recognizer_registration_config.float_confidence_threshold,
                int_proximity_window_chars=recognizer_registration_config.int_proximity_window_chars,
                int_min_positive_context_matches=recognizer_registration_config.int_min_positive_context_matches,
                tuple_str_context_terms=recognizer_registration_config.tuple_str_context_terms,
                tuple_str_negative_context_terms=recognizer_registration_config.tuple_str_negative_context_terms,
            )
        )
        # Normal return with strict context-constrained SSN last-4 recognizer.
        return pattern_recognizer

    @staticmethod
    def _build_cc_last4_context_constrained_recognizer(
        str_supported_entity: str,
        str_recognizer_name: str,
        str_language: str,
        recognizer_registration_config: RecognizerRegistrationConfig,
    ) -> PatternRecognizer:
        """
        Builds a strict context-constrained recognizer for payment-card last-4 detection.

        Args:
            str_supported_entity: Provider entity type emitted by the recognizer.
            str_recognizer_name: Presidio registry recognizer name.
            str_language: Language code used by Presidio recognizers.
            recognizer_registration_config: Rule-level registration settings from middleware config.

        Returns:
            ContextConstrainedPatternRecognizer configured for card last-4 logic.
        """
        pattern_recognizer: ContextConstrainedPatternRecognizer = (
            ContextConstrainedPatternRecognizer(
                str_supported_entity=str_supported_entity,
                str_recognizer_name=str_recognizer_name,
                str_language=str_language,
                str_pattern_name=CC_LAST4_PATTERN_NAME,
                str_pattern_regex=CC_LAST4_PATTERN_REGEX,
                # Pattern score is intentionally decoupled from threshold because
                # Presidio drops zero-score regex candidates before contextual gates.
                float_pattern_score=CC_LAST4_PATTERN_BASE_SCORE,
                float_confidence_threshold=recognizer_registration_config.float_confidence_threshold,
                int_proximity_window_chars=recognizer_registration_config.int_proximity_window_chars,
                int_min_positive_context_matches=recognizer_registration_config.int_min_positive_context_matches,
                tuple_str_context_terms=recognizer_registration_config.tuple_str_context_terms,
                tuple_str_negative_context_terms=recognizer_registration_config.tuple_str_negative_context_terms,
            )
        )
        # Normal return with strict context-constrained card last-4 recognizer.
        return pattern_recognizer

    @staticmethod
    def _build_dob_context_constrained_recognizer(
        str_supported_entity: str,
        str_recognizer_name: str,
        str_language: str,
        recognizer_registration_config: RecognizerRegistrationConfig,
    ) -> PatternRecognizer:
        """
        Builds a strict context-constrained recognizer for DOB detection.

        Args:
            str_supported_entity: Provider entity type emitted by the recognizer.
            str_recognizer_name: Presidio registry recognizer name.
            str_language: Language code used by Presidio recognizers.
            recognizer_registration_config: Rule-level registration settings from middleware config.

        Returns:
            ContextConstrainedPatternRecognizer configured for DOB logic.
        """
        pattern_recognizer: ContextConstrainedPatternRecognizer = (
            ContextConstrainedPatternRecognizer(
                str_supported_entity=str_supported_entity,
                str_recognizer_name=str_recognizer_name,
                str_language=str_language,
                str_pattern_name=DOB_DATE_PATTERN_NAME,
                str_pattern_regex=DOB_DATE_PATTERN_REGEX,
                # Pattern score is intentionally decoupled from threshold because
                # Presidio drops zero-score regex candidates before contextual gates.
                float_pattern_score=DOB_DATE_PATTERN_BASE_SCORE,
                float_confidence_threshold=recognizer_registration_config.float_confidence_threshold,
                int_proximity_window_chars=recognizer_registration_config.int_proximity_window_chars,
                int_min_positive_context_matches=recognizer_registration_config.int_min_positive_context_matches,
                tuple_str_context_terms=recognizer_registration_config.tuple_str_context_terms,
                tuple_str_negative_context_terms=recognizer_registration_config.tuple_str_negative_context_terms,
            )
        )
        # Normal return with strict context-constrained DOB recognizer.
        return pattern_recognizer
