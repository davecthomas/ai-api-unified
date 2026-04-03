from copy import deepcopy
from functools import cache
import importlib.util
import logging
from typing import Any, Iterable, Iterator, cast

from presidio_analyzer import AnalyzerEngine, RecognizerResult
from presidio_analyzer.nlp_engine import NlpEngineProvider, NlpEngine, NlpArtifacts
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

from ai_api_unified.middleware.middleware_config import (
    DEFAULT_CC_LAST4_CONFIDENCE_THRESHOLD,
    DEFAULT_DOB_CONFIDENCE_THRESHOLD,
    DEFAULT_PROXIMITY_WINDOW_CHARS,
    DEFAULT_RECOGNIZER_CONFIDENCE_THRESHOLD,
    DEFAULT_SSN_LAST4_CONFIDENCE_THRESHOLD,
    LIST_STR_DEFAULT_CC_LAST4_CONTEXT_TERMS,
    LIST_STR_DEFAULT_DOB_CONTEXT_TERMS,
    LIST_STR_DEFAULT_SSN_LAST4_CONTEXT_TERMS,
    PiiRedactionSettingsModel,
)

from .base_redactor import BaseRedactorLayer, RedactionResult
from .custom_recognizer_factory import (
    CustomRecognizerFactory,
    PROVIDER_ENTITY_CREDIT_CARD_LAST4,
    PROVIDER_ENTITY_DOB_DATE,
    RecognizerRegistrationConfig,
    RECOGNIZER_RULE_KEY_CC_LAST4,
    RECOGNIZER_RULE_KEY_DOB,
    RECOGNIZER_RULE_KEY_SSN_LAST4,
    TUPLE_STR_RECOGNIZER_RULE_KEYS,
)
from .presidio_log_control import configure_presidio_log_noise_filter
from ..redaction_exceptions import PiiRedactionRuntimeError

logger: logging.Logger = logging.getLogger(__name__)

# Install one-time Presidio warning-noise filtering for this process.
configure_presidio_log_noise_filter()

try:
    import usaddress
except ModuleNotFoundError:
    usaddress = None

DETECTION_PROFILE_KEY: str = "detection_profile"
DETECTION_PROFILE_LOW_MEMORY: str = "low_memory"
DETECTION_PROFILE_BALANCED: str = "balanced"
DETECTION_PROFILE_HIGH_ACCURACY: str = "high_accuracy"
PROFILE_LARGE: str = "large_spacy_500MB"
PROFILE_SMALL: str = "small_spacy_15MB"
PROFILE_REGEX: str = "micro_regex"
DICT_STR_DETECTION_PROFILE_TO_RUNTIME_PROFILE: dict[str, str] = {
    DETECTION_PROFILE_LOW_MEMORY: PROFILE_REGEX,
    DETECTION_PROFILE_BALANCED: PROFILE_SMALL,
    DETECTION_PROFILE_HIGH_ACCURACY: PROFILE_LARGE,
}
INVALID_DETECTION_PROFILE_FALLBACK_WARNING_MESSAGE: str = (
    "Invalid detection profile value %r; falling back to '%s'."
)
BOOL_TRUTHINESS_FALLBACK_WARNING_MESSAGE: str = (
    "Setting `%s` received unsupported value type %s; coercing via truthiness."
)
SPACY_MODEL_SMALL: str = "en_core_web_sm"
SPACY_MODEL_LARGE: str = "en_core_web_lg"

LANGUAGE_KEY: str = "language"
DEFAULT_EN: str = "en"
DEFAULT_REDACTION_LABEL_KEY: str = "default_redaction_label"
DEFAULT_REDACTED_TEXT: str = "REDACTED"
ALLOWED_ENTITIES_KEY: str = "allowed_entities"
ENTITY_LABEL_MAP_KEY: str = "entity_label_map"
REDACTION_RECOGNIZERS_KEY: str = "redaction_recognizers"
RECOGNIZER_ENABLED_KEY: str = "enabled"
RECOGNIZER_CONFIDENCE_THRESHOLD_KEY: str = "confidence_threshold"
RECOGNIZER_CONTEXT_TERMS_KEY: str = "context_terms"
RECOGNIZER_NEGATIVE_CONTEXT_TERMS_KEY: str = "negative_context_terms"
PROXIMITY_WINDOW_CHARS_KEY: str = "proximity_window_chars"
COUNTRY_SCOPE_KEY: str = "country_scope"
COUNTRY_SCOPE_US: str = "US"
ADDRESS_DETECTION_ENABLED_KEY: str = "address_detection_enabled"
ADDRESS_DETECTION_PROVIDER_KEY: str = "address_detection_provider"
ADDRESS_DETECTION_PROVIDER_USADDRESS: str = "usaddress"
SPAN_CONFLICT_POLICY_KEY: str = "span_conflict_policy"
SPAN_CONFLICT_POLICY_PREFER_USADDRESS_LONGEST: str = "prefer_usaddress_longest"
SET_STR_TRUE_VALUES: set[str] = {"true", "1", "yes", "on", "t"}
PERSON_ENTITY: str = "PERSON"
PHONE_ENTITY: str = "PHONE_NUMBER"
EMAIL_ENTITY: str = "EMAIL_ADDRESS"
SSN_ENTITY: str = "US_SSN"
SSN_LAST4_ENTITY: str = "US_SSN_LAST4"
LOCATION_ENTITY: str = "LOCATION"
ORGANIZATION_ENTITY: str = "ORGANIZATION"
DATE_TIME_ENTITY: str = "DATE_TIME"
NRP_ENTITY: str = "NRP"

NAME_LABEL: str = "NAME"
PHONE_LABEL: str = "PHONE"
EMAIL_LABEL: str = "EMAIL"
SSN_LABEL: str = "SSN"
ADDRESS_LABEL: str = "ADDRESS"
DOB_LABEL: str = "DOB"

CANONICAL_NAME: str = "NAME"
CANONICAL_PHONE: str = "PHONE"
CANONICAL_EMAIL: str = "EMAIL"
CANONICAL_SSN: str = "SSN"
CANONICAL_ADDRESS: str = "ADDRESS"
CANONICAL_DOB: str = "DOB"
CANONICAL_CC_LAST4: str = "CC_LAST4"
TUPLE_STR_SUPPORTED_CANONICAL_REDACTION_CATEGORIES: tuple[str, ...] = (
    CANONICAL_NAME,
    CANONICAL_PHONE,
    CANONICAL_EMAIL,
    CANONICAL_SSN,
    CANONICAL_ADDRESS,
    CANONICAL_DOB,
    CANONICAL_CC_LAST4,
)

REPLACE_OPERATOR: str = "replace"
NEW_VALUE_KEY: str = "new_value"
DEFAULT_OPERATOR_KEY: str = "DEFAULT"
ENGINE_CACHE_NAMESPACE_PRESIDIO: str = "presidio_pii_redactor"
CACHE_IDENTITY_SEPARATOR: str = ","
USADDRESS_RECOGNIZER_NAME: str = "USAddressRecognizer"
USADDRESS_RECOGNIZER_IDENTIFIER: str = "usaddress"
USADDRESS_RECOGNIZER_SCORE: float = 0.99
RECOGNIZER_UNKNOWN: str = "unknown"
OVERLAP_ALLOWED_GAP: int = 0

SET_STR_WHITESPACE_AND_PUNCTUATION: set[str] = {" ", "\t", "\n", "\r", ",", "."}
REDACTION_RUNTIME_ERROR_MESSAGE: str = (
    "PII redaction failed while sanitizing text. "
    "Blocking request to avoid forwarding unsanitized content."
)

DICT_STR_MODEL_TO_PRESIDIO_ENTITY_MAPPING: dict[str, str] = {
    "PER": PERSON_ENTITY,
    PERSON_ENTITY: PERSON_ENTITY,
    "NORP": NRP_ENTITY,
    "FAC": LOCATION_ENTITY,
    "LOC": LOCATION_ENTITY,
    "GPE": LOCATION_ENTITY,
    LOCATION_ENTITY: LOCATION_ENTITY,
    "ORG": ORGANIZATION_ENTITY,
    ORGANIZATION_ENTITY: ORGANIZATION_ENTITY,
    "DATE": DATE_TIME_ENTITY,
    "TIME": DATE_TIME_ENTITY,
}

LIST_STR_NER_LABELS_TO_IGNORE: list[str] = [
    ORGANIZATION_ENTITY,
    "CARDINAL",
    "EVENT",
    "LANGUAGE",
    "LAW",
    "MONEY",
    "ORDINAL",
    "PERCENT",
    "PRODUCT",
    "QUANTITY",
    "WORK_OF_ART",
]

DICT_STR_NER_MODEL_CONFIGURATION: dict[str, Any] = {
    "model_to_presidio_entity_mapping": DICT_STR_MODEL_TO_PRESIDIO_ENTITY_MAPPING,
    "low_confidence_score_multiplier": 0.4,
    "low_score_entity_names": [],
    "labels_to_ignore": LIST_STR_NER_LABELS_TO_IGNORE,
}

SET_STR_USADDRESS_COMPONENT_LABELS: set[str] = {
    "AddressNumber",
    "AddressNumberPrefix",
    "AddressNumberSuffix",
    "BuildingName",
    "CornerOf",
    "IntersectionSeparator",
    "LandmarkName",
    "OccupancyIdentifier",
    "OccupancyType",
    "PlaceName",
    "StateName",
    "StreetName",
    "StreetNamePostDirectional",
    "StreetNamePostModifier",
    "StreetNamePostType",
    "StreetNamePreDirectional",
    "StreetNamePreModifier",
    "StreetNamePreType",
    "SubaddressIdentifier",
    "SubaddressType",
    "USPSBoxGroupID",
    "USPSBoxGroupType",
    "USPSBoxID",
    "USPSBoxType",
    "ZipCode",
}

SET_STR_USADDRESS_CONNECTOR_TOKENS: set[str] = {
    ",",
    ".",
    "and",
    "at",
    "of",
    "the",
}
SET_STR_USADDRESS_RECORD_DELIMITER_TOKENS: set[str] = {
    "address",
    "dob",
    "email",
    "phone",
    "ssn",
}
SET_STR_USADDRESS_RECORD_DELIMITER_LABELS: set[str] = {
    "Recipient",
    "StreetName",
    "SubaddressIdentifier",
    "SubaddressType",
}

SET_STR_USADDRESS_STREET_CORE_LABELS: set[str] = {
    "AddressNumber",
    "StreetName",
    "StreetNamePostType",
    "StreetNamePreType",
    "USPSBoxType",
    "USPSBoxID",
    "IntersectionSeparator",
    "LandmarkName",
    "BuildingName",
}

SET_STR_USADDRESS_LOCALITY_LABELS: set[str] = {
    "PlaceName",
    "StateName",
    "ZipCode",
}

SET_STR_SPAN_CONFLICT_ENTITIES: set[str] = {
    PERSON_ENTITY,
    LOCATION_ENTITY,
}
SET_STR_PROTECTED_CUSTOM_CONFLICT_ENTITIES: set[str] = {
    SSN_LAST4_ENTITY,
    PROVIDER_ENTITY_CREDIT_CARD_LAST4,
    PROVIDER_ENTITY_DOB_DATE,
}
SET_STR_PROTECTED_STRUCTURED_CONFLICT_ENTITIES: set[str] = {
    PHONE_ENTITY,
    EMAIL_ENTITY,
    SSN_ENTITY,
    SSN_LAST4_ENTITY,
    PROVIDER_ENTITY_CREDIT_CARD_LAST4,
    PROVIDER_ENTITY_DOB_DATE,
}

DICT_STR_PROVIDER_TO_CANONICAL_CATEGORY: dict[str, str] = {
    PERSON_ENTITY: CANONICAL_NAME,
    PHONE_ENTITY: CANONICAL_PHONE,
    EMAIL_ENTITY: CANONICAL_EMAIL,
    SSN_ENTITY: CANONICAL_SSN,
    SSN_LAST4_ENTITY: CANONICAL_SSN,
    PROVIDER_ENTITY_DOB_DATE: CANONICAL_DOB,
    PROVIDER_ENTITY_CREDIT_CARD_LAST4: CANONICAL_CC_LAST4,
    LOCATION_ENTITY: CANONICAL_ADDRESS,
    "FAC": CANONICAL_ADDRESS,
    "GPE": CANONICAL_ADDRESS,
    "LOC": CANONICAL_ADDRESS,
}

DICT_STR_CANONICAL_TO_PROVIDER_CATEGORIES: dict[str, list[str]] = {
    CANONICAL_NAME: [PERSON_ENTITY],
    CANONICAL_PHONE: [PHONE_ENTITY],
    CANONICAL_EMAIL: [EMAIL_ENTITY],
    CANONICAL_SSN: [SSN_ENTITY, SSN_LAST4_ENTITY],
    CANONICAL_ADDRESS: [LOCATION_ENTITY],
    CANONICAL_DOB: [PROVIDER_ENTITY_DOB_DATE],
    CANONICAL_CC_LAST4: [PROVIDER_ENTITY_CREDIT_CARD_LAST4],
}


class RegexOnlyNlpEngine(NlpEngine):
    """
    A 0MB NLP engine abstraction that completely bypasses spaCy.
    It returns empty NLP artifacts to satisfy the Presidio AnalyzerEngine requirement,
    enabling hyper-fast deterministic Regex execution without loading ML models.
    """

    def is_loaded(self) -> bool:
        """
        Signals to Presidio whether the NLP engine is fully initialized and ready.

        Args:
            None

        Returns:
            Always returns True, as this zero-dependency engine has no loading phase.
        """
        # Return true normally as there is no model to load.
        return True

    def load(self) -> None:
        """
        Attempts to load the underlying NLP machine learning model.

        Args:
            None

        Returns:
            None (intentionally implemented as a no-op to bypass ML model loading).
        """
        # Normal execution bypass. No ML model loading necessary.
        pass

    def is_stopword(self, word: str, language: str) -> bool:
        """
        Checks if a given word is a recognized grammatical stopword (e.g., "the", "and").

        Args:
            word: The string token to evaluate.
            language: The two-letter locale code.

        Returns:
            Always returns False, as this engine does not perform grammatical analysis.
        """
        # Normal return, bypass stopword analysis.
        return False

    def is_punct(self, word: str, language: str) -> bool:
        """
        Checks if a given word is pure punctuation.

        Args:
            word: The string token to evaluate.
            language: The two-letter locale code.

        Returns:
            Always returns False, as this engine does not perform grammatical tokenization.
        """
        # Normal return, bypass punctuation analysis.
        return False

    def get_nlp(self) -> Any:
        """
        Retrieves the underlying native NLP engine representation (typically a spaCy model object).

        Args:
            None

        Returns:
            Always returns None, as there is no underlying ML object.
        """
        # Normal return, bypass native NLP object retrieval.
        return None

    def get_supported_entities(self) -> list[str]:
        """
        Retrieves the list of PII entities this NLP engine can detect natively using ML context.

        Args:
            None

        Returns:
            Always returns an empty list. Since it is a regex-only implementation,
            it relies entirely on the Analyzer's PatternRecognizer components instead.
        """
        # Normal return, we don't natively extract any ML entities.
        return []

    def get_supported_languages(self) -> list[str]:
        """
        Retrieves the list of languages this NLP engine supports for context analysis.

        Args:
            None

        Returns:
            Always returns an empty list, as no native NLP parsing is supported.
        """
        # Normal return, no language ML dictionary support.
        return []

    def process_text(self, text: str, language: str) -> NlpArtifacts:
        """
        Performs natural language tokenization, lemmatization, and entity extraction on the text.

        Args:
            text: The raw input string to parse.
            language: The two-letter locale code defining the language dictionaries to use.

        Returns:
            An NlpArtifacts object populated entirely with empty lists.
            This safely tricks the caller into believing ML processing succeeded,
            While completely bypassing the compute overhead.
            (Tokens expects a spaCy Doc, but we bypass spaCy, so we type ignore None).
        """
        # Normal execution bypass. Return empty data structures.
        return NlpArtifacts(
            entities=[],
            tokens=cast(Any, None),
            tokens_indices=[],
            lemmas=[],
            nlp_engine=self,
            language=language,
        )

    def process_batch(
        self, texts: Iterable[str], language: str, *args: Any, **kwargs: Any
    ) -> Iterator[tuple[str, NlpArtifacts]]:
        for text in texts:
            yield text, self.process_text(text, language)


def _register_custom_recognizer_plumbing(
    analyzer_engine: AnalyzerEngine,
    tuple_custom_recognizer_registration_configs: tuple[
        RecognizerRegistrationConfig, ...
    ],
    str_language: str,
) -> None:
    """
    Registers custom recognizers in the analyzer registry.

    Args:
        analyzer_engine: Analyzer engine instance whose registry should receive recognizers.
        tuple_custom_recognizer_registration_configs: Ordered tuple of enabled middleware
                                                      recognizer registration settings.
        str_language: Language code used by registered recognizers.

    Returns:
        None
    """
    if not tuple_custom_recognizer_registration_configs:
        # Early return because no custom recognizer rules were enabled for this engine.
        return
    tuple_str_registered_names: tuple[str, ...] = (
        CustomRecognizerFactory.register_pattern_recognizers(
            recognizer_registry=analyzer_engine.registry,
            tuple_recognizer_registration_configs=tuple_custom_recognizer_registration_configs,
            str_language=str_language,
        )
    )
    logger.info(
        "Registered custom recognizer plumbing rules: %s",
        tuple_str_registered_names,
    )
    # Normal return after registry registration completes.
    return


def _get_default_confidence_threshold_for_rule(str_recognizer_rule_name: str) -> float:
    """
    Returns the middleware-config default confidence threshold for one recognizer rule.

    Args:
        str_recognizer_rule_name: Recognizer rule name used in middleware config.

    Returns:
        Default confidence threshold for the requested rule, or the shared
        recognizer fallback threshold when the rule is unknown.
    """
    if str_recognizer_rule_name == RECOGNIZER_RULE_KEY_SSN_LAST4:
        # Early return with the SSN last-4 default threshold.
        return DEFAULT_SSN_LAST4_CONFIDENCE_THRESHOLD
    if str_recognizer_rule_name == RECOGNIZER_RULE_KEY_CC_LAST4:
        # Early return with the card last-4 default threshold.
        return DEFAULT_CC_LAST4_CONFIDENCE_THRESHOLD
    if str_recognizer_rule_name == RECOGNIZER_RULE_KEY_DOB:
        # Early return with the DOB default threshold.
        return DEFAULT_DOB_CONFIDENCE_THRESHOLD
    # Normal return with the generic recognizer threshold fallback.
    return DEFAULT_RECOGNIZER_CONFIDENCE_THRESHOLD


def _get_default_context_terms_for_rule(
    str_recognizer_rule_name: str,
) -> tuple[str, ...]:
    """
    Returns the middleware-config default positive context terms for one rule.

    Args:
        str_recognizer_rule_name: Recognizer rule name used in middleware config.

    Returns:
        Tuple of default positive context terms for the requested rule, or an
        empty tuple when the rule is unknown.
    """
    if str_recognizer_rule_name == RECOGNIZER_RULE_KEY_SSN_LAST4:
        # Early return with SSN last-4 default positive context terms.
        return tuple(LIST_STR_DEFAULT_SSN_LAST4_CONTEXT_TERMS)
    if str_recognizer_rule_name == RECOGNIZER_RULE_KEY_CC_LAST4:
        # Early return with card last-4 default positive context terms.
        return tuple(LIST_STR_DEFAULT_CC_LAST4_CONTEXT_TERMS)
    if str_recognizer_rule_name == RECOGNIZER_RULE_KEY_DOB:
        # Early return with DOB default positive context terms.
        return tuple(LIST_STR_DEFAULT_DOB_CONTEXT_TERMS)
    # Normal return with no default context terms for unknown rules.
    return ()


def _get_min_positive_context_matches_for_rule(str_recognizer_rule_name: str) -> int:
    """
    Returns the minimum positive context matches required for one recognizer rule.

    Args:
        str_recognizer_rule_name: Recognizer rule name used in middleware config.

    Returns:
        Internal minimum positive context match count for the requested rule.
    """
    # All current middleware recognizer rules use the visible one-anchor
    # contract documented in README/design docs: at least one configured
    # positive context term near the candidate span is enough to qualify the
    # regex match. Hidden multi-anchor requirements would make custom
    # `context_terms` lists surprising and can silently disable detection.
    return 1


@cache
def get_cached_analyzer_engine(
    str_language: str,
    str_profile: str,
    str_engine_cache_identity_signature: str = "",
    tuple_custom_recognizer_registration_configs: tuple[
        RecognizerRegistrationConfig, ...
    ] = (),
) -> AnalyzerEngine:
    """
    Caches the NLP models (spaCy) and Presidio recognizers in memory based on the selected runtime profile.
    This guarantees that the models are only loaded exactly once per language and profile
    for the lifespan of the application process.

    Args:
        str_language: The two-letter language code (e.g., 'en') to load recognizers for.
        str_profile: The selected runtime profile ('large_spacy_500MB', 'small_spacy_15MB', or 'micro_regex').
        str_engine_cache_identity_signature: Deterministic cache-partition signature
                                             representing recognizer/config identity.
        tuple_custom_recognizer_registration_configs: Ordered tuple of enabled
                                                      middleware recognizer registration settings.

    Returns:
        An instantiated presidio_analyzer.AnalyzerEngine ready to perform PII detection.
    """
    if str_profile == PROFILE_REGEX:
        # 0MB Footprint, ~0ms latency. Bypasses NLP entirely.
        logger.info("Initialized Presidio Analyzer with %s engine.", PROFILE_REGEX)
        analyzer_engine: AnalyzerEngine = AnalyzerEngine(
            nlp_engine=RegexOnlyNlpEngine(), supported_languages=[str_language]
        )
        _register_custom_recognizer_plumbing(
            analyzer_engine=analyzer_engine,
            tuple_custom_recognizer_registration_configs=tuple_custom_recognizer_registration_configs,
            str_language=str_language,
        )
        # Normal execution path for Regex profile.
        return analyzer_engine

    # Either large_spacy_500MB or small_spacy_15MB
    str_spacy_model: str = (
        SPACY_MODEL_LARGE if str_profile == PROFILE_LARGE else SPACY_MODEL_SMALL
    )

    dict_nlp_config: dict[str, Any] = {
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": str_language, "model_name": str_spacy_model}],
        "ner_model_configuration": deepcopy(DICT_STR_NER_MODEL_CONFIGURATION),
    }
    nlp_engine: NlpEngine = NlpEngineProvider(
        nlp_configuration=dict_nlp_config
    ).create_engine()
    logger.info(
        "Initialized Presidio Analyzer with %s (%s).",
        str_profile,
        str_spacy_model,
    )

    analyzer_engine: AnalyzerEngine = AnalyzerEngine(
        nlp_engine=nlp_engine, supported_languages=[str_language]
    )
    _register_custom_recognizer_plumbing(
        analyzer_engine=analyzer_engine,
        tuple_custom_recognizer_registration_configs=tuple_custom_recognizer_registration_configs,
        str_language=str_language,
    )

    # Normal execution path for ML profiles.
    return analyzer_engine


@cache
def get_cached_anonymizer_engine() -> AnonymizerEngine:
    """
    Caches the anonymizer engine which handles redaction substitution.

    Args:
        None

    Returns:
        An instantiated presidio_anonymizer.AnonymizerEngine.
    """
    # Normal cache return for the Anonymizer singleton.
    return AnonymizerEngine()


class PiiRedactor(BaseRedactorLayer):
    """
    Encapsulates Presidio detection and redaction logic.
    Uses module-level cached engines to optimize performance.
    """

    @property
    def str_engine_cache_namespace(self) -> str:
        """
        Returns cache namespace for shared Presidio engine entries.

        Args:
            None

        Returns:
            Stable namespace token used to partition Presidio cache entries from
            other potential redaction engines.
        """
        # Normal return with Presidio cache namespace.
        return ENGINE_CACHE_NAMESPACE_PRESIDIO

    @property
    def bool_uses_shared_engine_cache(self) -> bool:
        """
        Signals whether this implementation uses reusable shared engine instances.

        Args:
            None

        Returns:
            Always returns True because PiiRedactor uses module-level cached
            AnalyzerEngine and AnonymizerEngine instances.
        """
        # Normal return because shared engine caching is enabled for Presidio.
        return True

    @property
    def tuple_str_engine_cache_identity(self) -> tuple[str, ...]:
        """
        Returns deterministic cache identity components for this redactor instance.

        Args:
            None

        Returns:
            Tuple of string components representing the effective engine and
            recognizer-facing configuration that influences cache reuse safety.
        """
        list_str_sorted_supported_entities: list[str] = sorted(
            self.list_str_supported_entities
        )
        str_supported_entities_signature: str = CACHE_IDENTITY_SEPARATOR.join(
            list_str_sorted_supported_entities
        )
        str_custom_recognizer_rules_signature: str = CACHE_IDENTITY_SEPARATOR.join(
            self.tuple_str_enabled_custom_recognizer_rules
        )
        list_str_custom_recognizer_config_tokens: list[str] = []
        # Loop through custom recognizer registration configs and build stable cache tokens.
        for (
            recognizer_registration_config
        ) in self.tuple_custom_recognizer_registration_configs:
            list_str_custom_recognizer_config_tokens.append(
                f"{recognizer_registration_config.str_recognizer_rule_name}|"
                f"{recognizer_registration_config.float_confidence_threshold}|"
                f"{recognizer_registration_config.int_proximity_window_chars}|"
                f"{recognizer_registration_config.int_min_positive_context_matches}|"
                f"{CACHE_IDENTITY_SEPARATOR.join(recognizer_registration_config.tuple_str_context_terms)}|"
                f"{CACHE_IDENTITY_SEPARATOR.join(recognizer_registration_config.tuple_str_negative_context_terms)}"
            )
        str_custom_recognizer_config_signature: str = CACHE_IDENTITY_SEPARATOR.join(
            list_str_custom_recognizer_config_tokens
        )
        # Normal return with stable cache identity components for shared engine reuse.
        return (
            self.str_engine_cache_namespace,
            self.str_language,
            self.str_detection_profile,
            self.str_nlp_profile,
            self.str_country_scope,
            self.str_address_detection_provider,
            str(self.bool_address_detection_enabled).lower(),
            str_supported_entities_signature,
            str_custom_recognizer_rules_signature,
            str_custom_recognizer_config_signature,
        )

    def __init__(self, dict_config: dict[str, Any] | None = None) -> None:
        """
        Initializes the Presidio redactor wrapper with configuration overrides.

        Args:
            dict_config: Optional dictionary containing settings for language, allowed entities,
                         redaction labels, label mapping, and middleware detection profile.

        Returns:
            None
        """
        if dict_config is None:
            dict_config = {}

        self.str_detection_profile: str = self._normalize_detection_profile(
            dict_config.get(DETECTION_PROFILE_KEY, DETECTION_PROFILE_BALANCED)
        )
        self.str_nlp_profile: str = self._resolve_runtime_profile(
            self.str_detection_profile
        )
        self.str_language: str = str(dict_config.get(LANGUAGE_KEY, DEFAULT_EN))
        self.str_default_redaction_label: str = str(
            dict_config.get(DEFAULT_REDACTION_LABEL_KEY, DEFAULT_REDACTED_TEXT)
        )
        list_raw_allowed_entities: list[str] = list(
            dict_config.get(ALLOWED_ENTITIES_KEY, [])
        )
        self.list_str_allowed_entities: list[str] = [
            self._map_category_to_canonical(str_entity)
            for str_entity in list_raw_allowed_entities
        ]
        self.str_country_scope: str = str(
            dict_config.get(COUNTRY_SCOPE_KEY, COUNTRY_SCOPE_US)
        ).upper()
        self.str_address_detection_provider: str = str(
            dict_config.get(
                ADDRESS_DETECTION_PROVIDER_KEY,
                ADDRESS_DETECTION_PROVIDER_USADDRESS,
            )
        ).lower()
        self.str_span_conflict_policy: str = str(
            dict_config.get(
                SPAN_CONFLICT_POLICY_KEY,
                SPAN_CONFLICT_POLICY_PREFER_USADDRESS_LONGEST,
            )
        ).lower()
        self.bool_address_detection_enabled: bool = self._coerce_bool(
            dict_config.get(ADDRESS_DETECTION_ENABLED_KEY, True),
            True,
            ADDRESS_DETECTION_ENABLED_KEY,
        )

        dict_str_default_map: dict[str, str] = {
            CANONICAL_NAME: NAME_LABEL,
            CANONICAL_PHONE: PHONE_LABEL,
            CANONICAL_EMAIL: EMAIL_LABEL,
            CANONICAL_SSN: SSN_LABEL,
            CANONICAL_ADDRESS: ADDRESS_LABEL,
            CANONICAL_DOB: DOB_LABEL,
            CANONICAL_CC_LAST4: CANONICAL_CC_LAST4,
        }
        dict_entity_label_map: dict[str, str] = dict(
            dict_config.get(ENTITY_LABEL_MAP_KEY, dict_str_default_map)
        )
        self.dict_str_entity_label_map: dict[str, str] = (
            self._normalize_entity_label_map(
                dict_entity_label_map=dict_entity_label_map,
                dict_default_entity_label_map=dict_str_default_map,
            )
        )
        raw_redaction_recognizers_settings: Any = dict_config.get(
            REDACTION_RECOGNIZERS_KEY,
            {},
        )
        # Normalize middleware-level recognizer settings through the same typed
        # config model used by YAML loading so direct-constructor behavior stays
        # aligned with the library's documented middleware contract.
        self.dict_redaction_recognizers_settings: dict[str, Any] = (
            self._normalize_runtime_redaction_recognizers_settings(
                raw_redaction_recognizers_settings=raw_redaction_recognizers_settings
            )
        )
        self.tuple_str_enabled_custom_recognizer_rules: tuple[str, ...] = (
            self._resolve_enabled_custom_recognizer_rule_keys(
                dict_redaction_recognizers_settings=self.dict_redaction_recognizers_settings
            )
        )
        self.tuple_custom_recognizer_registration_configs: tuple[
            RecognizerRegistrationConfig, ...
        ] = self._build_custom_recognizer_registration_configs(
            dict_redaction_recognizers_settings=self.dict_redaction_recognizers_settings,
            tuple_str_enabled_custom_recognizer_rules=self.tuple_str_enabled_custom_recognizer_rules,
        )

        # Redaction scope is derived only from supported canonical categories and
        # allow-list rules. Entity-label mapping is labels-only.
        self.list_str_supported_canonical_entities: list[str] = [
            str_category
            for str_category in TUPLE_STR_SUPPORTED_CANONICAL_REDACTION_CATEGORIES
            if str_category not in self.list_str_allowed_entities
        ]
        self.list_str_supported_entities: list[str] = [
            str_provider_entity
            for str_category in self.list_str_supported_canonical_entities
            for str_provider_entity in DICT_STR_CANONICAL_TO_PROVIDER_CATEGORIES.get(
                str_category, [str_category]
            )
        ]
        self.list_str_supported_entities = self._deduplicate_category_tokens(
            self.list_str_supported_entities
        )
        self.list_str_conflict_protection_entities: list[str] = (
            self._resolve_conflict_protection_entities(
                list_str_allowed_entities=self.list_str_allowed_entities
            )
        )
        self.list_str_analysis_entities: list[str] = self._deduplicate_category_tokens(
            self.list_str_supported_entities
            + self.list_str_conflict_protection_entities
        )
        self.bool_usaddress_detection_active: bool = (
            self.bool_address_detection_enabled
            and self.str_country_scope == COUNTRY_SCOPE_US
            and self.str_address_detection_provider
            == ADDRESS_DETECTION_PROVIDER_USADDRESS
            and usaddress is not None
            and self.str_nlp_profile != PROFILE_REGEX
        )
        if (
            self.bool_address_detection_enabled
            and self.str_address_detection_provider
            == ADDRESS_DETECTION_PROVIDER_USADDRESS
            and usaddress is None
        ):
            logger.warning(
                "Address detection provider is set to usaddress but the package is not installed."
            )
        self._log_profile_readiness_warnings()

    @staticmethod
    def _resolve_conflict_protection_entities(
        list_str_allowed_entities: list[str],
    ) -> list[str]:
        """
        Resolves pass-through provider entities that still need overlap protection.

        Args:
            list_str_allowed_entities: Canonical categories configured for pass-through.

        Returns:
            Provider entities that should still be analyzed so they can defend
            their spans against parser-backed ADDRESS overreach, even though
            they will not be redacted.
        """
        list_str_conflict_protection_entities: list[str] = []
        # Loop through allow-listed canonical categories and retain only the
        # protected structured provider entities needed for overlap resolution.
        for str_category in list_str_allowed_entities:
            for str_provider_entity in DICT_STR_CANONICAL_TO_PROVIDER_CATEGORIES.get(
                str_category, []
            ):
                if (
                    str_provider_entity
                    not in SET_STR_PROTECTED_STRUCTURED_CONFLICT_ENTITIES
                ):
                    continue
                list_str_conflict_protection_entities.append(str_provider_entity)
        # Normal return with pass-through entities still needed for conflict protection.
        return list_str_conflict_protection_entities

    @staticmethod
    def _normalize_runtime_redaction_recognizers_settings(
        raw_redaction_recognizers_settings: Any,
    ) -> dict[str, Any]:
        """
        Normalizes runtime recognizer settings using the shared typed config model.

        Args:
            raw_redaction_recognizers_settings: Raw recognizer settings value supplied to PiiRedactor.

        Returns:
            Normalized recognizer settings with the same coercion/default rules used by YAML loading.
        """
        pii_redaction_settings: PiiRedactionSettingsModel = (
            PiiRedactionSettingsModel.model_validate(
                {REDACTION_RECOGNIZERS_KEY: raw_redaction_recognizers_settings}
            )
        )
        dict_normalized_redaction_recognizers_settings: dict[str, Any] = (
            pii_redaction_settings.redaction_recognizers.model_dump()
        )
        # Normal return with shared-model normalized recognizer settings.
        return dict_normalized_redaction_recognizers_settings

    @staticmethod
    def _normalize_detection_profile(value: Any) -> str:
        """
        Normalizes incoming detection profile values into known canonical profile names.

        Args:
            value: Raw detection profile token from middleware configuration.

        Returns:
            A normalized canonical detection profile value.
        """
        str_detection_profile: str = str(value).strip().lower()
        if str_detection_profile in DICT_STR_DETECTION_PROFILE_TO_RUNTIME_PROFILE:
            # Normal return with validated detection profile token.
            return str_detection_profile
        logger.warning(
            INVALID_DETECTION_PROFILE_FALLBACK_WARNING_MESSAGE,
            value,
            DETECTION_PROFILE_BALANCED,
        )
        # Early return because unknown profile values fall back to balanced.
        return DETECTION_PROFILE_BALANCED

    def _resolve_enabled_custom_recognizer_rule_keys(
        self, dict_redaction_recognizers_settings: dict[str, Any]
    ) -> tuple[str, ...]:
        """
        Resolves enabled middleware recognizer rule keys for analyzer registration.

        Args:
            dict_redaction_recognizers_settings: Normalized middleware recognizer settings block.

        Returns:
            Ordered tuple of enabled recognizer rule keys.
        """
        list_str_enabled_rule_keys: list[str] = []
        # Loop through known middleware rule keys and retain only enabled rules.
        for str_rule_key in TUPLE_STR_RECOGNIZER_RULE_KEYS:
            dict_rule_settings: dict[str, Any] | None = (
                dict_redaction_recognizers_settings.get(str_rule_key)
            )
            if not isinstance(dict_rule_settings, dict):
                continue
            bool_enabled: bool = self._coerce_bool(
                dict_rule_settings.get(RECOGNIZER_ENABLED_KEY, False),
                False,
                f"{REDACTION_RECOGNIZERS_KEY}.{str_rule_key}.{RECOGNIZER_ENABLED_KEY}",
            )
            if not bool_enabled:
                continue
            list_str_enabled_rule_keys.append(str_rule_key)
        tuple_str_enabled_rule_keys: tuple[str, ...] = tuple(list_str_enabled_rule_keys)
        # Normal return with enabled custom recognizer rule keys.
        return tuple_str_enabled_rule_keys

    def _build_custom_recognizer_registration_configs(
        self,
        dict_redaction_recognizers_settings: dict[str, Any],
        tuple_str_enabled_custom_recognizer_rules: tuple[str, ...],
    ) -> tuple[RecognizerRegistrationConfig, ...]:
        """
        Builds hashable runtime registration settings for enabled custom recognizer rules.

        Args:
            dict_redaction_recognizers_settings: Normalized middleware recognizer settings block.
            tuple_str_enabled_custom_recognizer_rules: Ordered tuple of enabled recognizer rule names.

        Returns:
            Ordered tuple of runtime recognizer registration settings.
        """
        raw_proximity_window_chars: Any = dict_redaction_recognizers_settings.get(
            PROXIMITY_WINDOW_CHARS_KEY,
            DEFAULT_PROXIMITY_WINDOW_CHARS,
        )
        int_proximity_window_chars: int = DEFAULT_PROXIMITY_WINDOW_CHARS
        try:
            # Runtime settings normally arrive here already typed by the shared
            # Pydantic middleware model. This fallback conversion keeps the
            # helper defensive if a direct dict reaches it before typing.
            int_proximity_window_chars = max(1, int(raw_proximity_window_chars))
        except (TypeError, ValueError):
            int_proximity_window_chars = DEFAULT_PROXIMITY_WINDOW_CHARS
        list_recognizer_registration_configs: list[RecognizerRegistrationConfig] = []

        # Loop through enabled recognizer rule names and build registration settings.
        for str_recognizer_rule_name in tuple_str_enabled_custom_recognizer_rules:
            dict_rule_settings: dict[str, Any] | None = (
                dict_redaction_recognizers_settings.get(str_recognizer_rule_name)
            )
            if not isinstance(dict_rule_settings, dict):
                continue
            float_default_confidence_threshold: float = (
                _get_default_confidence_threshold_for_rule(
                    str_recognizer_rule_name=str_recognizer_rule_name
                )
            )
            raw_confidence_threshold: Any = dict_rule_settings.get(
                RECOGNIZER_CONFIDENCE_THRESHOLD_KEY,
                float_default_confidence_threshold,
            )
            float_confidence_threshold: float = float_default_confidence_threshold
            try:
                # Runtime settings normally arrive here already typed by the shared
                # Pydantic middleware model. This fallback conversion keeps the
                # clamp logic aligned with numeric-string inputs if an untyped
                # dict bypasses the standard config normalization path.
                float_confidence_threshold = max(
                    0.0,
                    min(1.0, float(raw_confidence_threshold)),
                )
            except (TypeError, ValueError):
                float_confidence_threshold = float_default_confidence_threshold
            bool_context_terms_field_present: bool = (
                RECOGNIZER_CONTEXT_TERMS_KEY in dict_rule_settings
            )
            tuple_str_context_terms: tuple[str, ...] = tuple(
                dict_rule_settings.get(RECOGNIZER_CONTEXT_TERMS_KEY, ())
            )
            tuple_str_negative_context_terms: tuple[str, ...] = tuple(
                dict_rule_settings.get(RECOGNIZER_NEGATIVE_CONTEXT_TERMS_KEY, ())
            )
            if not bool_context_terms_field_present:
                # Only missing fields inherit defaults. An explicit empty list
                # is a meaningful config choice and must remain empty instead of
                # silently re-enabling the default positive context anchors.
                tuple_str_context_terms = _get_default_context_terms_for_rule(
                    str_recognizer_rule_name=str_recognizer_rule_name
                )
            recognizer_registration_config: RecognizerRegistrationConfig = (
                RecognizerRegistrationConfig(
                    str_recognizer_rule_name=str_recognizer_rule_name,
                    float_confidence_threshold=float_confidence_threshold,
                    int_proximity_window_chars=int_proximity_window_chars,
                    int_min_positive_context_matches=_get_min_positive_context_matches_for_rule(
                        str_recognizer_rule_name=str_recognizer_rule_name
                    ),
                    tuple_str_context_terms=tuple_str_context_terms,
                    tuple_str_negative_context_terms=tuple_str_negative_context_terms,
                )
            )
            list_recognizer_registration_configs.append(recognizer_registration_config)
        tuple_recognizer_registration_configs: tuple[
            RecognizerRegistrationConfig, ...
        ] = tuple(list_recognizer_registration_configs)
        # Normal return with hashable runtime recognizer registration settings.
        return tuple_recognizer_registration_configs

    @staticmethod
    def _resolve_runtime_profile(str_detection_profile: str) -> str:
        """
        Resolves middleware-level detection profile into internal runtime engine profile.

        Args:
            str_detection_profile: Canonical middleware detection profile token.

        Returns:
            Runtime profile token used by Presidio engine construction.
        """
        str_runtime_profile: str = DICT_STR_DETECTION_PROFILE_TO_RUNTIME_PROFILE.get(
            str_detection_profile,
            PROFILE_SMALL,
        )
        # Normal return with resolved runtime profile.
        return str_runtime_profile

    def _log_profile_readiness_warnings(self) -> None:
        """
        Logs actionable warnings when the selected detection profile runtime assets are missing.

        Args:
            None

        Returns:
            None
        """
        if self.str_nlp_profile == PROFILE_REGEX:
            # Early return because regex profile has no spaCy model dependency.
            return
        str_required_spacy_model: str = (
            SPACY_MODEL_LARGE
            if self.str_nlp_profile == PROFILE_LARGE
            else SPACY_MODEL_SMALL
        )
        bool_model_available: bool = (
            importlib.util.find_spec(str_required_spacy_model) is not None
        )
        if bool_model_available:
            # Normal return because required spaCy model dependency is present.
            return
        logger.warning(
            "Detection profile '%s' requires spaCy model '%s', which is not installed in this runtime.",
            self.str_detection_profile,
            str_required_spacy_model,
        )
        # Normal return after logging readiness warning.
        return

    @staticmethod
    def _map_category_to_canonical(str_category: str) -> str:
        """
        Normalizes provider or canonical categories into canonical middleware categories.

        Args:
            str_category: Raw category token from configuration or provider output.

        Returns:
            Canonical category string when mapping is known, otherwise the normalized
            upper-case input category.
        """
        str_normalized_category: str = str(str_category).strip().upper()
        str_canonical_category: str = DICT_STR_PROVIDER_TO_CANONICAL_CATEGORY.get(
            str_normalized_category,
            str_normalized_category,
        )
        # Normal return with canonicalized category token.
        return str_canonical_category

    def _normalize_entity_label_map(
        self,
        dict_entity_label_map: dict[str, str],
        dict_default_entity_label_map: dict[str, str],
    ) -> dict[str, str]:
        """
        Normalizes config-provided entity label map keys into canonical categories.

        Args:
            dict_entity_label_map: Raw entity label map from middleware configuration.
            dict_default_entity_label_map: Default canonical label map used as fallback.

        Returns:
            Canonical-category keyed label map. Returns defaults when normalization
            produces no usable keys.
        """
        dict_normalized_map: dict[str, str] = dict(dict_default_entity_label_map)

        # Loop through config map entries and canonicalize each category key.
        for str_category, str_label in dict_entity_label_map.items():
            str_canonical_category: str = self._map_category_to_canonical(str_category)
            if (
                not str_canonical_category
                or str_canonical_category
                not in TUPLE_STR_SUPPORTED_CANONICAL_REDACTION_CATEGORIES
            ):
                continue
            dict_normalized_map[str_canonical_category] = str(str_label).strip()

        # Normal return with defaults overlaid by canonicalized custom map values.
        return dict_normalized_map

    @staticmethod
    def _deduplicate_category_tokens(list_str_categories: list[str]) -> list[str]:
        """
        Deduplicates category tokens while preserving first-seen order.

        Args:
            list_str_categories: Raw category sequence that may include duplicates.

        Returns:
            Ordered unique category sequence.
        """
        set_str_seen_categories: set[str] = set()
        list_str_unique_categories: list[str] = []

        # Loop through categories once and retain first occurrence ordering.
        for str_category in list_str_categories:
            if str_category in set_str_seen_categories:
                continue
            set_str_seen_categories.add(str_category)
            list_str_unique_categories.append(str_category)
        # Normal return with de-duplicated ordered categories.
        return list_str_unique_categories

    @staticmethod
    def _coerce_bool(value: Any, bool_default: bool, str_setting_name: str) -> bool:
        """
        Converts arbitrary configuration values into deterministic booleans.

        Args:
            value: Raw value from configuration settings.
            bool_default: Fallback value when value is missing or unrecognized.
            str_setting_name: Human-readable config key used in warning logs.

        Returns:
            True or False based on parsed value semantics. Falls back to bool_default when needed.
        """
        if isinstance(value, bool):
            # Normal return when the value is already a boolean type.
            return value
        if isinstance(value, str):
            bool_parsed_value: bool = value.strip().lower() in SET_STR_TRUE_VALUES
            # Normal return after parsing string tokens into boolean semantics.
            return bool_parsed_value
        if value is None:
            # Early return because no value was supplied.
            return bool_default
        logger.warning(
            BOOL_TRUTHINESS_FALLBACK_WARNING_MESSAGE,
            str_setting_name,
            type(value).__name__,
        )
        bool_fallback_value: bool = bool(value)
        # Normal return using Python truthiness as a fallback conversion.
        return bool_fallback_value

    @staticmethod
    def _spans_overlap(
        int_start_one: int, int_end_one: int, int_start_two: int, int_end_two: int
    ) -> bool:
        """
        Determines whether two character spans overlap.

        Args:
            int_start_one: Inclusive start index for the first span.
            int_end_one: Exclusive end index for the first span.
            int_start_two: Inclusive start index for the second span.
            int_end_two: Exclusive end index for the second span.

        Returns:
            True when spans overlap by at least one character; otherwise False.
        """
        bool_overlaps: bool = not (
            int_end_one <= int_start_two or int_end_two <= int_start_one
        )
        # Normal return conveying overlap status.
        return bool_overlaps

    def _map_usaddress_tokens_to_offsets(
        self, str_text: str, list_tuple_str_token_label: list[tuple[str, str]]
    ) -> list[tuple[str, str, int, int]]:
        """
        Maps sequential usaddress token outputs back onto character offsets in the source text.

        Args:
            str_text: Full original text being analyzed.
            list_tuple_str_token_label: Ordered list of (token, label) pairs returned by usaddress.parse().

        Returns:
            A list of tuples (token, label, start, end) with absolute character indices.
            Tokens that cannot be aligned are skipped.
        """
        list_tuple_mapped_tokens: list[tuple[str, str, int, int]] = []
        int_cursor: int = 0

        # Loop through each parsed token in sequence and map it to the next matching text location.
        for str_token, str_label in list_tuple_str_token_label:
            int_match_start: int = str_text.find(str_token, int_cursor)
            if int_match_start == -1:
                continue
            int_match_end: int = int_match_start + len(str_token)
            list_tuple_mapped_tokens.append(
                (str_token, str_label, int_match_start, int_match_end)
            )
            int_cursor = int_match_end

        # Normal return containing all successfully mapped tokens with their offsets.
        return list_tuple_mapped_tokens

    def _is_valid_usaddress_span(
        self, set_str_span_labels: set[str], int_span_start: int, int_span_end: int
    ) -> bool:
        """
        Validates whether a candidate usaddress span has sufficient address structure.

        Args:
            set_str_span_labels: Unique usaddress labels present in the candidate span.
            int_span_start: Inclusive character start index of the candidate span.
            int_span_end: Exclusive character end index of the candidate span.

        Returns:
            True when the candidate has address-like structural labels, otherwise False.
        """
        int_span_length: int = int_span_end - int_span_start
        bool_has_street_core: bool = bool(
            set_str_span_labels & SET_STR_USADDRESS_STREET_CORE_LABELS
        )
        bool_has_locality: bool = bool(
            set_str_span_labels & SET_STR_USADDRESS_LOCALITY_LABELS
        )
        bool_has_pobox_pair: bool = {
            "USPSBoxType",
            "USPSBoxID",
        }.issubset(set_str_span_labels)
        bool_is_valid: bool = int_span_length > 5 and (
            bool_has_pobox_pair
            or (bool_has_street_core and bool_has_locality)
            or (
                bool_has_street_core
                and "AddressNumber" in set_str_span_labels
                and "StreetName" in set_str_span_labels
            )
        )
        # Normal return indicating whether this candidate span should be treated as a US address.
        return bool_is_valid

    def _is_usaddress_hard_break_token(self, str_token: str, str_label: str) -> bool:
        """
        Determines whether a parsed usaddress token should forcibly terminate the current address run.

        Args:
            str_token: Raw token emitted by usaddress.parse(), including any trailing punctuation.
            str_label: usaddress semantic label assigned to the token.

        Returns:
            True when the token is a structured-record delimiter or clearly non-address token
            which should split address parsing; otherwise False.
        """
        str_normalized_token: str = str_token.strip().strip(",:;.").lower()
        bool_contains_email_marker: bool = "@" in str_token
        bool_is_record_delimiter: bool = (
            str_label in SET_STR_USADDRESS_RECORD_DELIMITER_LABELS
            and str_normalized_token in SET_STR_USADDRESS_RECORD_DELIMITER_TOKENS
        )
        # Normal return indicating whether the current token must terminate the address run.
        return bool_contains_email_marker or bool_is_record_delimiter

    def _collect_usaddress_location_results(
        self, str_text: str
    ) -> list[RecognizerResult]:
        """
        Parses US addresses from raw text and converts them into LOCATION recognizer results.

        Args:
            str_text: Full original text to parse.

        Returns:
            A list of RecognizerResult objects tagged as LOCATION from US parser-backed detection.
            Returns an empty list when disabled, unavailable, or no valid spans are found.
        """
        if not self.bool_usaddress_detection_active:
            # Early return because US address parser detection is disabled for this configuration.
            return []
        if usaddress is None:
            # Early return because the optional usaddress dependency is unavailable.
            return []

        try:
            list_tuple_str_token_label: list[tuple[str, str]] = list(
                usaddress.parse(str_text)
            )
        except Exception as exception:
            logger.debug("usaddress parse failed with exception: %s", exception)
            # Early return because token parsing failed.
            return []

        if not list_tuple_str_token_label:
            # Early return because parsing produced no token labels.
            return []

        list_tuple_mapped_tokens: list[tuple[str, str, int, int]] = (
            self._map_usaddress_tokens_to_offsets(
                str_text=str_text, list_tuple_str_token_label=list_tuple_str_token_label
            )
        )
        list_tuple_candidate_spans: list[tuple[int, int]] = []
        int_run_start: int | None = None
        int_run_end: int | None = None
        set_str_run_labels: set[str] = set()

        # Loop through mapped tokens and aggregate contiguous address-like runs.
        for (
            str_token,
            str_label,
            int_token_start,
            int_token_end,
        ) in list_tuple_mapped_tokens:
            bool_is_hard_break_token: bool = self._is_usaddress_hard_break_token(
                str_token=str_token, str_label=str_label
            )
            if bool_is_hard_break_token:
                if int_run_start is not None and int_run_end is not None:
                    bool_valid_hard_break_span: bool = self._is_valid_usaddress_span(
                        set_str_span_labels=set_str_run_labels,
                        int_span_start=int_run_start,
                        int_span_end=int_run_end,
                    )
                    if bool_valid_hard_break_span:
                        list_tuple_candidate_spans.append((int_run_start, int_run_end))
                int_run_start = None
                int_run_end = None
                set_str_run_labels = set()
                continue

            bool_is_address_component: bool = (
                str_label in SET_STR_USADDRESS_COMPONENT_LABELS
            )
            bool_is_connector_token: bool = (
                str_token.strip().lower() in SET_STR_USADDRESS_CONNECTOR_TOKENS
            )

            if bool_is_address_component:
                if int_run_start is None:
                    int_run_start = int_token_start
                int_run_end = int_token_end
                set_str_run_labels.add(str_label)
                continue

            if int_run_start is not None and bool_is_connector_token:
                int_run_end = int_token_end
                continue

            if int_run_start is not None and int_run_end is not None:
                bool_valid_span: bool = self._is_valid_usaddress_span(
                    set_str_span_labels=set_str_run_labels,
                    int_span_start=int_run_start,
                    int_span_end=int_run_end,
                )
                if bool_valid_span:
                    list_tuple_candidate_spans.append((int_run_start, int_run_end))
                int_run_start = None
                int_run_end = None
                set_str_run_labels = set()

        if int_run_start is not None and int_run_end is not None:
            bool_valid_trailing_span: bool = self._is_valid_usaddress_span(
                set_str_span_labels=set_str_run_labels,
                int_span_start=int_run_start,
                int_span_end=int_run_end,
            )
            if bool_valid_trailing_span:
                list_tuple_candidate_spans.append((int_run_start, int_run_end))

        list_usaddress_results: list[RecognizerResult] = []

        # Loop through valid span coordinates and convert each span into a LOCATION RecognizerResult.
        for int_span_start, int_span_end in list_tuple_candidate_spans:
            usaddress_result: RecognizerResult = RecognizerResult(
                entity_type=LOCATION_ENTITY,
                start=int_span_start,
                end=int_span_end,
                score=USADDRESS_RECOGNIZER_SCORE,
                recognition_metadata={
                    RecognizerResult.RECOGNIZER_NAME_KEY: USADDRESS_RECOGNIZER_NAME,
                    RecognizerResult.RECOGNIZER_IDENTIFIER_KEY: USADDRESS_RECOGNIZER_IDENTIFIER,
                },
            )
            list_usaddress_results.append(usaddress_result)

        # Normal return containing parser-backed LOCATION spans.
        return list_usaddress_results

    def _merge_usaddress_location_results(
        self, list_usaddress_results: list[RecognizerResult]
    ) -> list[RecognizerResult]:
        """
        Merges overlapping usaddress LOCATION spans into maximal contiguous blocks.

        Args:
            list_usaddress_results: Raw parser-backed LOCATION recognizer results.

        Returns:
            A merged list of parser-backed LOCATION spans.
        """
        if not list_usaddress_results:
            # Early return because there are no parser-backed spans to merge.
            return []

        list_sorted_results: list[RecognizerResult] = sorted(
            list_usaddress_results,
            key=lambda result: (
                result.start,
                -(result.end - result.start),
            ),
        )
        current_span: RecognizerResult = list_sorted_results[0]
        list_merged_spans: list[RecognizerResult] = []

        # Loop through sorted parser-backed spans and merge any overlapping ranges.
        for next_span in list_sorted_results[1:]:
            bool_overlaps_or_touches: bool = (
                next_span.start <= current_span.end + OVERLAP_ALLOWED_GAP
            )
            if bool_overlaps_or_touches:
                current_span.end = max(current_span.end, next_span.end)
                current_span.score = max(current_span.score, next_span.score)
                continue
            list_merged_spans.append(current_span)
            current_span = next_span
        list_merged_spans.append(current_span)

        # Normal return with merged parser-backed LOCATION spans.
        return list_merged_spans

    def _apply_span_conflict_policy(
        self,
        list_analyzer_results: list[RecognizerResult],
        list_usaddress_results: list[RecognizerResult],
    ) -> list[RecognizerResult]:
        """
        Applies overlap conflict resolution between parser-backed spans and analyzer spans.

        Args:
            list_analyzer_results: AnalyzerEngine results from Presidio recognizers.
            list_usaddress_results: Parser-backed LOCATION results from usaddress.

        Returns:
            A resolved list of RecognizerResult objects honoring the configured span policy.
        """
        if (
            self.str_span_conflict_policy
            != SPAN_CONFLICT_POLICY_PREFER_USADDRESS_LONGEST
        ):
            list_passthrough_results: list[RecognizerResult] = (
                list_analyzer_results + list_usaddress_results
            )
            # Normal return because alternate policy is configured as pass-through.
            return list_passthrough_results

        list_preferred_address_spans: list[RecognizerResult] = (
            self._merge_usaddress_location_results(list_usaddress_results)
        )
        if not list_preferred_address_spans:
            # Early return because no parser-backed address spans exist to enforce precedence.
            return list_analyzer_results

        list_protected_structured_results: list[RecognizerResult] = []
        # Loop through analyzer results and retain structured entities which
        # should never be overridden by parser-backed LOCATION spans over the
        # same text sequence. This protects strict custom recognizers such as
        # SSN/card last-4 and DOB, while also preventing usaddress from
        # stealing clearly typed entities such as phone numbers.
        for analyzer_result in list_analyzer_results:
            if (
                analyzer_result.entity_type
                in SET_STR_PROTECTED_STRUCTURED_CONFLICT_ENTITIES
            ):
                list_protected_structured_results.append(analyzer_result)

        list_filtered_address_spans: list[RecognizerResult] = []
        # Loop through parser-backed address spans and drop any span which
        # overlaps a protected structured recognizer result such as a phone
        # number, SSN last-4, payment-card last-4, or DOB. This preserves the
        # stricter detector when usaddress overreaches on short numeric phrases
        # inside mixed prompts.
        for parser_span in list_preferred_address_spans:
            bool_overlaps_protected_structured_result: bool = any(
                self._spans_overlap(
                    int_start_one=parser_span.start,
                    int_end_one=parser_span.end,
                    int_start_two=protected_result.start,
                    int_end_two=protected_result.end,
                )
                for protected_result in list_protected_structured_results
            )
            if bool_overlaps_protected_structured_result:
                continue
            list_filtered_address_spans.append(parser_span)

        if not list_filtered_address_spans:
            # Early return because all parser-backed address spans lost precedence
            # to stricter protected custom recognizer results.
            return list_analyzer_results

        list_filtered_results: list[RecognizerResult] = []

        # Loop through analyzer results and drop overlapping PERSON/LOCATION spans when parser-backed address spans win.
        for analyzer_result in list_analyzer_results:
            bool_overlaps_preferred_span: bool = any(
                self._spans_overlap(
                    int_start_one=analyzer_result.start,
                    int_end_one=analyzer_result.end,
                    int_start_two=parser_span.start,
                    int_end_two=parser_span.end,
                )
                for parser_span in list_filtered_address_spans
            )
            if (
                bool_overlaps_preferred_span
                and analyzer_result.entity_type in SET_STR_SPAN_CONFLICT_ENTITIES
            ):
                continue
            list_filtered_results.append(analyzer_result)

        list_resolved_results: list[RecognizerResult] = (
            list_filtered_results + list_filtered_address_spans
        )
        list_resolved_results.sort(key=lambda result: result.start)

        # Normal return with conflicts resolved by parser-span precedence.
        return list_resolved_results

    def _build_operators(
        self, list_recognizer_results: list[RecognizerResult]
    ) -> dict[str, OperatorConfig]:
        """
        Dynamically builds the redaction replacement operators mapped to the detected entities.
        Translates raw Presidio entity labels into custom requested labels (e.g., PERSON -> NAME).

        Args:
            list_recognizer_results: List of coordinate hits returned from the AnalyzerEngine.

        Returns:
            A dictionary mapping string entity types to presidio OperatorConfig objects which define
            the required text replacement action for the AnonymizerEngine.
        """
        dict_operators: dict[str, OperatorConfig] = {}
        set_str_entity_types: set[str] = {
            result.entity_type for result in list_recognizer_results
        }

        # Loop through all unique detected entities to dynamically generate their redaction replacement string rules.
        for str_entity in set_str_entity_types:
            str_canonical_category: str = self._map_category_to_canonical(str_entity)
            str_mapped_label: str = self.dict_str_entity_label_map.get(
                str_canonical_category,
                str_canonical_category,
            )
            str_replacement_text: str = (
                f"[{self.str_default_redaction_label}:{str_mapped_label}]"
            )
            dict_operators[str_entity] = OperatorConfig(
                REPLACE_OPERATOR,
                {NEW_VALUE_KEY: str_replacement_text},
            )
        dict_operators[DEFAULT_OPERATOR_KEY] = OperatorConfig(
            REPLACE_OPERATOR,
            {NEW_VALUE_KEY: f"[{self.str_default_redaction_label}]"},
        )
        # Normal return mapping operators perfectly.
        return dict_operators

    def _extend_person_with_capitalized_tokens(
        self, str_text: str, int_end_idx: int
    ) -> int:
        """
        Extends a PERSON span to include immediately following capitalized tokens (e.g., surnames).
        This patches holes in the underlying NLP engine where it fails to grab the entire name string.

        Args:
            str_text: The complete original text string being scanned for PII.
            int_end_idx: The current trailing character index of the detected PERSON entity.

        Returns:
            An integer representing the new, extended trailing character index covering the full name.
        """
        int_idx: int = int_end_idx
        int_length: int = len(str_text)

        # Loop forward through the string to capture consecutive capitalized potential surname tokens.
        while int_idx < int_length:
            # Skip whitespace and basic punctuation between tokens.
            # Small inline loop to gobble whitespace before checking the next word bounds.
            while (
                int_idx < int_length
                and str_text[int_idx] in SET_STR_WHITESPACE_AND_PUNCTUATION
            ):
                int_idx += 1
            if int_idx >= int_length:
                break
            int_token_start: int = int_idx
            # Small inline loop to gobble the full contiguous alphabetical word token block
            while int_idx < int_length and str_text[int_idx].isalpha():
                int_idx += 1
            str_token: str = str_text[int_token_start:int_idx]
            if str_token and str_token[0].isupper():
                int_end_idx = int_idx
                continue
            break

        # Normal return giving back the updated maximum spanning index.
        return int_end_idx

    def _merge_person_spans(
        self, list_recognizer_results: list[RecognizerResult], str_text: str
    ) -> list[RecognizerResult]:
        """
        Merges adjacent PERSON spans into a single block so full names (First Last) are redacted together.
        Instead of returning "[REDACTED:NAME] [REDACTED:NAME]", this ensures a single continuous replacement.

        Args:
            list_recognizer_results: List of coordinate hits returned from the AnalyzerEngine.
            str_text: The original scanned text string.

        Returns:
            A consolidated list of RecognizerResult objects representing the merged entity spans.
        """
        if not list_recognizer_results:
            # Early return because there were zero entity hits in the payload.
            return list_recognizer_results

        # Sort results sequentially to align overlapping/adjacent spans.
        list_recognizer_results.sort(key=lambda x: x.start)
        list_merged_spans: list[RecognizerResult] = []

        # Loop through all sequentially sorted entities to identify touching/overlapping PERSON blocks to combine.
        for result in list_recognizer_results:
            if result.entity_type != PERSON_ENTITY:
                list_merged_spans.append(result)
                continue

            result.end = self._extend_person_with_capitalized_tokens(
                str_text, result.end
            )

            if not list_merged_spans:
                list_merged_spans.append(result)
                continue

            prev: RecognizerResult = list_merged_spans[-1]

            if prev.entity_type == PERSON_ENTITY:
                str_between: str = str_text[prev.end : result.start]
                bool_only_whitespace_or_punctuation: bool = all(
                    char in SET_STR_WHITESPACE_AND_PUNCTUATION for char in str_between
                )
                if (prev.end >= result.start) or bool_only_whitespace_or_punctuation:
                    prev.end = max(prev.end, result.end)
                    prev.score = max(prev.score, result.score)
                    continue

            list_merged_spans.append(result)

        # Normal return with the pruned and merged list of spans.
        return list_merged_spans

    def sanitize_with_result(self, str_text: str) -> RedactionResult:
        """
        Detects and redacts PII from text and returns canonical category metadata.

        Args:
            str_text: The raw text string intended for the AI provider or returned by the AI provider.

        Returns:
            A RedactionResult containing sanitized text and canonical detected category metadata.
            Raises PiiRedactionRuntimeError when sanitization fails.
        """
        if not str_text or not str_text.strip():
            # Early return because there is no characters to process.
            return RedactionResult(str_sanitized_text=str_text)

        # If allowed_entities wiped out all supported entities, skip analysis.
        if not self.list_str_supported_entities:
            # Early return because all supported categories are explicitly allowed.
            return RedactionResult(str_sanitized_text=str_text)

        try:
            if self.bool_uses_shared_engine_cache:
                logger.debug(
                    "Using shared redaction engine cache: namespace=%s identity=%s",
                    self.str_engine_cache_namespace,
                    self.tuple_str_engine_cache_identity,
                )
            str_engine_cache_identity_signature: str = CACHE_IDENTITY_SEPARATOR.join(
                self.tuple_str_engine_cache_identity
            )
            analyzer: AnalyzerEngine = get_cached_analyzer_engine(
                self.str_language,
                self.str_nlp_profile,
                str_engine_cache_identity_signature,
                self.tuple_custom_recognizer_registration_configs,
            )
            anonymizer: AnonymizerEngine = get_cached_anonymizer_engine()

            # Analyze text for supported entities.
            list_recognizer_results: list[RecognizerResult] = analyzer.analyze(
                text=str_text,
                entities=self.list_str_analysis_entities,
                language=self.str_language,
            )
            list_usaddress_results: list[RecognizerResult] = (
                self._collect_usaddress_location_results(str_text=str_text)
            )
            list_resolved_results: list[RecognizerResult] = (
                self._apply_span_conflict_policy(
                    list_analyzer_results=list_recognizer_results,
                    list_usaddress_results=list_usaddress_results,
                )
            )

            # Check if detection found anything.
            if not list_resolved_results:
                # Early return because no PII was detected in this text.
                return RedactionResult(str_sanitized_text=str_text)

            # Post-process spans (merge and extend names).
            list_merged_results: list[RecognizerResult] = self._merge_person_spans(
                list_resolved_results, str_text
            )
            list_redactable_results: list[RecognizerResult] = [
                recognizer_result
                for recognizer_result in list_merged_results
                if recognizer_result.entity_type
                not in self.list_str_conflict_protection_entities
            ]

            # Build operator configs mapped to detected entities.
            dict_operators: dict[str, OperatorConfig] = self._build_operators(
                list_redactable_results
            )

            # Redact text securely using only the redactable subset. Pass-through
            # protected entities were intentionally analyzed for overlap
            # suppression, but they must not reach anonymization once allow-list
            # policy says to preserve them.
            anonymizer_result: Any = anonymizer.anonymize(
                text=str_text,
                analyzer_results=list_redactable_results,
                operators=dict_operators,
            )
            str_final_text: str = str(anonymizer_result.text)

            list_str_provider_categories: list[str] = self._deduplicate_category_tokens(
                [
                    str(recognizer_result.entity_type).strip().upper()
                    for recognizer_result in list_redactable_results
                ]
            )
            list_str_canonical_categories: list[str] = (
                self._deduplicate_category_tokens(
                    [
                        self._map_category_to_canonical(str_provider_category)
                        for str_provider_category in list_str_provider_categories
                    ]
                )
            )

            redaction_result: RedactionResult = RedactionResult(
                str_sanitized_text=str_final_text,
                list_str_detected_categories=list_str_canonical_categories,
                int_redaction_count=len(list_redactable_results),
            )
            # Normal return outputting the successfully sanitized text and categories.
            return redaction_result

        except Exception as exception:
            logger.exception(REDACTION_RUNTIME_ERROR_MESSAGE)
            raise PiiRedactionRuntimeError(
                REDACTION_RUNTIME_ERROR_MESSAGE
            ) from exception

    def sanitize_text(self, str_text: str) -> str:
        """
        Sanitizes text and returns only the redacted string.

        Args:
            str_text: The raw text string intended for redaction.

        Returns:
            The sanitized text from sanitize_with_result.
        """
        redaction_result: RedactionResult = self.sanitize_with_result(str_text)
        # Normal return with sanitized text payload.
        return redaction_result.str_sanitized_text
