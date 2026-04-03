import logging
import os
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from ai_api_unified.util.env_settings import EnvSettings

logger: logging.Logger = logging.getLogger(__name__)

MIDDLEWARE_KEY: str = "middleware"
NAME_KEY: str = "name"
ENABLED_KEY: str = "enabled"
SETTINGS_KEY: str = "settings"
PII_REDACTION: str = "pii_redaction"
OBSERVABILITY: str = "observability"
INPUT_ONLY: str = "input_only"
EN_LANGUAGE: str = "en"
REDACTED_LABEL: str = "REDACTED"
FILE_READ_MODE: str = "r"
FILE_ENCODING: str = "utf-8"
ENV_CONFIGURATION_PATH: str = "AI_MIDDLEWARE_CONFIG_PATH"

DETECTION_PROFILE_KEY: str = "detection_profile"
DETECTION_PROFILE_LOW_MEMORY: str = "low_memory"
DETECTION_PROFILE_BALANCED: str = "balanced"
DETECTION_PROFILE_HIGH_ACCURACY: str = "high_accuracy"
COUNTRY_SCOPE_US: str = "US"
ADDRESS_DETECTION_PROVIDER_USADDRESS: str = "usaddress"
SPAN_CONFLICT_POLICY_PREFER_USADDRESS_LONGEST: str = "prefer_usaddress_longest"
DIRECTION_KEY: str = "direction"
STRICT_MODE_KEY: str = "strict_mode"
LANGUAGE_KEY: str = "language"
COUNTRY_SCOPE_KEY: str = "country_scope"
ADDRESS_DETECTION_ENABLED_KEY: str = "address_detection_enabled"
ADDRESS_DETECTION_PROVIDER_KEY: str = "address_detection_provider"
SPAN_CONFLICT_POLICY_KEY: str = "span_conflict_policy"
DEFAULT_REDACTION_LABEL_KEY: str = "default_redaction_label"
ALLOWED_ENTITIES_KEY: str = "allowed_entities"
ENTITY_LABEL_MAP_KEY: str = "entity_label_map"
REDACTION_RECOGNIZERS_KEY: str = "redaction_recognizers"
PROXIMITY_WINDOW_CHARS_KEY: str = "proximity_window_chars"
RECOGNIZER_SSN_LAST4_KEY: str = "ssn_last4"
RECOGNIZER_CC_LAST4_KEY: str = "cc_last4"
RECOGNIZER_DOB_KEY: str = "dob"
RECOGNIZER_ENABLED_KEY: str = "enabled"
RECOGNIZER_CONFIDENCE_THRESHOLD_KEY: str = "confidence_threshold"
RECOGNIZER_CONTEXT_TERMS_KEY: str = "context_terms"
RECOGNIZER_NEGATIVE_CONTEXT_TERMS_KEY: str = "negative_context_terms"
DIRECTION_OUTPUT_ONLY: str = "output_only"
DIRECTION_INPUT_OUTPUT: str = "input_output"
CAPABILITIES_KEY: str = "capabilities"
LOG_LEVEL_KEY: str = "log_level"
TOKEN_COUNT_MODE_KEY: str = "token_count_mode"
INCLUDE_MEDIA_DETAILS_KEY: str = "include_media_details"
INCLUDE_PROVIDER_USAGE_KEY: str = "include_provider_usage"
INCLUDE_AUDIO_BYTE_COUNT_KEY: str = "include_audio_byte_count"
INCLUDE_IMAGE_BYTE_COUNT_KEY: str = "include_image_byte_count"
EMIT_ERROR_EVENTS_KEY: str = "emit_error_events"
CAPABILITY_COMPLETIONS: str = "completions"
CAPABILITY_EMBEDDINGS: str = "embeddings"
CAPABILITY_IMAGES: str = "images"
CAPABILITY_TTS: str = "tts"
LOG_LEVEL_DEBUG: str = "DEBUG"
LOG_LEVEL_INFO: str = "INFO"
LOG_LEVEL_WARNING: str = "WARNING"
LOG_LEVEL_ERROR: str = "ERROR"
LOG_LEVEL_CRITICAL: str = "CRITICAL"
TOKEN_COUNT_MODE_PROVIDER_ONLY: str = "provider_only"
TOKEN_COUNT_MODE_PROVIDER_OR_ESTIMATE: str = "provider_or_estimate"
TOKEN_COUNT_MODE_NONE: str = "none"
DEFAULT_PROXIMITY_WINDOW_CHARS: int = 28
MIN_PROXIMITY_WINDOW_CHARS: int = 1
DEFAULT_RECOGNIZER_CONFIDENCE_THRESHOLD: float = 0.0
MAX_RECOGNIZER_CONFIDENCE_THRESHOLD: float = 1.0
DEFAULT_SSN_LAST4_CONFIDENCE_THRESHOLD: float = 0.75
DEFAULT_CC_LAST4_CONFIDENCE_THRESHOLD: float = 0.80
DEFAULT_DOB_CONFIDENCE_THRESHOLD: float = 0.80
DEFAULT_OBSERVABILITY_LOG_LEVEL: str = LOG_LEVEL_INFO
DEFAULT_OBSERVABILITY_TOKEN_COUNT_MODE: str = TOKEN_COUNT_MODE_PROVIDER_OR_ESTIMATE
LIST_STR_DEFAULT_SSN_LAST4_CONTEXT_TERMS: list[str] = [
    "ssn",
    "social security",
]
LIST_STR_DEFAULT_SSN_LAST4_NEGATIVE_CONTEXT_TERMS: list[str] = [
    "cc",
    "credit card",
    "debit",
    "visa",
    "mastercard",
    "master card",
    "amex",
    "american express",
    "discover",
]
LIST_STR_DEFAULT_CC_LAST4_CONTEXT_TERMS: list[str] = [
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
]
LIST_STR_DEFAULT_CC_LAST4_NEGATIVE_CONTEXT_TERMS: list[str] = [
    "ssn",
    "social security",
]
LIST_STR_DEFAULT_DOB_CONTEXT_TERMS: list[str] = [
    "dob",
    "date of birth",
    "born",
    "birth date",
]
LIST_STR_DEFAULT_DOB_NEGATIVE_CONTEXT_TERMS: list[str] = [
    "invoice date",
    "appointment date",
    "due date",
]
SET_STR_DIRECTION_VALUES: set[str] = {
    INPUT_ONLY,
    DIRECTION_OUTPUT_ONLY,
    DIRECTION_INPUT_OUTPUT,
}
SET_STR_DETECTION_PROFILE_VALUES: set[str] = {
    DETECTION_PROFILE_LOW_MEMORY,
    DETECTION_PROFILE_BALANCED,
    DETECTION_PROFILE_HIGH_ACCURACY,
}
SET_STR_OBSERVABILITY_CAPABILITY_VALUES: set[str] = {
    CAPABILITY_COMPLETIONS,
    CAPABILITY_EMBEDDINGS,
    CAPABILITY_IMAGES,
    CAPABILITY_TTS,
}
SET_STR_OBSERVABILITY_LOG_LEVEL_VALUES: set[str] = {
    LOG_LEVEL_DEBUG,
    LOG_LEVEL_INFO,
    LOG_LEVEL_WARNING,
    LOG_LEVEL_ERROR,
    LOG_LEVEL_CRITICAL,
}
SET_STR_TOKEN_COUNT_MODE_VALUES: set[str] = {
    TOKEN_COUNT_MODE_PROVIDER_ONLY,
    TOKEN_COUNT_MODE_PROVIDER_OR_ESTIMATE,
    TOKEN_COUNT_MODE_NONE,
}
DICT_STR_DEFAULT_ENTITY_LABEL_MAP: dict[str, str] = {
    "NAME": "NAME",
    "PHONE": "PHONE",
    "EMAIL": "EMAIL",
    "SSN": "SSN",
    "ADDRESS": "ADDRESS",
    "DOB": "DOB",
    "CC_LAST4": "CC_LAST4",
}
DICT_STR_CANONICAL_CATEGORY_TOKENS: dict[str, str] = {
    "NAME": "NAME",
    "PHONE": "PHONE",
    "EMAIL": "EMAIL",
    "SSN": "SSN",
    "ADDRESS": "ADDRESS",
    "DOB": "DOB",
    "CC_LAST4": "CC_LAST4",
}
SET_STR_VALID_CANONICAL_CATEGORIES: set[str] = set(
    DICT_STR_CANONICAL_CATEGORY_TOKENS.keys()
)
MIDDLEWARE_LIST_TYPE_ERROR_MESSAGE: str = (
    "Middleware config key `middleware` must be a list; treating middleware as disabled."
)
MIDDLEWARE_EMPTY_SETTINGS_DISABLED_MESSAGE: str = (
    "Middleware '%s' is enabled but has missing, non-dictionary, or empty settings; "
    "treating it as disabled."
)
MIDDLEWARE_YAML_PARSE_ERROR_MESSAGE: str = (
    "Failed to parse middleware config YAML from %s: %s"
)
MIDDLEWARE_FILE_READ_ERROR_MESSAGE: str = (
    "Failed to read middleware config file from %s: %s"
)
MIDDLEWARE_ENCODING_ERROR_MESSAGE: str = (
    "Failed to decode middleware config file from %s with %s encoding: %s"
)
MIDDLEWARE_EMPTY_FILE_WARNING_MESSAGE: str = (
    "Middleware config file is empty at %s; treating middleware as disabled."
)
MIDDLEWARE_PATH_SET_BUT_MISSING_WARNING_MESSAGE: str = (
    "Middleware config path from `%s` does not exist: %s. "
    "Treating middleware as disabled."
)
MIDDLEWARE_INVALID_ENTRY_MESSAGE: str = (
    "Failed to validate middleware entry; skipping this item. Entry: %s Error: %s"
)
PII_SETTINGS_VALIDATION_ERROR_MESSAGE: str = (
    "PII middleware settings failed validation; treating component as disabled. Error: %s"
)
OBSERVABILITY_SETTINGS_VALIDATION_ERROR_MESSAGE: str = (
    "Observability middleware settings failed validation; treating component as disabled. Error: %s"
)
OBSERVABILITY_SETTINGS_INVALID_VALUE_WARNING_MESSAGE: str = (
    "Observability setting `%s` is invalid; using default value. value=%r"
)
OBSERVABILITY_CAPABILITIES_INVALID_TYPE_WARNING_MESSAGE: str = (
    "Observability setting `%s` must be a string or list-like value; using defaults. type=%s"
)
OBSERVABILITY_CAPABILITIES_DROPPED_VALUES_WARNING_MESSAGE: str = (
    "Observability setting `%s` dropped unsupported capability values: %s"
)
PII_RECOGNIZERS_MISSING_INFO_MESSAGE: str = (
    "PII redaction setting `%s` is missing; using defaults."
)
PII_RECOGNIZERS_INVALID_TYPE_WARNING_MESSAGE: str = (
    "PII redaction setting `%s` must be a dictionary; using defaults. type=%s"
)
PII_RECOGNIZERS_MISSING_VALUE_INFO_MESSAGE: str = (
    "PII redaction setting `%s.%s` is missing/null; using default value."
)
PII_RECOGNIZERS_INVALID_VALUE_WARNING_MESSAGE: str = (
    "PII redaction setting `%s.%s` is invalid; using default value. value=%r"
)
PII_RECOGNIZER_RULE_INVALID_TYPE_WARNING_MESSAGE: str = (
    "PII redaction recognizer rule `%s.%s` must be a dictionary; using defaults. type=%s"
)
PII_RECOGNIZER_RULE_VALIDATION_FAILED_WARNING_MESSAGE: str = (
    "PII redaction recognizer rule `%s.%s` failed validation; using defaults. error=%s"
)
PII_SETTINGS_INVALID_VALUE_WARNING_MESSAGE: str = (
    "PII redaction setting `%s` is invalid; using default value. value=%r"
)
PII_ALLOWED_ENTITIES_INVALID_TYPE_WARNING_MESSAGE: str = (
    "PII redaction setting `%s` must be a string or list-like value; using empty list. type=%s"
)
PII_ALLOWED_ENTITIES_DROPPED_VALUES_WARNING_MESSAGE: str = (
    "PII redaction setting `%s` dropped unsupported category values: %s"
)
PII_ENTITY_LABEL_MAP_INVALID_TYPE_WARNING_MESSAGE: str = (
    "PII redaction setting `%s` must be a dictionary; using defaults. type=%s"
)
PII_ENTITY_LABEL_MAP_DROPPED_KEYS_WARNING_MESSAGE: str = (
    "PII redaction setting `%s` dropped unsupported category keys: %s"
)
UNSUPPORTED_TYPED_MIDDLEWARE_DEBUG_MESSAGE: str = (
    "Typed middleware settings requested for unsupported middleware name: %s"
)


class MiddlewareEntryModel(BaseModel):
    """
    Typed model for one middleware entry in the configuration profile.
    """

    model_config: ConfigDict = ConfigDict(extra="allow")

    name: str
    enabled: bool = False
    settings: Any = None


class MiddlewareConfigurationModel(BaseModel):
    """
    Typed root model for the middleware configuration profile.
    """

    model_config: ConfigDict = ConfigDict(extra="allow", populate_by_name=True)

    list_middleware: list[MiddlewareEntryModel] = Field(
        default_factory=list,
        alias=MIDDLEWARE_KEY,
    )


class RedactionRecognizerRuleSettingsModel(BaseModel):
    """
    Typed model for one configurable custom recognizer rule block.

    This model intentionally exposes middleware-level semantics only.
    It does not expose implementation-provider identifiers in the YAML contract.
    """

    model_config: ConfigDict = ConfigDict(extra="allow", str_strip_whitespace=True)

    enabled: bool = True
    confidence_threshold: float = DEFAULT_RECOGNIZER_CONFIDENCE_THRESHOLD
    context_terms: list[str] = Field(default_factory=list)
    negative_context_terms: list[str] = Field(default_factory=list)

    @field_validator(RECOGNIZER_CONFIDENCE_THRESHOLD_KEY, mode="before")
    @classmethod
    def _normalize_confidence_threshold(cls, value: Any) -> float:
        """
        Normalizes recognizer confidence thresholds into inclusive range [0.0, 1.0].

        Args:
            value: Raw confidence threshold value from YAML.

        Returns:
            Clamped floating-point confidence threshold value.
        """
        if value is None:
            # Early return because confidence threshold is omitted.
            return DEFAULT_RECOGNIZER_CONFIDENCE_THRESHOLD
        try:
            float_confidence_threshold: float = float(value)
        except (TypeError, ValueError):
            logger.warning(
                PII_RECOGNIZERS_INVALID_VALUE_WARNING_MESSAGE,
                REDACTION_RECOGNIZERS_KEY,
                RECOGNIZER_CONFIDENCE_THRESHOLD_KEY,
                value,
            )
            # Early return because invalid threshold values fall back to default.
            return DEFAULT_RECOGNIZER_CONFIDENCE_THRESHOLD
        if float_confidence_threshold < DEFAULT_RECOGNIZER_CONFIDENCE_THRESHOLD:
            logger.warning(
                PII_RECOGNIZERS_INVALID_VALUE_WARNING_MESSAGE,
                REDACTION_RECOGNIZERS_KEY,
                RECOGNIZER_CONFIDENCE_THRESHOLD_KEY,
                value,
            )
            # Early return because threshold floor is 0.0.
            return DEFAULT_RECOGNIZER_CONFIDENCE_THRESHOLD
        if float_confidence_threshold > MAX_RECOGNIZER_CONFIDENCE_THRESHOLD:
            logger.warning(
                PII_RECOGNIZERS_INVALID_VALUE_WARNING_MESSAGE,
                REDACTION_RECOGNIZERS_KEY,
                RECOGNIZER_CONFIDENCE_THRESHOLD_KEY,
                value,
            )
            # Early return because threshold ceiling is 1.0.
            return MAX_RECOGNIZER_CONFIDENCE_THRESHOLD
        # Normal return with normalized confidence threshold.
        return float_confidence_threshold

    @classmethod
    def _normalize_string_list(cls, value: Any) -> list[str]:
        """
        Coerces string-list-like values into non-empty stripped string lists.

        Args:
            value: Raw list-like value from YAML.

        Returns:
            List of normalized non-empty string values.
        """
        if value is None:
            # Early return because list value is omitted.
            return []
        if isinstance(value, str):
            str_item: str = value.strip()
            if str_item:
                # Early return for single-string list values.
                return [str_item]
            # Early return for blank single-string values.
            return []
        if not isinstance(value, (list, tuple, set)):
            # Early return because unsupported list types map to empty.
            return []

        list_str_items: list[str] = []
        list_iterable_values: list[Any] = []
        if isinstance(value, set):
            # Normalize set input into deterministic order for stable config output and logs.
            list_iterable_values = sorted(value, key=lambda item: str(item))
        else:
            # Preserve caller-provided ordering for list/tuple values.
            list_iterable_values = list(value)
        # Loop through incoming values and keep non-empty normalized string items.
        for item in list_iterable_values:
            if not isinstance(item, str):
                continue
            str_item: str = item.strip()
            if str_item:
                list_str_items.append(str_item)
        # Normal return with normalized list values.
        return list_str_items

    @field_validator(RECOGNIZER_CONTEXT_TERMS_KEY, mode="before")
    @classmethod
    def _normalize_context_terms(cls, value: Any) -> list[str]:
        """
        Normalizes positive context term values for one recognizer.

        Args:
            value: Raw context terms value from YAML.

        Returns:
            List of normalized non-empty context terms.
        """
        list_str_context_terms: list[str] = cls._normalize_string_list(value)
        # Normal return with normalized context terms.
        return list_str_context_terms

    @field_validator(RECOGNIZER_NEGATIVE_CONTEXT_TERMS_KEY, mode="before")
    @classmethod
    def _normalize_negative_context_terms(cls, value: Any) -> list[str]:
        """
        Normalizes negative context term values for one recognizer.

        Args:
            value: Raw negative context terms value from YAML.

        Returns:
            List of normalized non-empty negative context terms.
        """
        list_str_negative_context_terms: list[str] = cls._normalize_string_list(value)
        # Normal return with normalized negative context terms.
        return list_str_negative_context_terms


class RedactionRecognizersSettingsModel(BaseModel):
    """
    Typed model for `redaction_recognizers` middleware extension settings.
    """

    model_config: ConfigDict = ConfigDict(extra="allow", str_strip_whitespace=True)

    proximity_window_chars: int = DEFAULT_PROXIMITY_WINDOW_CHARS
    ssn_last4: RedactionRecognizerRuleSettingsModel = Field(
        default_factory=lambda: RedactionRecognizerRuleSettingsModel(
            enabled=True,
            confidence_threshold=DEFAULT_SSN_LAST4_CONFIDENCE_THRESHOLD,
            context_terms=list(LIST_STR_DEFAULT_SSN_LAST4_CONTEXT_TERMS),
            negative_context_terms=list(
                LIST_STR_DEFAULT_SSN_LAST4_NEGATIVE_CONTEXT_TERMS
            ),
        )
    )
    cc_last4: RedactionRecognizerRuleSettingsModel = Field(
        default_factory=lambda: RedactionRecognizerRuleSettingsModel(
            enabled=True,
            confidence_threshold=DEFAULT_CC_LAST4_CONFIDENCE_THRESHOLD,
            context_terms=list(LIST_STR_DEFAULT_CC_LAST4_CONTEXT_TERMS),
            negative_context_terms=list(
                LIST_STR_DEFAULT_CC_LAST4_NEGATIVE_CONTEXT_TERMS
            ),
        )
    )
    dob: RedactionRecognizerRuleSettingsModel = Field(
        default_factory=lambda: RedactionRecognizerRuleSettingsModel(
            enabled=True,
            confidence_threshold=DEFAULT_DOB_CONFIDENCE_THRESHOLD,
            context_terms=list(LIST_STR_DEFAULT_DOB_CONTEXT_TERMS),
            negative_context_terms=list(LIST_STR_DEFAULT_DOB_NEGATIVE_CONTEXT_TERMS),
        )
    )

    @field_validator(PROXIMITY_WINDOW_CHARS_KEY, mode="before")
    @classmethod
    def _normalize_proximity_window_chars(cls, value: Any) -> int:
        """
        Normalizes proximity window values for recognizer context matching.

        Args:
            value: Raw proximity window value from YAML.

        Returns:
            Positive integer proximity window size.
        """
        if value is None:
            logger.info(
                PII_RECOGNIZERS_MISSING_VALUE_INFO_MESSAGE,
                REDACTION_RECOGNIZERS_KEY,
                PROXIMITY_WINDOW_CHARS_KEY,
            )
            # Early return because proximity window is omitted.
            return DEFAULT_PROXIMITY_WINDOW_CHARS
        try:
            int_proximity_window_chars: int = int(value)
        except (TypeError, ValueError):
            logger.warning(
                PII_RECOGNIZERS_INVALID_VALUE_WARNING_MESSAGE,
                REDACTION_RECOGNIZERS_KEY,
                PROXIMITY_WINDOW_CHARS_KEY,
                value,
            )
            # Early return because invalid values fall back to default.
            return DEFAULT_PROXIMITY_WINDOW_CHARS
        if int_proximity_window_chars < MIN_PROXIMITY_WINDOW_CHARS:
            logger.warning(
                PII_RECOGNIZERS_INVALID_VALUE_WARNING_MESSAGE,
                REDACTION_RECOGNIZERS_KEY,
                PROXIMITY_WINDOW_CHARS_KEY,
                value,
            )
            # Early return because proximity window must be positive.
            return DEFAULT_PROXIMITY_WINDOW_CHARS
        # Normal return with normalized proximity window.
        return int_proximity_window_chars

    @classmethod
    def _build_default_rule_settings(
        cls,
        float_confidence_threshold: float,
        list_str_context_terms: list[str],
        list_str_negative_context_terms: list[str] | None = None,
    ) -> RedactionRecognizerRuleSettingsModel:
        """
        Builds default recognizer rule settings used as fallback normalization targets.

        Args:
            float_confidence_threshold: Default confidence threshold for the recognizer.
            list_str_context_terms: Default positive context terms.
            list_str_negative_context_terms: Optional default negative context terms.

        Returns:
            A RedactionRecognizerRuleSettingsModel with standard defaults applied.
        """
        list_str_normalized_negative_context_terms: list[str] = (
            list_str_negative_context_terms
            if list_str_negative_context_terms is not None
            else []
        )
        recognizer_rule_settings: RedactionRecognizerRuleSettingsModel = (
            RedactionRecognizerRuleSettingsModel(
                enabled=True,
                confidence_threshold=float_confidence_threshold,
                context_terms=list(list_str_context_terms),
                negative_context_terms=list(list_str_normalized_negative_context_terms),
            )
        )
        # Normal return with default recognizer rule settings.
        return recognizer_rule_settings

    @classmethod
    def _normalize_rule_value(
        cls,
        str_rule_name: str,
        value: Any,
        recognizer_rule_defaults: RedactionRecognizerRuleSettingsModel,
    ) -> RedactionRecognizerRuleSettingsModel:
        """
        Normalizes one recognizer rule value with fallback-safe default behavior.

        Args:
            str_rule_name: Recognizer rule key (for example, "ssn_last4").
            value: Raw recognizer rule value from YAML.
            recognizer_rule_defaults: Default rule settings used for fallback behavior.

        Returns:
            Normalized recognizer rule settings model.
        """
        if isinstance(value, RedactionRecognizerRuleSettingsModel):
            # Normal return when value is already typed.
            return value
        if not isinstance(value, dict):
            logger.warning(
                PII_RECOGNIZER_RULE_INVALID_TYPE_WARNING_MESSAGE,
                REDACTION_RECOGNIZERS_KEY,
                str_rule_name,
                type(value).__name__,
            )
            # Early return because unsupported rule types use defaults.
            return recognizer_rule_defaults

        dict_raw_rule_values: dict[str, Any] = dict(value)
        if RECOGNIZER_CONFIDENCE_THRESHOLD_KEY in dict_raw_rule_values:
            raw_confidence_threshold_value: Any = dict_raw_rule_values[
                RECOGNIZER_CONFIDENCE_THRESHOLD_KEY
            ]
            try:
                float(raw_confidence_threshold_value)
            except (TypeError, ValueError):
                logger.warning(
                    PII_RECOGNIZERS_INVALID_VALUE_WARNING_MESSAGE,
                    f"{REDACTION_RECOGNIZERS_KEY}.{str_rule_name}",
                    RECOGNIZER_CONFIDENCE_THRESHOLD_KEY,
                    raw_confidence_threshold_value,
                )
                dict_raw_rule_values.pop(RECOGNIZER_CONFIDENCE_THRESHOLD_KEY, None)

        if RECOGNIZER_CONTEXT_TERMS_KEY in dict_raw_rule_values:
            raw_context_terms_value: Any = dict_raw_rule_values[
                RECOGNIZER_CONTEXT_TERMS_KEY
            ]
            bool_context_terms_value_supported: bool = (
                isinstance(raw_context_terms_value, (str, list, tuple, set))
                or raw_context_terms_value is None
            )
            if not bool_context_terms_value_supported:
                logger.warning(
                    PII_RECOGNIZERS_INVALID_VALUE_WARNING_MESSAGE,
                    f"{REDACTION_RECOGNIZERS_KEY}.{str_rule_name}",
                    RECOGNIZER_CONTEXT_TERMS_KEY,
                    raw_context_terms_value,
                )
                dict_raw_rule_values.pop(RECOGNIZER_CONTEXT_TERMS_KEY, None)

        if RECOGNIZER_NEGATIVE_CONTEXT_TERMS_KEY in dict_raw_rule_values:
            raw_negative_context_terms_value: Any = dict_raw_rule_values[
                RECOGNIZER_NEGATIVE_CONTEXT_TERMS_KEY
            ]
            bool_negative_context_terms_value_supported: bool = (
                isinstance(raw_negative_context_terms_value, (str, list, tuple, set))
                or raw_negative_context_terms_value is None
            )
            if not bool_negative_context_terms_value_supported:
                logger.warning(
                    PII_RECOGNIZERS_INVALID_VALUE_WARNING_MESSAGE,
                    f"{REDACTION_RECOGNIZERS_KEY}.{str_rule_name}",
                    RECOGNIZER_NEGATIVE_CONTEXT_TERMS_KEY,
                    raw_negative_context_terms_value,
                )
                dict_raw_rule_values.pop(RECOGNIZER_NEGATIVE_CONTEXT_TERMS_KEY, None)

        dict_default_rule_values: dict[str, Any] = recognizer_rule_defaults.model_dump()
        dict_merged_rule_values: dict[str, Any] = {
            **dict_default_rule_values,
            **dict_raw_rule_values,
        }
        try:
            normalized_rule_settings: RedactionRecognizerRuleSettingsModel = (
                RedactionRecognizerRuleSettingsModel.model_validate(
                    dict_merged_rule_values
                )
            )
        except ValidationError as exception:
            logger.warning(
                PII_RECOGNIZER_RULE_VALIDATION_FAILED_WARNING_MESSAGE,
                REDACTION_RECOGNIZERS_KEY,
                str_rule_name,
                exception,
            )
            # Early return because nested rule validation failed and defaults are safer.
            return recognizer_rule_defaults
        # Normal return with normalized rule settings.
        return normalized_rule_settings

    @field_validator(RECOGNIZER_SSN_LAST4_KEY, mode="before")
    @classmethod
    def _normalize_ssn_last4_rule(
        cls, value: Any
    ) -> RedactionRecognizerRuleSettingsModel:
        """
        Normalizes SSN last-4 recognizer rule settings.

        Args:
            value: Raw SSN last-4 recognizer settings from YAML.

        Returns:
            Normalized SSN last-4 recognizer rule settings.
        """
        recognizer_rule_defaults: RedactionRecognizerRuleSettingsModel = (
            cls._build_default_rule_settings(
                float_confidence_threshold=DEFAULT_SSN_LAST4_CONFIDENCE_THRESHOLD,
                list_str_context_terms=LIST_STR_DEFAULT_SSN_LAST4_CONTEXT_TERMS,
                list_str_negative_context_terms=LIST_STR_DEFAULT_SSN_LAST4_NEGATIVE_CONTEXT_TERMS,
            )
        )
        normalized_rule_settings: RedactionRecognizerRuleSettingsModel = (
            cls._normalize_rule_value(
                str_rule_name=RECOGNIZER_SSN_LAST4_KEY,
                value=value,
                recognizer_rule_defaults=recognizer_rule_defaults,
            )
        )
        # Normal return with normalized SSN last-4 settings.
        return normalized_rule_settings

    @field_validator(RECOGNIZER_CC_LAST4_KEY, mode="before")
    @classmethod
    def _normalize_cc_last4_rule(
        cls, value: Any
    ) -> RedactionRecognizerRuleSettingsModel:
        """
        Normalizes credit-card last-4 recognizer rule settings.

        Args:
            value: Raw credit-card last-4 recognizer settings from YAML.

        Returns:
            Normalized credit-card last-4 recognizer rule settings.
        """
        recognizer_rule_defaults: RedactionRecognizerRuleSettingsModel = (
            cls._build_default_rule_settings(
                float_confidence_threshold=DEFAULT_CC_LAST4_CONFIDENCE_THRESHOLD,
                list_str_context_terms=LIST_STR_DEFAULT_CC_LAST4_CONTEXT_TERMS,
                list_str_negative_context_terms=LIST_STR_DEFAULT_CC_LAST4_NEGATIVE_CONTEXT_TERMS,
            )
        )
        normalized_rule_settings: RedactionRecognizerRuleSettingsModel = (
            cls._normalize_rule_value(
                str_rule_name=RECOGNIZER_CC_LAST4_KEY,
                value=value,
                recognizer_rule_defaults=recognizer_rule_defaults,
            )
        )
        # Normal return with normalized credit-card last-4 settings.
        return normalized_rule_settings

    @field_validator(RECOGNIZER_DOB_KEY, mode="before")
    @classmethod
    def _normalize_dob_rule(cls, value: Any) -> RedactionRecognizerRuleSettingsModel:
        """
        Normalizes DOB recognizer rule settings.

        Args:
            value: Raw DOB recognizer settings from YAML.

        Returns:
            Normalized DOB recognizer rule settings.
        """
        recognizer_rule_defaults: RedactionRecognizerRuleSettingsModel = (
            cls._build_default_rule_settings(
                float_confidence_threshold=DEFAULT_DOB_CONFIDENCE_THRESHOLD,
                list_str_context_terms=LIST_STR_DEFAULT_DOB_CONTEXT_TERMS,
                list_str_negative_context_terms=LIST_STR_DEFAULT_DOB_NEGATIVE_CONTEXT_TERMS,
            )
        )
        normalized_rule_settings: RedactionRecognizerRuleSettingsModel = (
            cls._normalize_rule_value(
                str_rule_name=RECOGNIZER_DOB_KEY,
                value=value,
                recognizer_rule_defaults=recognizer_rule_defaults,
            )
        )
        # Normal return with normalized DOB settings.
        return normalized_rule_settings


class PiiRedactionSettingsModel(BaseModel):
    """
    Typed model for the `pii_redaction.settings` block.
    Defaults are intentionally lenient to keep configuration migration smooth.
    """

    model_config: ConfigDict = ConfigDict(extra="allow", str_strip_whitespace=True)

    direction: str = INPUT_ONLY
    strict_mode: bool = False
    detection_profile: str = DETECTION_PROFILE_BALANCED
    language: str = EN_LANGUAGE
    country_scope: str = COUNTRY_SCOPE_US
    address_detection_enabled: bool = True
    address_detection_provider: str = ADDRESS_DETECTION_PROVIDER_USADDRESS
    span_conflict_policy: str = SPAN_CONFLICT_POLICY_PREFER_USADDRESS_LONGEST
    default_redaction_label: str = REDACTED_LABEL
    allowed_entities: list[str] = Field(default_factory=list)
    entity_label_map: dict[str, str] = Field(
        default_factory=lambda: dict(DICT_STR_DEFAULT_ENTITY_LABEL_MAP)
    )
    redaction_recognizers: RedactionRecognizersSettingsModel = Field(
        default_factory=RedactionRecognizersSettingsModel
    )

    @classmethod
    def _normalize_category_to_canonical(cls, value: Any) -> str:
        """
        Normalizes raw category inputs into canonical middleware category names.

        Args:
            value: Raw category token from configuration.

        Returns:
            Canonical category string when token is valid, or an empty string when invalid.
        """
        str_category_upper: str = str(value).strip().upper()
        str_canonical_category: str = DICT_STR_CANONICAL_CATEGORY_TOKENS.get(
            str_category_upper,
            "",
        )
        # Normal return with canonicalized category token.
        return str_canonical_category

    @field_validator(DIRECTION_KEY, mode="before")
    @classmethod
    def _normalize_direction(cls, value: Any) -> str:
        """
        Normalizes direction values and falls back to default when invalid.

        Args:
            value: Raw direction input from YAML.

        Returns:
            Canonical direction string.
        """
        if value is None:
            # Early return because no direction was supplied.
            return INPUT_ONLY
        str_direction: str = str(value).strip().lower()
        if str_direction in SET_STR_DIRECTION_VALUES:
            # Normal return for accepted direction values.
            return str_direction
        logger.warning(
            PII_SETTINGS_INVALID_VALUE_WARNING_MESSAGE,
            DIRECTION_KEY,
            value,
        )
        # Early return because invalid direction values fall back to default.
        return INPUT_ONLY

    @field_validator(DETECTION_PROFILE_KEY, mode="before")
    @classmethod
    def _normalize_detection_profile(cls, value: Any) -> str:
        """
        Normalizes detection profile values and falls back to default when invalid.

        Args:
            value: Raw detection profile input from YAML.

        Returns:
            Canonical middleware detection profile value.
        """
        if value is None:
            # Early return because no detection profile was supplied.
            return DETECTION_PROFILE_BALANCED
        str_detection_profile: str = str(value).strip().lower()
        if str_detection_profile in SET_STR_DETECTION_PROFILE_VALUES:
            # Normal return for accepted detection profile values.
            return str_detection_profile
        logger.warning(
            PII_SETTINGS_INVALID_VALUE_WARNING_MESSAGE,
            DETECTION_PROFILE_KEY,
            value,
        )
        # Early return because unknown values fall back to default.
        return DETECTION_PROFILE_BALANCED

    @field_validator(LANGUAGE_KEY, mode="before")
    @classmethod
    def _normalize_language(cls, value: Any) -> str:
        """
        Normalizes language code values.

        Args:
            value: Raw language input from YAML.

        Returns:
            Lower-case language code.
        """
        if value is None:
            # Early return because no language was supplied.
            return EN_LANGUAGE
        str_language: str = str(value).strip().lower()
        if str_language:
            # Normal return with normalized language.
            return str_language
        # Early return for blank language values.
        return EN_LANGUAGE

    @field_validator(COUNTRY_SCOPE_KEY, mode="before")
    @classmethod
    def _normalize_country_scope(cls, value: Any) -> str:
        """
        Normalizes country scope values.

        Args:
            value: Raw country scope input from YAML.

        Returns:
            Upper-case country scope.
        """
        if value is None:
            # Early return because no country scope was supplied.
            return COUNTRY_SCOPE_US
        str_country_scope: str = str(value).strip().upper()
        if str_country_scope:
            # Normal return with normalized country scope.
            return str_country_scope
        # Early return for blank country scope values.
        return COUNTRY_SCOPE_US

    @field_validator(ADDRESS_DETECTION_PROVIDER_KEY, mode="before")
    @classmethod
    def _normalize_address_detection_provider(cls, value: Any) -> str:
        """
        Normalizes address detection provider values.

        Args:
            value: Raw provider input from YAML.

        Returns:
            Lower-case provider name.
        """
        if value is None:
            # Early return because no provider was supplied.
            return ADDRESS_DETECTION_PROVIDER_USADDRESS
        str_provider: str = str(value).strip().lower()
        if str_provider:
            # Normal return with normalized provider.
            return str_provider
        # Early return for blank provider values.
        return ADDRESS_DETECTION_PROVIDER_USADDRESS

    @field_validator(SPAN_CONFLICT_POLICY_KEY, mode="before")
    @classmethod
    def _normalize_span_conflict_policy(cls, value: Any) -> str:
        """
        Normalizes span conflict policy values.

        Args:
            value: Raw span conflict policy input from YAML.

        Returns:
            Lower-case policy name.
        """
        if value is None:
            # Early return because no policy was supplied.
            return SPAN_CONFLICT_POLICY_PREFER_USADDRESS_LONGEST
        str_policy: str = str(value).strip().lower()
        if str_policy:
            # Normal return with normalized policy.
            return str_policy
        # Early return for blank policy values.
        return SPAN_CONFLICT_POLICY_PREFER_USADDRESS_LONGEST

    @field_validator(ALLOWED_ENTITIES_KEY, mode="before")
    @classmethod
    def _normalize_allowed_entities(cls, value: Any) -> list[str]:
        """
        Coerces allow-list values into a predictable list of strings.

        Args:
            value: Raw allow-list input from YAML.

        Returns:
            List of upper-case entity names.
        """
        if value is None:
            # Early return because allow-list is omitted.
            return []
        if isinstance(value, str):
            # Normal return for single string input.
            str_canonical_category: str = cls._normalize_category_to_canonical(value)
            if str_canonical_category:
                # Normal return for supported single canonical category.
                return [str_canonical_category]
            logger.warning(
                PII_ALLOWED_ENTITIES_DROPPED_VALUES_WARNING_MESSAGE,
                ALLOWED_ENTITIES_KEY,
                [value],
            )
            # Early return because unsupported category token is ignored.
            return []
        if isinstance(value, (list, tuple, set)):
            list_str_canonical_categories: list[str] = []
            list_raw_dropped_entities: list[str] = []
            list_iterable_values: list[Any] = []
            if isinstance(value, set):
                # Normalize set input into deterministic order for stable config output and logs.
                list_iterable_values = sorted(value, key=lambda item: str(item))
            else:
                # Preserve caller-provided ordering for list/tuple values.
                list_iterable_values = list(value)
            # Loop through incoming category tokens and retain only supported canonical categories.
            for str_entity in list_iterable_values:
                str_canonical_category: str = cls._normalize_category_to_canonical(
                    str_entity
                )
                if str_canonical_category:
                    list_str_canonical_categories.append(str_canonical_category)
                    continue
                list_raw_dropped_entities.append(str(str_entity))
            if list_raw_dropped_entities:
                logger.warning(
                    PII_ALLOWED_ENTITIES_DROPPED_VALUES_WARNING_MESSAGE,
                    ALLOWED_ENTITIES_KEY,
                    list_raw_dropped_entities,
                )
            # Normal return for iterable input values.
            return list_str_canonical_categories
        logger.warning(
            PII_ALLOWED_ENTITIES_INVALID_TYPE_WARNING_MESSAGE,
            ALLOWED_ENTITIES_KEY,
            type(value).__name__,
        )
        # Early return because unsupported types are treated as empty allow-lists.
        return []

    @field_validator(ENTITY_LABEL_MAP_KEY, mode="before")
    @classmethod
    def _normalize_entity_label_map(cls, value: Any) -> dict[str, str]:
        """
        Coerces entity label map values into a normalized dictionary.

        Args:
            value: Raw entity-label mapping input from YAML.

        Returns:
            Dictionary with upper-case keys and stripped string labels.
        """
        if value is None:
            # Early return because mapping is omitted.
            return dict(DICT_STR_DEFAULT_ENTITY_LABEL_MAP)
        if not isinstance(value, dict):
            logger.warning(
                PII_ENTITY_LABEL_MAP_INVALID_TYPE_WARNING_MESSAGE,
                ENTITY_LABEL_MAP_KEY,
                type(value).__name__,
            )
            # Early return because unsupported mapping types use defaults.
            return dict(DICT_STR_DEFAULT_ENTITY_LABEL_MAP)
        dict_entity_label_map: dict[str, str] = {}
        list_raw_dropped_entity_keys: list[str] = []

        # Loop through the incoming map entries and normalize key/value tokens.
        for entity, label in value.items():
            str_entity: str = cls._normalize_category_to_canonical(entity)
            if not str_entity:
                list_raw_dropped_entity_keys.append(str(entity))
                continue
            str_label: str = str(label).strip()
            dict_entity_label_map[str_entity] = str_label

        if list_raw_dropped_entity_keys:
            logger.warning(
                PII_ENTITY_LABEL_MAP_DROPPED_KEYS_WARNING_MESSAGE,
                ENTITY_LABEL_MAP_KEY,
                list_raw_dropped_entity_keys,
            )
        if dict_entity_label_map:
            # Normal return with normalized mapping values.
            return dict_entity_label_map
        # Early return because empty maps fall back to defaults.
        return dict(DICT_STR_DEFAULT_ENTITY_LABEL_MAP)

    @field_validator(REDACTION_RECOGNIZERS_KEY, mode="before")
    @classmethod
    def _normalize_redaction_recognizers(
        cls, value: Any
    ) -> RedactionRecognizersSettingsModel:
        """
        Coerces redaction recognizer extension settings into a typed safe-default model.

        Args:
            value: Raw redaction recognizer settings input from YAML.

        Returns:
            A normalized RedactionRecognizersSettingsModel value.
        """
        if value is None:
            logger.info(
                PII_RECOGNIZERS_MISSING_INFO_MESSAGE,
                REDACTION_RECOGNIZERS_KEY,
            )
            # Early return because extension settings are omitted.
            return RedactionRecognizersSettingsModel()
        if isinstance(value, RedactionRecognizersSettingsModel):
            # Normal return when value is already typed.
            return value
        if not isinstance(value, dict):
            logger.warning(
                PII_RECOGNIZERS_INVALID_TYPE_WARNING_MESSAGE,
                REDACTION_RECOGNIZERS_KEY,
                type(value).__name__,
            )
            # Early return because unsupported value types map to defaults.
            return RedactionRecognizersSettingsModel()
        normalized_recognizer_settings: RedactionRecognizersSettingsModel = (
            RedactionRecognizersSettingsModel.model_validate(value)
        )
        # Normal return with normalized recognizer settings.
        return normalized_recognizer_settings


class ObservabilitySettingsModel(BaseModel):
    """
    Typed model for logger-backed observability middleware settings.
    """

    model_config: ConfigDict = ConfigDict(extra="allow", str_strip_whitespace=True)

    direction: str = DIRECTION_INPUT_OUTPUT
    capabilities: list[str] = Field(default_factory=list)
    log_level: str = DEFAULT_OBSERVABILITY_LOG_LEVEL
    token_count_mode: str = DEFAULT_OBSERVABILITY_TOKEN_COUNT_MODE
    include_media_details: bool = True
    include_provider_usage: bool = True
    include_audio_byte_count: bool = True
    include_image_byte_count: bool = True
    emit_error_events: bool = True

    @field_validator(DIRECTION_KEY, mode="before")
    @classmethod
    def _normalize_direction(cls, value: Any) -> str:
        """
        Normalizes the observability direction setting into a supported value.

        Args:
            value: Raw direction value from YAML.

        Returns:
            Supported direction value for observability middleware execution.
        """
        if isinstance(value, str):
            str_direction: str = value.strip().lower()
            if str_direction in SET_STR_DIRECTION_VALUES:
                # Normal return with normalized direction.
                return str_direction
        logger.warning(
            OBSERVABILITY_SETTINGS_INVALID_VALUE_WARNING_MESSAGE,
            DIRECTION_KEY,
            value,
        )
        # Early return because invalid direction values fall back to the default.
        return DIRECTION_INPUT_OUTPUT

    @field_validator(CAPABILITIES_KEY, mode="before")
    @classmethod
    def _normalize_capabilities(cls, value: Any) -> list[str]:
        """
        Normalizes the observability capability allow-list into supported values.

        Args:
            value: Raw capability value from YAML.

        Returns:
            List of normalized supported capability identifiers.
        """
        if value is None:
            # Early return because omitted capability settings imply all capabilities.
            return []
        list_raw_values: list[Any]
        if isinstance(value, str):
            list_raw_values = [value]
        elif isinstance(value, (list, tuple)):
            list_raw_values = list(value)
        elif isinstance(value, set):
            # Sort set values to keep normalized capability ordering deterministic across processes.
            list_raw_values = sorted(value)
        else:
            logger.warning(
                OBSERVABILITY_CAPABILITIES_INVALID_TYPE_WARNING_MESSAGE,
                CAPABILITIES_KEY,
                type(value).__name__,
            )
            # Early return because unsupported types fall back to the default.
            return []

        list_str_capabilities: list[str] = []
        set_seen_capabilities: set[str] = set()
        list_str_dropped_values: list[str] = []
        # Loop through raw capability values and keep only supported normalized items.
        for raw_value in list_raw_values:
            if not isinstance(raw_value, str):
                list_str_dropped_values.append(str(raw_value))
                continue
            str_capability: str = raw_value.strip().lower()
            if not str_capability:
                continue
            if str_capability not in SET_STR_OBSERVABILITY_CAPABILITY_VALUES:
                list_str_dropped_values.append(raw_value)
                continue
            if str_capability in set_seen_capabilities:
                continue
            set_seen_capabilities.add(str_capability)
            list_str_capabilities.append(str_capability)
        if list_str_dropped_values:
            logger.warning(
                OBSERVABILITY_CAPABILITIES_DROPPED_VALUES_WARNING_MESSAGE,
                CAPABILITIES_KEY,
                list_str_dropped_values,
            )
        # Normal return with normalized capability values.
        return list_str_capabilities

    @field_validator(LOG_LEVEL_KEY, mode="before")
    @classmethod
    def _normalize_log_level(cls, value: Any) -> str:
        """
        Normalizes the configured log level into a supported standard logging level.

        Args:
            value: Raw log-level value from YAML.

        Returns:
            Supported uppercase log-level name.
        """
        if isinstance(value, str):
            str_log_level: str = value.strip().upper()
            if str_log_level in SET_STR_OBSERVABILITY_LOG_LEVEL_VALUES:
                # Normal return with normalized log-level value.
                return str_log_level
        logger.warning(
            OBSERVABILITY_SETTINGS_INVALID_VALUE_WARNING_MESSAGE,
            LOG_LEVEL_KEY,
            value,
        )
        # Early return because invalid log levels fall back to INFO.
        return DEFAULT_OBSERVABILITY_LOG_LEVEL

    @field_validator(TOKEN_COUNT_MODE_KEY, mode="before")
    @classmethod
    def _normalize_token_count_mode(cls, value: Any) -> str:
        """
        Normalizes token-count mode values into the supported observability modes.

        Args:
            value: Raw token-count mode value from YAML.

        Returns:
            Supported normalized token-count mode value.
        """
        if isinstance(value, str):
            str_token_count_mode: str = value.strip().lower()
            if str_token_count_mode in SET_STR_TOKEN_COUNT_MODE_VALUES:
                # Normal return with normalized token-count mode.
                return str_token_count_mode
        logger.warning(
            OBSERVABILITY_SETTINGS_INVALID_VALUE_WARNING_MESSAGE,
            TOKEN_COUNT_MODE_KEY,
            value,
        )
        # Early return because invalid token-count modes fall back to the default.
        return DEFAULT_OBSERVABILITY_TOKEN_COUNT_MODE


class MiddlewareConfig:
    """
    Parses and serves typed middleware configuration data.
    """

    def __init__(self) -> None:
        """
        Initializes the loader and materializes a typed middleware profile.

        Args:
            None

        Returns:
            None
        """
        self._configuration: MiddlewareConfigurationModel = (
            MiddlewareConfigurationModel()
        )
        self._load_configuration()

    def _load_configuration(self) -> None:
        """
        Loads configuration from YAML when a valid configuration path is provided.

        Args:
            None

        Returns:
            None
        """
        env: EnvSettings = EnvSettings()
        str_configuration_path: str | None = env.get_setting(ENV_CONFIGURATION_PATH)

        if str_configuration_path is None:
            self._configuration = MiddlewareConfigurationModel()
            # Early return because no YAML config path was provided.
            return

        if not os.path.exists(str_configuration_path):
            logger.warning(
                MIDDLEWARE_PATH_SET_BUT_MISSING_WARNING_MESSAGE,
                ENV_CONFIGURATION_PATH,
                str_configuration_path,
            )
            self._configuration = MiddlewareConfigurationModel()
            # Early return because YAML config path was provided but file is missing.
            return

        try:
            with open(
                str_configuration_path,
                FILE_READ_MODE,
                encoding=FILE_ENCODING,
            ) as file:
                loaded_configuration: Any = yaml.safe_load(file)
        except yaml.YAMLError as exception:
            logger.error(
                MIDDLEWARE_YAML_PARSE_ERROR_MESSAGE,
                str_configuration_path,
                exception,
            )
            self._configuration = MiddlewareConfigurationModel()
            return
        except UnicodeDecodeError as exception:
            logger.error(
                MIDDLEWARE_ENCODING_ERROR_MESSAGE,
                str_configuration_path,
                FILE_ENCODING,
                exception,
            )
            self._configuration = MiddlewareConfigurationModel()
            return
        except OSError as exception:
            logger.error(
                MIDDLEWARE_FILE_READ_ERROR_MESSAGE,
                str_configuration_path,
                exception,
            )
            self._configuration = MiddlewareConfigurationModel()
            return

        if not isinstance(loaded_configuration, dict):
            if loaded_configuration is None:
                logger.warning(
                    MIDDLEWARE_EMPTY_FILE_WARNING_MESSAGE,
                    str_configuration_path,
                )
                self._configuration = MiddlewareConfigurationModel()
                return
            logger.error(
                "Loaded middleware config root must be a dictionary; "
                "falling back to empty config. Path: %s",
                str_configuration_path,
            )
            self._configuration = MiddlewareConfigurationModel()
            return

        list_middlewares: Any = loaded_configuration.get(MIDDLEWARE_KEY, [])
        if not isinstance(list_middlewares, list):
            logger.error(MIDDLEWARE_LIST_TYPE_ERROR_MESSAGE)
            self._configuration = MiddlewareConfigurationModel()
            return

        list_entries: list[MiddlewareEntryModel] = []

        # Loop through raw middleware entries and validate each independently.
        for middleware in list_middlewares:
            try:
                middleware_entry: MiddlewareEntryModel = (
                    MiddlewareEntryModel.model_validate(middleware)
                )
            except ValidationError as exception:
                logger.warning(
                    MIDDLEWARE_INVALID_ENTRY_MESSAGE,
                    middleware,
                    exception,
                )
                continue
            list_entries.append(middleware_entry)

        self._configuration = MiddlewareConfigurationModel(list_middleware=list_entries)
        logger.debug("Loaded middleware config from %s", str_configuration_path)
        # Normal return after loading YAML middleware configuration.
        return

    def _get_component_entry(self, name: str) -> MiddlewareEntryModel | None:
        """
        Retrieves an enabled component entry by name.

        Args:
            name: Middleware component name.

        Returns:
            The matching enabled middleware entry, or None when missing/disabled.
        """
        # Loop through middleware entries to locate the requested component.
        for entry in self._configuration.list_middleware:
            if entry.name != name:
                continue
            if not entry.enabled:
                # Early return because the matched middleware component is disabled.
                return None
            # Normal return with enabled middleware entry.
            return entry
        # Normal return because no matching component was found.
        return None

    def get_pii_redaction_settings(self) -> PiiRedactionSettingsModel | None:
        """
        Retrieves typed PII redaction settings when the component is effectively enabled.

        Args:
            None

        Returns:
            A typed PiiRedactionSettingsModel when enabled with valid, non-empty settings.
            Returns None for missing middleware, disabled middleware, missing settings,
            empty settings dictionaries, non-dictionary settings, or validation errors.
        """
        entry: MiddlewareEntryModel | None = self._get_component_entry(PII_REDACTION)
        if entry is None:
            # Early return because no enabled PII middleware entry exists.
            return None

        settings: Any = entry.settings
        if not isinstance(settings, dict) or not settings:
            logger.warning(
                MIDDLEWARE_EMPTY_SETTINGS_DISABLED_MESSAGE,
                PII_REDACTION,
            )
            # Early return because effective settings are missing/invalid.
            return None
        if REDACTION_RECOGNIZERS_KEY not in settings:
            logger.info(
                PII_RECOGNIZERS_MISSING_INFO_MESSAGE,
                REDACTION_RECOGNIZERS_KEY,
            )

        try:
            pii_settings: PiiRedactionSettingsModel = (
                PiiRedactionSettingsModel.model_validate(settings)
            )
            # Normal return with typed, validated PII settings.
            return pii_settings
        except ValidationError as exception:
            logger.error(PII_SETTINGS_VALIDATION_ERROR_MESSAGE, exception)
            # Early return because invalid settings disable the component.
            return None

    def get_observability_settings(self) -> ObservabilitySettingsModel | None:
        """
        Retrieves typed observability settings when the component is effectively enabled.

        Args:
            None

        Returns:
            A typed ObservabilitySettingsModel when enabled with valid settings or
            defaultable empty settings. Returns None for missing middleware, disabled
            middleware, unsupported settings types, or validation errors.
        """
        entry: MiddlewareEntryModel | None = self._get_component_entry(OBSERVABILITY)
        if entry is None:
            # Early return because no enabled observability middleware entry exists.
            return None

        settings: Any = entry.settings
        if settings is None:
            # Normal return because enabled observability defaults all settings.
            return ObservabilitySettingsModel()
        if not isinstance(settings, dict):
            logger.warning(
                MIDDLEWARE_EMPTY_SETTINGS_DISABLED_MESSAGE,
                OBSERVABILITY,
            )
            # Early return because effective settings are not dictionary-compatible.
            return None

        try:
            observability_settings: ObservabilitySettingsModel = (
                ObservabilitySettingsModel.model_validate(settings)
            )
            # Normal return with typed, validated observability settings.
            return observability_settings
        except ValidationError as exception:
            logger.error(OBSERVABILITY_SETTINGS_VALIDATION_ERROR_MESSAGE, exception)
            # Early return because invalid settings disable the component.
            return None

    def get_middleware_settings(
        self, name: str
    ) -> PiiRedactionSettingsModel | ObservabilitySettingsModel | None:
        """
        Retrieves typed settings for supported middleware components.

        Args:
            name: Middleware component name.

        Returns:
            A typed settings object for supported middleware components.
            Returns None when the component is unsupported, missing, disabled,
            or invalid for runtime use.
        """
        if name == PII_REDACTION:
            # Normal return for typed PII middleware settings.
            return self.get_pii_redaction_settings()
        if name == OBSERVABILITY:
            # Normal return for typed observability middleware settings.
            return self.get_observability_settings()
        logger.debug(UNSUPPORTED_TYPED_MIDDLEWARE_DEBUG_MESSAGE, name)
        # Early return because the requested middleware has no typed settings model.
        return None
