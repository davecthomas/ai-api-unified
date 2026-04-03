from __future__ import annotations

import logging
import os
import statistics
import time
from dataclasses import dataclass

import pytest

from ai_api_unified.middleware.impl.base_redactor import (
    BaseRedactorLayer,
    RedactionResult,
)
from ai_api_unified.middleware.middleware_config import (
    PiiRedactionSettingsModel,
)
from ai_api_unified.middleware.pii_redactor import (
    AiApiPiiMiddleware,
    PII_REDACTION,
)

AUDIT_LOGGER_NAME: str = "ai_api_unified.middleware.audit"
METRICS_LOGGER_NAME: str = "ai_api_unified.middleware.metrics"
PRESIDIO_REDACTOR_LOGGER_NAME: str = (
    "ai_api_unified.middleware.impl._presidio_redactor"
)
PRESIDIO_ANALYZER_LOGGER_NAME: str = "presidio-analyzer"
INPUT_DIRECTION: str = "input_only"
OUTPUT_DIRECTION: str = "output_only"
DETECTION_PROFILE_BALANCED: str = "balanced"
DETECTION_PROFILE_HIGH_ACCURACY: str = "high_accuracy"
DETECTION_PROFILE_LOW_MEMORY: str = "low_memory"
TEXT_RAW_INPUT: str = "Claire Foster"
TEXT_RAW_OUTPUT: str = "The patient is Claire Foster."
TEXT_REDACTED_NAME: str = "[REDACTED:NAME]"
TEXT_REDACTED_OUTPUT: str = "The patient is [REDACTED:NAME]."
RUN_PERFORMANCE_TESTS_ENV_VAR: str = "RUN_PII_MIDDLEWARE_PERFORMANCE_TESTS"
TLDEXTRACT_CACHE_ENV_VAR: str = "TLDEXTRACT_CACHE"
TLDEXTRACT_CACHE_PATH: str = "/tmp/ai_api_unified_tldextract_cache"
PERFORMANCE_ITERATION_COUNT: int = 10
PERFORMANCE_WARMUP_ITERATION_COUNT: int = 1
TEXT_SIMPLE_PERFORMANCE_INPUT: str = "SSN ending in 6789."
TEXT_COMPLEX_PERFORMANCE_INPUT: str = (
    "Test mixed record: Claire Foster, date of birth 1992-06-18, phone "
    "(646) 555-0121, email claire.foster@hotmail.com, Visa card ending in "
    "1129, SSN last 4 is 5542."
)
EXPECTED_SIMPLE_REDACTION_TOKEN: str = "[REDACTED:SSN]"
EXPECTED_COMPLEX_REDACTION_TOKENS: tuple[str, ...] = (
    "[REDACTED:NAME]",
    "[REDACTED:DOB]",
    "[REDACTED:PHONE]",
    "[REDACTED:EMAIL]",
    "[REDACTED:CC_LAST4]",
    "[REDACTED:SSN]",
)
EXPECTED_SIMPLE_REDACTION_COUNT: int = 1
EXPECTED_COMPLEX_REDACTION_COUNT: int = 6
EXPECTED_LOW_MEMORY_COMPLEX_REDACTION_COUNT: int = 5
EXPECTED_LOW_MEMORY_COMPLEX_REDACTION_TOKENS: tuple[str, ...] = (
    "[REDACTED:DOB]",
    "[REDACTED:PHONE]",
    "[REDACTED:EMAIL]",
    "[REDACTED:CC_LAST4]",
    "[REDACTED:SSN]",
)
SCENARIO_NAME_SIMPLE_SINGLE_ENTITY: str = "simple_single_entity"
SCENARIO_NAME_COMPLEX_MULTI_ENTITY: str = "complex_multi_entity"
LIST_TUPLE_STR_PROFILE_DISPLAY_NAME: list[tuple[str, str]] = [
    (DETECTION_PROFILE_LOW_MEMORY, "low_memory"),
    (DETECTION_PROFILE_BALANCED, "balanced"),
    (DETECTION_PROFILE_HIGH_ACCURACY, "high_accuracy"),
]
TUPLE_STR_BENCHMARK_QUIET_LOGGER_NAMES: tuple[str, ...] = (
    AUDIT_LOGGER_NAME,
    METRICS_LOGGER_NAME,
    PRESIDIO_REDACTOR_LOGGER_NAME,
    PRESIDIO_ANALYZER_LOGGER_NAME,
)


@dataclass(frozen=True)
class BenchmarkScenario:
    """
    Defines one middleware performance benchmark scenario.

    Args:
        str_scenario_name: Stable scenario identifier shown in summary output.
        str_input_text: Raw input text passed through the middleware.
        dict_str_profile_to_redaction_count: Expected redaction counts keyed by detection profile.
        dict_str_profile_to_tuple_str_expected_output_tokens: Expected sanitized output markers keyed by profile.

    Returns:
        Immutable benchmark scenario record for manual performance tests.
    """

    str_scenario_name: str
    str_input_text: str
    dict_str_profile_to_redaction_count: dict[str, int]
    dict_str_profile_to_tuple_str_expected_output_tokens: dict[str, tuple[str, ...]]


@dataclass(frozen=True)
class BenchmarkSummaryRow:
    """
    Stores one benchmark result row for human-readable summary output.

    Args:
        str_profile_name: Detection profile used in the benchmark row.
        str_scenario_name: Scenario identifier used in the benchmark row.
        float_average_elapsed_ms: Average elapsed milliseconds across iterations.
        float_median_elapsed_ms: Median elapsed milliseconds across iterations.
        float_average_ms_per_redaction: Average milliseconds per redacted entity.
        float_median_ms_per_redaction: Median milliseconds per redacted entity.

    Returns:
        Immutable summary row used for printing a side-by-side performance summary.
    """

    str_profile_name: str
    str_scenario_name: str
    float_average_elapsed_ms: float
    float_median_elapsed_ms: float
    float_average_ms_per_redaction: float
    float_median_ms_per_redaction: float


LIST_BENCHMARK_SCENARIOS: list[BenchmarkScenario] = [
    BenchmarkScenario(
        str_scenario_name=SCENARIO_NAME_SIMPLE_SINGLE_ENTITY,
        str_input_text=TEXT_SIMPLE_PERFORMANCE_INPUT,
        dict_str_profile_to_redaction_count={
            DETECTION_PROFILE_LOW_MEMORY: EXPECTED_SIMPLE_REDACTION_COUNT,
            DETECTION_PROFILE_BALANCED: EXPECTED_SIMPLE_REDACTION_COUNT,
            DETECTION_PROFILE_HIGH_ACCURACY: EXPECTED_SIMPLE_REDACTION_COUNT,
        },
        dict_str_profile_to_tuple_str_expected_output_tokens={
            DETECTION_PROFILE_LOW_MEMORY: (EXPECTED_SIMPLE_REDACTION_TOKEN,),
            DETECTION_PROFILE_BALANCED: (EXPECTED_SIMPLE_REDACTION_TOKEN,),
            DETECTION_PROFILE_HIGH_ACCURACY: (EXPECTED_SIMPLE_REDACTION_TOKEN,),
        },
    ),
    BenchmarkScenario(
        str_scenario_name=SCENARIO_NAME_COMPLEX_MULTI_ENTITY,
        str_input_text=TEXT_COMPLEX_PERFORMANCE_INPUT,
        dict_str_profile_to_redaction_count={
            DETECTION_PROFILE_LOW_MEMORY: EXPECTED_LOW_MEMORY_COMPLEX_REDACTION_COUNT,
            DETECTION_PROFILE_BALANCED: EXPECTED_COMPLEX_REDACTION_COUNT,
            DETECTION_PROFILE_HIGH_ACCURACY: EXPECTED_COMPLEX_REDACTION_COUNT,
        },
        dict_str_profile_to_tuple_str_expected_output_tokens={
            DETECTION_PROFILE_LOW_MEMORY: EXPECTED_LOW_MEMORY_COMPLEX_REDACTION_TOKENS,
            DETECTION_PROFILE_BALANCED: EXPECTED_COMPLEX_REDACTION_TOKENS,
            DETECTION_PROFILE_HIGH_ACCURACY: EXPECTED_COMPLEX_REDACTION_TOKENS,
        },
    ),
]


class FakeRedactor(BaseRedactorLayer):
    """
    Test double implementing the base redactor protocol for middleware observability tests.

    Args:
        redaction_result: Structured result returned on every sanitize call.

    Returns:
        Fake protocol implementation suitable for middleware tests.
    """

    def __init__(self, redaction_result: RedactionResult) -> None:
        """
        Stores the fixed redaction result returned by this test double.

        Args:
            redaction_result: Structured redaction result emitted by sanitize calls.

        Returns:
            None after storing the fake result payload.
        """
        self.redaction_result: RedactionResult = redaction_result

    @property
    def str_engine_cache_namespace(self) -> str:
        """
        Returns a deterministic cache namespace for this test double.

        Args:
            None

        Returns:
            Static namespace string used only for protocol compliance in tests.
        """
        # Normal return with stable fake namespace.
        return "fake_test_redactor"

    @property
    def bool_uses_shared_engine_cache(self) -> bool:
        """
        Indicates whether the fake redactor uses shared engine caches.

        Args:
            None

        Returns:
            False because the test double uses no cache-backed runtime engines.
        """
        # Normal return because the fake redactor has no engine cache.
        return False

    @property
    def tuple_str_engine_cache_identity(self) -> tuple[str, ...]:
        """
        Returns a deterministic cache identity for protocol compliance.

        Args:
            None

        Returns:
            Single-item tuple identifying the fake redactor.
        """
        # Normal return with stable fake cache identity.
        return (self.str_engine_cache_namespace,)

    def sanitize_with_result(self, str_text: str) -> RedactionResult:
        """
        Returns the fixed test result regardless of input text.

        Args:
            str_text: Raw text passed by the middleware under test.

        Returns:
            The stored RedactionResult configured for this fake instance.
        """
        # Normal return with the fixed fake redaction result payload.
        return self.redaction_result

    def sanitize_text(self, str_text: str) -> str:
        """
        Returns only the sanitized text from the fixed fake result.

        Args:
            str_text: Raw text passed by the middleware under test.

        Returns:
            Sanitized text value from the stored RedactionResult.
        """
        # Normal return with the fixed sanitized text payload.
        return self.redaction_result.str_sanitized_text


def _install_real_pii_middleware_settings(
    monkeypatch: pytest.MonkeyPatch,
    middleware_settings: PiiRedactionSettingsModel,
) -> None:
    """
    Replaces middleware config lookup while keeping the real redactor construction path active.

    Args:
        monkeypatch: Pytest monkeypatch fixture used to override runtime behavior.
        middleware_settings: Typed middleware settings returned during middleware init.

    Returns:
        None after monkeypatching config lookup for pii_redaction.
    """

    def _get_fake_settings(
        self, str_middleware_name: str
    ) -> PiiRedactionSettingsModel | None:
        """
        Returns fake middleware settings for the pii_redaction middleware only.

        Args:
            self: MiddlewareConfig instance created by the middleware constructor.
            str_middleware_name: Requested middleware component name.

        Returns:
            Fake typed settings for pii_redaction, otherwise None.
        """
        if str_middleware_name == PII_REDACTION:
            # Early return with fake middleware settings for pii_redaction.
            return middleware_settings
        # Normal return with no settings for other middleware names.
        return None

    monkeypatch.setattr(
        "ai_api_unified.middleware.middleware_config.MiddlewareConfig.get_middleware_settings",
        _get_fake_settings,
    )
    # Normal return after monkeypatch installation.
    return None


def _run_timed_middleware_iterations(
    middleware: AiApiPiiMiddleware,
    str_input_text: str,
    int_iteration_count: int,
) -> tuple[list[float], str]:
    """
    Runs one input string through middleware repeatedly and records elapsed milliseconds.

    Args:
        middleware: Real middleware instance under benchmark.
        str_input_text: Raw input string passed to `process_input`.
        int_iteration_count: Number of measured benchmark iterations to execute.

    Returns:
        Tuple containing elapsed milliseconds for each measured iteration and the
        final sanitized output emitted by the middleware.
    """
    list_float_elapsed_ms: list[float] = []
    str_last_output: str = ""
    # Warm the middleware once so benchmark samples are less dominated by first-run setup.
    for int_warmup_idx in range(PERFORMANCE_WARMUP_ITERATION_COUNT):
        middleware.process_input(str_input_text)
    # Loop through measured benchmark iterations and record wall-clock latency per call.
    for int_iteration_idx in range(int_iteration_count):
        float_started_at: float = time.perf_counter()
        str_last_output = middleware.process_input(str_input_text)
        float_elapsed_ms: float = (time.perf_counter() - float_started_at) * 1000.0
        list_float_elapsed_ms.append(float_elapsed_ms)
    # Normal return with measured per-iteration timing samples and the final output payload.
    return list_float_elapsed_ms, str_last_output


def _build_performance_summary_row(
    str_profile_name: str,
    benchmark_scenario: BenchmarkScenario,
    list_float_elapsed_ms: list[float],
    int_redaction_count: int,
) -> BenchmarkSummaryRow:
    """
    Computes one benchmark summary row for manual side-by-side output.

    Args:
        str_profile_name: Detection profile label used for this benchmark run.
        benchmark_scenario: Benchmark scenario definition for this summary row.
        list_float_elapsed_ms: Measured elapsed milliseconds for each benchmark iteration.
        int_redaction_count: Expected number of redactions for the benchmark input.

    Returns:
        Structured benchmark summary row for later table rendering.
    """
    float_average_elapsed_ms: float = statistics.mean(list_float_elapsed_ms)
    float_median_elapsed_ms: float = statistics.median(list_float_elapsed_ms)
    float_average_ms_per_redaction: float = (
        float_average_elapsed_ms / int_redaction_count
    )
    float_median_ms_per_redaction: float = float_median_elapsed_ms / int_redaction_count
    benchmark_summary_row: BenchmarkSummaryRow = BenchmarkSummaryRow(
        str_profile_name=str_profile_name,
        str_scenario_name=benchmark_scenario.str_scenario_name,
        float_average_elapsed_ms=float_average_elapsed_ms,
        float_median_elapsed_ms=float_median_elapsed_ms,
        float_average_ms_per_redaction=float_average_ms_per_redaction,
        float_median_ms_per_redaction=float_median_ms_per_redaction,
    )
    # Normal return with one structured summary row for later table logging.
    return benchmark_summary_row


def _log_performance_summary_table(
    list_benchmark_summary_rows: list[BenchmarkSummaryRow],
) -> None:
    """
    Prints a human-readable summary table for manual middleware benchmark review.

    Args:
        list_benchmark_summary_rows: Benchmark rows covering each profile and scenario pair.

    Returns:
        None after printing the complete performance summary table.
    """
    print()
    print("PII middleware performance summary")
    print(
        "profile         scenario                avg_ms   median_ms   avg_ms_per_redaction   median_ms_per_redaction"
    )
    print(
        "--------------  ----------------------  -------  ----------  ----------------------  ------------------------"
    )
    # Loop through each benchmark summary row and print one summary line per result.
    for benchmark_summary_row in list_benchmark_summary_rows:
        print(
            f"{benchmark_summary_row.str_profile_name:<14}  "
            f"{benchmark_summary_row.str_scenario_name:<22}  "
            f"{benchmark_summary_row.float_average_elapsed_ms:>7.3f}  "
            f"{benchmark_summary_row.float_median_elapsed_ms:>10.3f}  "
            f"{benchmark_summary_row.float_average_ms_per_redaction:>22.3f}  "
            f"{benchmark_summary_row.float_median_ms_per_redaction:>24.3f}"
        )
    print()
    # Normal return after benchmark summary printing.
    return None


def _set_logger_levels(
    tuple_str_logger_names: tuple[str, ...], int_logger_level: int
) -> dict[str, int]:
    """
    Sets a uniform level across the supplied logger names and returns their prior levels.

    Args:
        tuple_str_logger_names: Logger names to update for the current benchmark scope.
        int_logger_level: Logging level applied to every named logger.

    Returns:
        Mapping of logger name to its previous configured level.
    """
    dict_str_previous_levels: dict[str, int] = {}
    # Loop through each requested logger and store its current level before overwriting it.
    for str_logger_name in tuple_str_logger_names:
        logger_to_update: logging.Logger = logging.getLogger(str_logger_name)
        dict_str_previous_levels[str_logger_name] = logger_to_update.level
        logger_to_update.setLevel(int_logger_level)
    # Normal return with previous levels so the caller can restore them later.
    return dict_str_previous_levels


def _restore_logger_levels(dict_str_previous_levels: dict[str, int]) -> None:
    """
    Restores logger levels previously captured by `_set_logger_levels`.

    Args:
        dict_str_previous_levels: Prior logger levels keyed by logger name.

    Returns:
        None after all logger levels are restored.
    """
    # Loop through each captured logger level and restore the original value after the benchmark.
    for str_logger_name, int_previous_level in dict_str_previous_levels.items():
        logging.getLogger(str_logger_name).setLevel(int_previous_level)
    # Normal return after logger level restoration.
    return None


def _install_fake_pii_middleware(
    monkeypatch: pytest.MonkeyPatch,
    middleware_settings: PiiRedactionSettingsModel,
    redactor: BaseRedactorLayer,
) -> None:
    """
    Replaces middleware config lookup and redactor construction with test doubles.

    Args:
        monkeypatch: Pytest monkeypatch fixture used to override runtime behavior.
        middleware_settings: Typed middleware settings returned during middleware init.
        redactor: Fake redactor returned by the middleware factory hook.

    Returns:
        None after monkeypatching config lookup and redactor construction.
    """

    def _get_fake_settings(
        self, str_middleware_name: str
    ) -> PiiRedactionSettingsModel | None:
        """
        Returns fake middleware settings for the pii_redaction middleware only.

        Args:
            self: MiddlewareConfig instance created by the middleware constructor.
            str_middleware_name: Requested middleware component name.

        Returns:
            Fake typed settings for pii_redaction, otherwise None.
        """
        if str_middleware_name == PII_REDACTION:
            # Early return with fake middleware settings for pii_redaction.
            return middleware_settings
        # Normal return with no settings for other middleware names.
        return None

    def _build_fake_redactor(
        configuration: PiiRedactionSettingsModel,
        bool_strict_mode: bool,
    ) -> BaseRedactorLayer:
        """
        Returns the fake redactor supplied by the calling test.

        Args:
            configuration: Typed middleware settings passed by the middleware constructor.
            bool_strict_mode: Strict-mode flag passed by the middleware constructor.

        Returns:
            The fake redactor instance provided by the test.
        """
        # Normal return with injected fake redactor.
        return redactor

    monkeypatch.setattr(
        "ai_api_unified.middleware.middleware_config.MiddlewareConfig.get_middleware_settings",
        _get_fake_settings,
    )
    monkeypatch.setattr(
        "ai_api_unified.middleware.pii_redactor._get_configured_redactor",
        _build_fake_redactor,
    )
    # Normal return after monkeypatch installation.
    return None


def test_pii_middleware_logs_audit_and_timing_for_redacted_input(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """
    Ensures input-side redaction emits info-level audit logs and info-level timing logs.

    Args:
        monkeypatch: Pytest monkeypatch fixture used to inject fake middleware dependencies.
        caplog: Pytest log capture fixture used to assert emitted observability logs.

    Returns:
        None for normal test completion.
    """
    middleware_settings: PiiRedactionSettingsModel = (
        PiiRedactionSettingsModel.model_validate({"direction": INPUT_DIRECTION})
    )
    fake_redaction_result: RedactionResult = RedactionResult(
        str_sanitized_text=TEXT_REDACTED_NAME,
        list_str_detected_categories=["NAME"],
        int_redaction_count=1,
    )
    fake_redactor: FakeRedactor = FakeRedactor(fake_redaction_result)
    _install_fake_pii_middleware(
        monkeypatch=monkeypatch,
        middleware_settings=middleware_settings,
        redactor=fake_redactor,
    )
    caplog.set_level(logging.INFO, logger=AUDIT_LOGGER_NAME)
    caplog.set_level(logging.INFO, logger=METRICS_LOGGER_NAME)

    middleware: AiApiPiiMiddleware = AiApiPiiMiddleware()
    str_output: str = middleware.process_input(TEXT_RAW_INPUT)

    assert str_output == TEXT_REDACTED_NAME
    list_str_audit_messages: list[str] = [
        log_record.getMessage()
        for log_record in caplog.records
        if log_record.name == AUDIT_LOGGER_NAME
    ]
    list_str_metrics_messages: list[str] = [
        log_record.getMessage()
        for log_record in caplog.records
        if log_record.name == METRICS_LOGGER_NAME
    ]
    assert any(
        "security_control_applied" in str_logged_message
        and "control=pii_redaction" in str_logged_message
        and "direction=input" in str_logged_message
        and "redaction_count=1" in str_logged_message
        and "categories=('NAME',)" in str_logged_message
        for str_logged_message in list_str_audit_messages
    )
    assert any(
        "middleware_execution_timing" in str_logged_message
        and "direction=input" in str_logged_message
        and "redaction_count=1" in str_logged_message
        and "ms_per_redaction=" in str_logged_message
        for str_logged_message in list_str_metrics_messages
    )


def test_pii_middleware_logs_only_timing_when_no_redactions_occur_on_output(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """
    Ensures output-side middleware emits timing logs without audit logs when nothing is redacted.

    Args:
        monkeypatch: Pytest monkeypatch fixture used to inject fake middleware dependencies.
        caplog: Pytest log capture fixture used to assert emitted observability logs.

    Returns:
        None for normal test completion.
    """
    middleware_settings: PiiRedactionSettingsModel = (
        PiiRedactionSettingsModel.model_validate({"direction": OUTPUT_DIRECTION})
    )
    fake_redaction_result: RedactionResult = RedactionResult(
        str_sanitized_text=TEXT_RAW_OUTPUT,
        list_str_detected_categories=[],
        int_redaction_count=0,
    )
    fake_redactor: FakeRedactor = FakeRedactor(fake_redaction_result)
    _install_fake_pii_middleware(
        monkeypatch=monkeypatch,
        middleware_settings=middleware_settings,
        redactor=fake_redactor,
    )
    caplog.set_level(logging.INFO, logger=AUDIT_LOGGER_NAME)
    caplog.set_level(logging.INFO, logger=METRICS_LOGGER_NAME)

    middleware: AiApiPiiMiddleware = AiApiPiiMiddleware()
    str_output: str = middleware.process_output(TEXT_RAW_OUTPUT)

    assert str_output == TEXT_RAW_OUTPUT
    list_str_audit_messages: list[str] = [
        log_record.getMessage()
        for log_record in caplog.records
        if log_record.name == AUDIT_LOGGER_NAME
    ]
    list_str_metrics_messages: list[str] = [
        log_record.getMessage()
        for log_record in caplog.records
        if log_record.name == METRICS_LOGGER_NAME
    ]
    assert list_str_audit_messages == []
    assert any(
        "middleware_execution_timing" in str_logged_message
        and "direction=output" in str_logged_message
        and "redaction_count=0" in str_logged_message
        and "ms_per_redaction=none" in str_logged_message
        for str_logged_message in list_str_metrics_messages
    )


@pytest.mark.skipif(
    os.getenv(RUN_PERFORMANCE_TESTS_ENV_VAR) != "1",
    reason=(
        "Manual-only performance benchmark. Set "
        "RUN_PII_MIDDLEWARE_PERFORMANCE_TESTS=1 to enable."
    ),
)
def test_pii_middleware_manual_performance_summary_for_simple_and_complex_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Runs a manual middleware benchmark and logs average and median timing summaries.

    Args:
        monkeypatch: Pytest monkeypatch fixture used to inject controlled middleware settings.

    Returns:
        None for normal benchmark completion after logging summary statistics.
    """
    monkeypatch.setenv(TLDEXTRACT_CACHE_ENV_VAR, TLDEXTRACT_CACHE_PATH)
    dict_str_previous_logger_levels: dict[str, int] = _set_logger_levels(
        tuple_str_logger_names=TUPLE_STR_BENCHMARK_QUIET_LOGGER_NAMES,
        int_logger_level=logging.WARNING,
    )
    list_benchmark_summary_rows: list[BenchmarkSummaryRow] = []

    try:
        # Loop through each supported benchmark profile so the manual output shows side-by-side performance.
        for (
            str_detection_profile,
            str_profile_display_name,
        ) in LIST_TUPLE_STR_PROFILE_DISPLAY_NAME:
            middleware_settings: PiiRedactionSettingsModel = (
                PiiRedactionSettingsModel.model_validate(
                    {
                        "direction": INPUT_DIRECTION,
                        "detection_profile": str_detection_profile,
                        "address_detection_enabled": False,
                    }
                )
            )
            _install_real_pii_middleware_settings(
                monkeypatch=monkeypatch,
                middleware_settings=middleware_settings,
            )
            middleware: AiApiPiiMiddleware = AiApiPiiMiddleware()

            # Loop through each benchmark scenario and collect timing plus output assertions for this profile.
            for benchmark_scenario in LIST_BENCHMARK_SCENARIOS:
                (
                    list_float_elapsed_ms,
                    str_sanitized_output,
                ) = _run_timed_middleware_iterations(
                    middleware=middleware,
                    str_input_text=benchmark_scenario.str_input_text,
                    int_iteration_count=PERFORMANCE_ITERATION_COUNT,
                )
                benchmark_summary_row: BenchmarkSummaryRow = (
                    _build_performance_summary_row(
                        str_profile_name=str_profile_display_name,
                        benchmark_scenario=benchmark_scenario,
                        list_float_elapsed_ms=list_float_elapsed_ms,
                        int_redaction_count=benchmark_scenario.dict_str_profile_to_redaction_count[
                            str_detection_profile
                        ],
                    )
                )
                list_benchmark_summary_rows.append(benchmark_summary_row)

                assert len(list_float_elapsed_ms) == PERFORMANCE_ITERATION_COUNT
                # Loop through expected output markers and verify the benchmark output still redacts the intended entities.
                for (
                    str_expected_output_token
                ) in benchmark_scenario.dict_str_profile_to_tuple_str_expected_output_tokens[
                    str_detection_profile
                ]:
                    assert str_expected_output_token in str_sanitized_output
    finally:
        _restore_logger_levels(dict_str_previous_levels=dict_str_previous_logger_levels)

    _log_performance_summary_table(list_benchmark_summary_rows)
