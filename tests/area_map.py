# area_map.py
# ============================================================================
# AI AGENT REMINDER — TEST SELECTION POLICY (read before running tests)
#
# During development, run only the test areas impacted by your change:
#
#     poetry run python scripts/run_impacted_tests.py
#
# The FULL mocked regression suite is REQUIRED (and enforced by publish.sh)
# before tagging or publishing a release:
#
#     poetry run pytest -q -m "not nonmock"
#
# This module is the single source of truth for both sides of that policy:
# which areas each test file belongs to, and which areas each source path
# impacts. Every new test file MUST be added to DICT_TEST_FILE_AREAS —
# conftest.py fails collection for unmapped files so the map stays complete.
# ============================================================================
"""
Test-area registry mapping test files to areas and source paths to areas.

Areas become pytest markers with the `area_` prefix (for example
`area_engine_openai`), applied automatically by conftest.py, so any subset
can be run directly:

    poetry run pytest -m "area_middleware and not nonmock"
"""

from __future__ import annotations

AREA_MARKER_PREFIX: str = "area_"

# Sentinel meaning "this change impacts everything; run the full suite".
ALL_AREAS: str = "__all__"

AREAS: tuple[str, ...] = (
    "core",
    "completions",
    "engine_anthropic",
    "engine_openai",
    "engine_gemini",
    "engine_bedrock",
    "middleware",
    "pricing",
    "embeddings",
    "classification",
    "images",
    "videos",
    "voice",
)

# Test file basename -> areas it exercises. A file may carry several areas;
# it runs when any of them is impacted.
DICT_TEST_FILE_AREAS: dict[str, tuple[str, ...]] = {
    # Core factory/registry/config surface
    "test_ai_api.py": ("core",),
    "test_env_settings_extra_keys.py": ("core",),
    "test_models_config.py": ("core", "completions", "embeddings", "images"),
    "test_models_env_overlay.py": ("core", "completions", "embeddings"),
    "test_factory_config_errors.py": ("core", "completions", "embeddings", "images"),
    "test_migrate2json.py": ("core",),
    "test_model_policy_nonmock.py": ("core", "completions"),
    "test_model_check_nonmock.py": ("core", "completions"),
    "test_ai_model_checker.py": ("core",),
    "test_ai_model_probes.py": (
        "core",
        "engine_openai",
        "engine_anthropic",
        "engine_gemini",
        "engine_bedrock",
    ),
    "test_configured_model_check.py": ("core", "completions"),
    "test_configured_models_nonmock.py": (
        "core",
        "completions",
        "engine_openai",
        "engine_anthropic",
        "engine_gemini",
    ),
    "test_base_url_override.py": (
        "core",
        "engine_anthropic",
        "engine_openai",
        "engine_gemini",
    ),
    "test_ai_factory_provider_loading.py": ("core",),
    "test_ai_provider_loader.py": ("core",),
    "test_ai_provider_registry.py": ("core",),
    "test_env_settings.py": ("core",),
    "test_package_metadata_validation.py": ("core",),
    "test_package_optional_import_safety.py": ("core",),
    "test_version_sync.py": ("core",),
    # Anthropic (native claude engine)
    "test_anthropic_batches.py": ("completions", "engine_anthropic"),
    "test_anthropic_completions.py": ("completions", "engine_anthropic"),
    "test_completions_conversation_api.py": ("completions", "engine_anthropic"),
    # OpenAI engines
    "test_openai_responses_completions.py": ("completions", "engine_openai"),
    "test_openai_us_endpoint.py": ("completions", "engine_openai"),
    # Google Gemini engine
    "test_google_gemini.py": ("completions", "engine_gemini", "embeddings"),
    "test_google_service_account_key.py": ("engine_gemini", "voice"),
    "test_gemini_vertex_location.py": ("engine_gemini", "completions"),
    "test_google_gemini_nonmock.py": ("completions", "engine_gemini", "embeddings"),
    # Multi-engine completions surfaces
    "test_cached_token_capture.py": (
        "completions",
        "engine_anthropic",
        "engine_openai",
        "engine_gemini",
        "engine_bedrock",
    ),
    "test_completions_streaming.py": (
        "completions",
        "engine_openai",
        "engine_gemini",
        "engine_bedrock",
    ),
    "test_image_input_completions.py": (
        "completions",
        "engine_openai",
        "engine_gemini",
    ),
    "test_model_switch_nonmock.py": (
        "completions",
        "engine_openai",
        "engine_gemini",
        "engine_bedrock",
    ),
    "test_multi_engine_conversation_api.py": (
        "completions",
        "engine_openai",
        "engine_gemini",
        "engine_bedrock",
    ),
    # Pricing registry
    "test_model_pricing.py": ("pricing",),
    "test_readme_supported_models.py": (
        "core",
        "completions",
        "embeddings",
        "images",
        "videos",
    ),
    "test_bedrock_open_weight_models.py": ("completions", "engine_bedrock", "pricing"),
    "test_bedrock_claude_strict_schema.py": ("completions", "engine_bedrock"),
    "test_openai_compatible_completions.py": ("completions", "engine_openai"),
    "test_model_fallback_nonmock.py": (
        "core",
        "completions",
        "engine_openai",
        "engine_gemini",
    ),
    "test_model_fallback.py": (
        "core",
        "completions",
        "engine_anthropic",
        "engine_openai",
    ),
    "test_fallback_reason.py": (
        "completions",
        "engine_anthropic",
        "engine_openai",
        "engine_gemini",
        "engine_bedrock",
    ),
    "test_prompt_cache_hints.py": (
        "completions",
        "engine_anthropic",
        "engine_openai",
        "engine_gemini",
        "engine_bedrock",
    ),
    "test_model_catalog_sep_2026.py": (
        "pricing",
        "completions",
        "engine_anthropic",
        "engine_openai",
        "engine_gemini",
        "engine_bedrock",
        "embeddings",
        "images",
        "videos",
    ),
    # Middleware: observability, finops, PII
    "test_custom_recognizer_registration.py": ("middleware",),
    "test_finops_cost_observability.py": ("middleware", "pricing"),
    "test_request_cost.py": ("middleware", "pricing", "completions", "core"),
    "test_prompt_result.py": ("middleware", "pricing", "completions", "core"),
    "test_model_pin.py": ("completions", "core", "engine_anthropic"),
    "test_turn_result_envelope.py": ("middleware", "pricing", "completions", "core"),
    "test_media_typed_errors.py": ("core", "embeddings", "middleware"),
    "test_capability_startup_checks.py": ("core", "embeddings", "images", "voice"),
    "test_billed_failures.py": ("middleware", "pricing", "completions", "core"),
    "test_thinking_levels.py": (
        "completions",
        "core",
        "engine_anthropic",
        "engine_openai",
        "engine_gemini",
        "engine_bedrock",
    ),
    "test_thinking_levels_nonmock.py": (
        "completions",
        "core",
        "engine_anthropic",
        "engine_openai",
        "engine_gemini",
    ),
    "test_request_cost_nonmock.py": (
        "middleware",
        "pricing",
        "completions",
        "core",
        "engine_anthropic",
    ),
    "test_finops_org_attribution.py": (
        "middleware",
        "engine_anthropic",
        "engine_openai",
        "engine_gemini",
        "engine_bedrock",
    ),
    "test_middleware_config.py": ("middleware",),
    "test_observability_docs_release_phase_h.py": ("middleware",),
    "test_observability_log_cleanup_phase_g.py": ("middleware",),
    "test_observability_middleware_phase_a.py": ("middleware",),
    "test_observability_shared_runtime.py": ("middleware",),
    "test_pii_middleware_observability.py": ("middleware",),
    "test_pii_redactor_nonmock.py": ("middleware",),
    "test_presidio_log_control.py": ("middleware",),
    # Middleware phases that construct real capability clients
    "test_observability_completions_phase_c.py": ("middleware", "completions"),
    "test_observability_embeddings_phase_d.py": ("middleware", "embeddings"),
    "test_observability_images_phase_e.py": ("middleware", "images"),
    "test_observability_tts_phase_f.py": ("middleware", "voice"),
    # Routing
    "test_model_router_config.py": ("completions",),
    "test_model_router.py": ("completions", "core"),
    "test_route_ceiling.py": ("completions", "core"),
    "test_async_twins.py": ("completions", "core", "images", "voice"),
    "test_response_cache.py": ("middleware", "completions", "embeddings", "core"),
    "test_guardrails.py": ("middleware", "completions", "core"),
    "test_moderation_backends.py": ("middleware", "completions", "core"),
    "test_model_router_nonmock.py": (
        "completions",
        "core",
        "engine_anthropic",
        "engine_openai",
    ),
    # Classification
    "test_classification_base.py": ("classification",),
    "test_structured_output_classifier.py": ("classification", "completions"),
    "test_ai_factory_classifier.py": ("classification", "core"),
    "test_classification_nonmock.py": (
        "classification",
        "engine_anthropic",
        "engine_gemini",
        "engine_openai",
    ),
    "test_openai_decisions_classifier.py": (
        "classification",
        "engine_openai",
        "pricing",
        "middleware",
    ),
    # Embeddings
    "test_embeddings_capabilities.py": ("embeddings",),
    "test_voyage_embeddings.py": ("embeddings",),
    # Images
    "test_image_generation_files.py": ("images",),
    "test_nova_canvas_image.py": ("images", "engine_bedrock"),
    # Videos
    "test_ai_base_videos.py": ("videos",),
    "test_frame_helpers.py": ("videos",),
    "test_google_gemini_videos.py": ("videos", "engine_gemini"),
    "test_gemini_init_outage.py": ("engine_gemini", "completions"),
    "test_fallback_cooldown_nonmock.py": ("completions", "engine_gemini"),
    "test_router_options_nonmock.py": ("completions", "engine_anthropic"),
    "test_config_checks.py": ("completions",),
    "test_config_schemas.py": ("completions",),
    "test_one_config_file.py": ("completions",),
    "test_metering.py": ("completions", "engine_gemini"),
    "test_route_preview.py": ("completions",),
    "test_route_preview_nonmock.py": ("completions",),
    "test_nova_reel_videos.py": ("videos", "engine_bedrock"),
    "test_openai_videos.py": ("videos", "engine_openai"),
    "test_video_generation_nonmock.py": ("videos",),
    # Voice
    "test_audio_dependency_isolation.py": ("core", "voice", "engine_anthropic"),
    "test_ai_voice_factory_provider_loading.py": ("voice",),
    "test_ai_factory_voice_client.py": ("core", "voice"),
    "test_voice_caller_id_attribution.py": ("voice", "middleware"),
    "test_voice_env_settings.py": ("voice",),
    "test_voice_nonmock.py": ("voice",),
}

# Source path prefixes -> areas impacted by a change under that prefix.
# Rules are matched in order; the FIRST matching prefix wins, so more
# specific prefixes must come before broader ones. ALL_AREAS means the
# change is load-bearing for everything and the full suite runs.
LIST_SOURCE_AREA_RULES: list[tuple[str, tuple[str, ...] | str]] = [
    # Engine-specific completions
    (
        "src/ai_api_unified/completions/ai_anthropic_completions.py",
        ("completions", "engine_anthropic"),
    ),
    (
        "src/ai_api_unified/completions/ai_openai_completions.py",
        ("completions", "engine_openai"),
    ),
    (
        "src/ai_api_unified/completions/ai_openai_responses_completions.py",
        ("completions", "engine_openai"),
    ),
    (
        "src/ai_api_unified/completions/ai_openai_compatible_completions.py",
        ("completions", "engine_openai"),
    ),
    (
        "src/ai_api_unified/completions/ai_google_gemini_completions.py",
        ("completions", "engine_gemini"),
    ),
    (
        "src/ai_api_unified/completions/ai_google_gemini_capabilities.py",
        ("completions", "engine_gemini"),
    ),
    (
        "src/ai_api_unified/completions/ai_bedrock_completions.py",
        ("completions", "engine_bedrock"),
    ),
    # Provider bases are shared across that vendor's capabilities
    (
        "src/ai_api_unified/ai_anthropic_base.py",
        ("completions", "engine_anthropic"),
    ),
    (
        "src/ai_api_unified/ai_openai_base.py",
        ("completions", "engine_openai", "images", "videos", "voice"),
    ),
    (
        "src/ai_api_unified/ai_google_base.py",
        ("completions", "engine_gemini", "embeddings", "images", "videos", "voice"),
    ),
    (
        "src/ai_api_unified/ai_bedrock_base.py",
        ("completions", "engine_bedrock", "embeddings", "images", "videos"),
    ),
    # Capability packages
    ("src/ai_api_unified/embeddings/", ("embeddings",)),
    ("src/ai_api_unified/classification/", ("classification",)),
    ("src/ai_api_unified/routing/", ("completions", "core")),
    ("src/ai_api_unified/thinking.py", ALL_AREAS),
    (
        "src/ai_api_unified/completions/ai_router_completions.py",
        ("completions", "core"),
    ),
    ("src/ai_api_unified/images/", ("images",)),
    ("src/ai_api_unified/videos/", ("videos",)),
    ("src/ai_api_unified/voice/", ("voice",)),
    ("src/ai_api_unified/middleware/", ("middleware",)),
    ("src/ai_api_unified/pricing/", ("pricing", "middleware", "completions")),
    # Load-bearing shared modules: run everything.
    ("src/ai_api_unified/ai_base.py", ALL_AREAS),
    ("src/ai_api_unified/ai_factory.py", ALL_AREAS),
    ("src/ai_api_unified/ai_provider_loader.py", ALL_AREAS),
    ("src/ai_api_unified/ai_provider_registry.py", ALL_AREAS),
    ("src/ai_api_unified/ai_provider_exceptions.py", ALL_AREAS),
    ("src/ai_api_unified/ai_completions_exceptions.py", ALL_AREAS),
    ("src/ai_api_unified/util/", ALL_AREAS),
    ("src/ai_api_unified/__init__.py", ALL_AREAS),
    ("src/ai_api_unified/__version__.py", ("core",)),
    # Test infrastructure changes invalidate selection itself.
    ("tests/conftest.py", ALL_AREAS),
    ("tests/area_map.py", ALL_AREAS),
    ("pyproject.toml", ALL_AREAS),
    ("poetry.lock", ALL_AREAS),
]


def areas_for_test_file(str_basename: str) -> tuple[str, ...] | None:
    """
    Returns the areas mapped for one test file basename, or None if unmapped.
    """
    return DICT_TEST_FILE_AREAS.get(str_basename)


def classify_changed_path(str_path: str) -> tuple[str, ...] | str | None:
    """
    Returns the impact of one changed repo path.

    Returns a tuple of areas, ALL_AREAS for full-suite triggers, the literal
    path for a changed test file (run it directly), or None when the path has
    no test impact (docs, scripts, CI, ADRs).
    """
    if str_path.startswith("tests/input/"):
        # Shared test fixtures: no per-fixture consumer map exists, so run
        # everything rather than silently skip the tests that read them.
        return ALL_AREAS
    if str_path.startswith("tests/") and str_path.endswith(".py"):
        str_basename: str = str_path.rsplit("/", 1)[-1]
        if str_basename in ("conftest.py", "area_map.py", "__init__.py"):
            return ALL_AREAS
        # A changed test file runs directly regardless of area mapping.
        return str_path
    # Loop over ordered rules so the most specific prefix wins.
    for str_prefix, impact in LIST_SOURCE_AREA_RULES:
        if str_path.startswith(str_prefix):
            return impact
    if str_path.startswith("src/"):
        # Unknown source paths are load-bearing until mapped here.
        return ALL_AREAS
    # Docs, scripts, CI, and memory shards trigger no tests.
    return None
