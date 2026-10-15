"""
Factory utilities for selecting provider implementations by runtime configuration.

This module resolves providers through the centralized AiProvider registry and
lazy loader to avoid import-time failures when optional extras are not installed.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import logging
from collections.abc import Callable, Iterable, Sequence
from typing import Any

from .ai_base import (
    AIBaseClassifier,
    AIBaseCompletions,
    AIBaseEmbeddings,
    AIBaseImages,
    AIBaseVideos,
)
from .ai_provider_exceptions import (
    AiProviderError,
    AiFallbackReason,
    AiProviderCapabilityUnsupportedError,
    AiProviderConfigurationError,
    AiProviderDependencyUnavailableError,
    AiProviderRuntimeError,
)
from .completions.ai_router_completions import AiRouterCompletions
from .routing.router_config import (
    AIRouterClassifierConfig,
    AIRouterConfig,
    load_router_config,
    prune_router_config,
)
from .ai_models_config import (
    MODEL_POLICY_EVENT_DENIED,
    AIModelPolicy,
    AIModelsConfig,
    model_label,
)
from .util import ai_model_probes  # noqa: F401  (registers the provider probes)
from .util.ai_model_checker import (
    AI_MODEL_CHECK_KEY,
    AI_MODEL_CHECK_TIMEOUT_SECONDS_KEY,
    MODEL_CHECK_MODE_OFF,
    MODEL_CHECK_MODE_STRICT,
    AiModelChecker,
    AiModelCheckResult,
    AiModelCheckStatus,
    prune_candidates,
    resolve_model_check_mode,
)
from .completions.ai_fallback_completions import (
    DEFAULT_FALLBACK_REASONS,
    AIFallbackCandidate,
    AiFallbackCompletions,
    parse_fallback_candidates,
    parse_fallback_reasons,
)
from .ai_provider_loader import load_ai_provider_class
from .ai_provider_registry import (
    AI_PROVIDER_CAPABILITY_CLASSIFICATION,
    AiProviderSpec,
    AI_PROVIDER_CAPABILITY_COMPLETIONS,
    AI_PROVIDER_CAPABILITY_EMBEDDINGS,
    AI_PROVIDER_CAPABILITY_IMAGES,
    AI_PROVIDER_CAPABILITY_VIDEOS,
    TypeAiProviderCapability,
    get_ai_provider_spec,
)
from .util.env_settings import EnvSettings
from .voice.ai_voice_base import AIVoiceBase
from .voice.ai_voice_factory import AI_VOICE_ENGINE_ENV_KEY, AIVoiceFactory

_LOGGER: logging.Logger = logging.getLogger(__name__)

DEFAULT_COMPLETIONS_ENGINE: str = ""
DEFAULT_EMBEDDING_ENGINE: str = ""
DEFAULT_IMAGE_ENGINE: str = ""
DEFAULT_VIDEO_ENGINE: str = ""
DEFAULT_EMBEDDING_DIMENSIONS: str = "0"
COMPLETIONS_MODEL_NAME_KEY: str = "COMPLETIONS_MODEL_NAME"
COMPLETIONS_ENGINE_KEY: str = "COMPLETIONS_ENGINE"
# Ordered engine:model pairs tried after the primary fails; see
# AiFallbackCompletions. Blank means no fallback.
COMPLETIONS_FALLBACKS_KEY: str = "COMPLETIONS_FALLBACKS"
# Comma-separated AiFallbackReason values that move a request to the next
# candidate. Blank means DEFAULT_FALLBACK_REASONS.
COMPLETIONS_FALLBACK_ON_KEY: str = "COMPLETIONS_FALLBACK_ON"
COMPLETIONS_ROUTER_CONFIG_KEY: str = "COMPLETIONS_ROUTER_CONFIG"
COMPLETIONS_ROUTER_CLASSIFIER_KEY: str = "COMPLETIONS_ROUTER_CLASSIFIER"
EMBEDDING_MODEL_NAME_KEY: str = "EMBEDDING_MODEL_NAME"
EMBEDDING_ENGINE_KEY: str = "EMBEDDING_ENGINE"
EMBEDDING_DIMENSIONS_KEY: str = "EMBEDDING_DIMENSIONS"
CLASSIFIER_ENGINE_KEY: str = "CLASSIFIER_ENGINE"
CLASSIFIER_MODEL_NAME_KEY: str = "CLASSIFIER_MODEL_NAME"
# Classification engine used when CLASSIFIER_ENGINE is unset and an OpenAI
# key is present: the native Decisions API.
DEFAULT_CLASSIFIER_ENGINE_WITH_OPENAI_KEY: str = "openai-decisions"
OPENAI_API_KEY_SETTING: str = "OPENAI_API_KEY"
IMAGE_MODEL_NAME_KEY: str = "IMAGE_MODEL_NAME"
IMAGE_ENGINE_KEY: str = "IMAGE_ENGINE"
VIDEO_MODEL_NAME_KEY: str = "VIDEO_MODEL_NAME"
# Engines whose provider SDK accepts a base-URL override. Others ignore the
# argument silently, so the factory rejects it for them instead.
FROZENSET_BASE_URL_OVERRIDE_COMPLETIONS_ENGINES: frozenset[str] = frozenset(
    {"claude", "openai", "openai-responses", "openai-compatible", "google-gemini"}
)
VIDEO_ENGINE_KEY: str = "VIDEO_ENGINE"


def _is_python_module_available(str_module_name: str) -> bool:
    """
    Determines whether a Python module can be discovered without importing it.

    Args:
        str_module_name: Fully qualified module path to probe in import metadata.

    Returns:
        True when the module can be resolved by import machinery, otherwise False.
    """
    list_name_parts: list[str] = str_module_name.split(".")
    if not list_name_parts:
        # Early return because empty module names cannot be resolved.
        return False

    str_current_full_name: str = list_name_parts[0]
    try:
        module_spec: importlib.machinery.ModuleSpec | None = importlib.util.find_spec(
            str_current_full_name
        )
    except (ModuleNotFoundError, ValueError):
        # Early return because malformed or missing modules are unavailable.
        return False
    if module_spec is None:
        # Early return because top-level module is unavailable.
        return False

    # Loop over dotted module segments without importing parent packages.
    for str_name_part in list_name_parts[1:]:
        if module_spec.submodule_search_locations is None:
            # Early return because the current segment is not a package.
            return False
        str_current_full_name = f"{str_current_full_name}.{str_name_part}"
        module_spec = importlib.machinery.PathFinder.find_spec(
            str_current_full_name,
            module_spec.submodule_search_locations,
        )
        if module_spec is None:
            # Early return because dotted submodule segment was not found.
            return False

    # Normal return because module path was resolved without importing it.
    return True


def _is_ai_provider_available(
    str_capability: TypeAiProviderCapability,
    str_engine: str,
) -> bool:
    """
    Probes whether required dependency roots for a provider exist in this runtime.

    Args:
        str_capability: Capability bucket used for provider lookup.
        str_engine: Engine selector token under the capability bucket.

    Returns:
        True when all dependency roots for the provider can be discovered, otherwise False.
    """
    try:
        ai_provider_spec: AiProviderSpec = get_ai_provider_spec(
            str_capability, str_engine
        )
    except AiProviderConfigurationError:
        # Early return because provider is unavailable in this runtime.
        return False

    # Loop over dependency roots and fail-fast on the first missing dependency.
    for str_dependency_root in ai_provider_spec.set_str_dependency_roots:
        if not _is_python_module_available(str_dependency_root):
            # Early return because at least one required dependency root is unavailable.
            return False

    # Normal return because every dependency root for this provider is available.
    return True


# Legacy availability flags are eagerly defined as plain module attributes to keep
# introspection simple and avoid module-level __getattr__ indirection.
GOOGLE_GEMINI_AVAILABLE: bool = _is_ai_provider_available(
    AI_PROVIDER_CAPABILITY_COMPLETIONS,
    "google-gemini",
)
BEDROCK_AVAILABLE: bool = _is_ai_provider_available(
    AI_PROVIDER_CAPABILITY_COMPLETIONS,
    "nova",
)
TITAN_AVAILABLE: bool = _is_ai_provider_available(
    AI_PROVIDER_CAPABILITY_EMBEDDINGS,
    "titan",
)
BEDROCK_IMAGES_AVAILABLE: bool = _is_ai_provider_available(
    AI_PROVIDER_CAPABILITY_IMAGES,
    "nova-canvas",
)


class AIFactory:
    """
    Factory for creating provider clients for completions, embeddings, and images.
    """

    @staticmethod
    def _normalize_engine(
        str_engine_override: str | None, str_default_engine: str
    ) -> str:
        """
        Resolves and normalizes engine values from explicit overrides.

        Args:
            str_engine_override: Optional engine override from method caller.
            str_default_engine: Default engine fallback when override is empty.

        Returns:
            Lowercase normalized engine token.
        """
        str_engine_normalized: str = str_default_engine
        if str_engine_override is not None and str_engine_override.strip():
            str_engine_normalized = str_engine_override.strip().lower()
        # Normal return with normalized engine selector.
        return str_engine_normalized

    @staticmethod
    def _resolve_required_engine(
        *,
        env_settings: EnvSettings,
        str_engine_key: str,
        str_engine_override: str | None = None,
    ) -> str:
        """
        Resolves a required engine selector from override or environment config.

        Args:
            env_settings: Shared environment/settings accessor.
            str_engine_key: Environment variable name for the target capability.
            str_engine_override: Optional explicit caller override.

        Returns:
            Lowercase normalized engine token.

        Raises:
            ValueError: When the engine selector is not configured.
        """
        if str_engine_override is not None and str_engine_override.strip():
            return str_engine_override.strip().lower()

        object_engine_value: object = env_settings.get_setting(str_engine_key, "")
        str_engine_value: str = (
            str(object_engine_value) if object_engine_value is not None else ""
        )
        str_engine_normalized: str = str_engine_value.strip().lower()
        if not str_engine_normalized:
            raise ValueError(
                f"{str_engine_key} must be configured explicitly; there is no default provider."
            )
        return str_engine_normalized

    @staticmethod
    def _translate_config_exception(
        exception: AiProviderConfigurationError,
        str_capability: str,
        str_engine: str,
    ) -> ValueError:
        """
        Converts registry configuration misses into legacy ValueError behavior.

        Args:
            exception: Original provider configuration exception.
            str_capability: Capability bucket that failed lookup.
            str_engine: Engine token that failed lookup.

        Returns:
            ValueError matching historical unsupported-engine behavior.
        """
        if str_capability == AI_PROVIDER_CAPABILITY_COMPLETIONS:
            # Normal return with legacy completions unsupported-engine error.
            return ValueError(f"Unsupported COMPLETIONS engine: {str_engine!r}")
        if str_capability == AI_PROVIDER_CAPABILITY_EMBEDDINGS:
            # Normal return with legacy embeddings unsupported-engine error.
            return ValueError(f"Unsupported EMBEDDING engine: {str_engine}")
        if str_capability == AI_PROVIDER_CAPABILITY_IMAGES:
            # Normal return with legacy images unsupported-engine error.
            return ValueError(f"Unsupported IMAGE engine: {str_engine!r}")
        if str_capability == AI_PROVIDER_CAPABILITY_VIDEOS:
            # Normal return with legacy videos unsupported-engine error.
            return ValueError(f"Unsupported VIDEO engine: {str_engine!r}")
        # Normal return with generic unsupported engine error.
        return ValueError(str(exception))

    @staticmethod
    def get_ai_completions_client(
        model_name: str | None = None,
        completions_engine: str | None = None,
        base_url: str | None = None,
        fallbacks: Sequence[AIFallbackCandidate | tuple[str, str]] | None = None,
        fallback_on: Iterable[AiFallbackReason | str] | None = None,
        router: AIRouterConfig | None = None,
    ) -> AIBaseCompletions:
        """
        Instantiates the configured completions client.

        With a fallback chain (the `fallbacks` argument, or the
        COMPLETIONS_FALLBACKS setting when the argument is None), the result
        is an AiFallbackCompletions that retries a failed request on the
        next candidate. The primary is built here as usual; each fallback is
        built the first time a request needs it. An empty `fallbacks` list
        turns the chain off even when the setting is present.

        Args:
            model_name: Optional model override; falls back to environment config.
            completions_engine: Optional engine override; falls back to environment config.
            base_url: Optional API base-URL override for the primary.
            fallbacks: Optional ordered candidates, as AIFallbackCandidate or
                (engine, model) tuples, tried after the primary fails.
            fallback_on: Optional reasons that trigger the move to the next
                candidate; defaults to COMPLETIONS_FALLBACK_ON, then to
                UNAVAILABLE, RATE_LIMITED, QUOTA_EXHAUSTED, and AUTH_REJECTED.
            router: Optional complexity router config; defaults to the
                COMPLETIONS_ROUTER_CONFIG setting (a JSON file path or inline
                JSON). When present, the returned client classifies each
                request and routes it to a tier of models, with this engine
                and its chain as the default route.

        Returns:
            Concrete AIBaseCompletions implementation for the requested engine,
            or an AiFallbackCompletions wrapping it when a chain is configured.
            Raises ValueError for unsupported engines (including a fallback
            engine, checked up front) and RuntimeError-derived provider
            exceptions for dependency/runtime loading failures.
        """
        env_settings: EnvSettings = EnvSettings()
        str_engine: str = AIFactory._resolve_required_engine(
            env_settings=env_settings,
            str_engine_key=COMPLETIONS_ENGINE_KEY,
            str_engine_override=completions_engine,
        )

        str_model_name: str
        if model_name is None:
            str_model_name = env_settings.get_setting(COMPLETIONS_MODEL_NAME_KEY, "")
        else:
            str_model_name = model_name

        list_fallbacks: list[AIFallbackCandidate] = AIFactory._resolve_fallback_chain(
            env_settings=env_settings, fallbacks=fallbacks
        )
        # Validate every fallback engine token before building anything, so
        # a typo fails at startup rather than during an outage.
        for candidate in list_fallbacks:
            try:
                get_ai_provider_spec(
                    AI_PROVIDER_CAPABILITY_COMPLETIONS, candidate.engine
                )
            except AiProviderConfigurationError as exception:
                raise AIFactory._translate_config_exception(
                    exception, AI_PROVIDER_CAPABILITY_COMPLETIONS, candidate.engine
                ) from exception
        router_config: AIRouterConfig | None = AIFactory._resolve_router_config(
            env_settings=env_settings, router=router
        )
        if router_config is not None:
            # Loop over tier candidates so a typo fails at startup, as for fallbacks.
            for tier in router_config.tiers:
                for candidate in tier.candidates:
                    try:
                        get_ai_provider_spec(
                            AI_PROVIDER_CAPABILITY_COMPLETIONS, candidate.engine
                        )
                    except AiProviderConfigurationError as exception:
                        raise AIFactory._translate_config_exception(
                            exception,
                            AI_PROVIDER_CAPABILITY_COMPLETIONS,
                            candidate.engine,
                        ) from exception
        policy: AIModelPolicy = AIFactory._model_policy(env_settings)
        AIFactory._require_allowed(
            policy, str_engine, str_model_name, "completions model"
        )
        list_fallbacks, router_config, str_classifier_missing = (
            AIFactory._apply_model_policy(
                env_settings=env_settings,
                policy=policy,
                list_fallbacks=list_fallbacks,
                router_config=router_config,
            )
        )
        if (
            router_config is not None or list_fallbacks
        ) and str_classifier_missing is None:
            list_fallbacks, router_config, str_classifier_missing = (
                AIFactory._check_configured_models(
                    env_settings=env_settings,
                    primary_candidate=AIFallbackCandidate(
                        engine=str_engine, model=str_model_name, base_url=base_url
                    ),
                    list_fallbacks=list_fallbacks,
                    router_config=router_config,
                )
            )
        primary: AIBaseCompletions = AIFactory._build_completions_client(
            str_engine, str_model_name, base_url
        )
        frozenset_reasons: frozenset[AiFallbackReason] = (
            AIFactory._resolve_fallback_reasons(
                env_settings=env_settings, fallback_on=fallback_on
            )
        )
        default_client: AIBaseCompletions = primary
        if list_fallbacks:
            default_client = AiFallbackCompletions(
                primary=primary,
                primary_candidate=AIFallbackCandidate(
                    engine=str_engine, model=str_model_name, base_url=base_url
                ),
                fallback_candidates=list_fallbacks,
                fallback_on=frozenset_reasons,
                client_builder=lambda candidate: AIFactory._build_completions_client(
                    candidate.engine, candidate.model, candidate.base_url
                ),
            )
        if router_config is None:
            # Normal return without a router: the plain engine or its chain.
            return default_client
        classifier: AIBaseClassifier | None
        str_unavailable: str | None
        if str_classifier_missing is not None:
            classifier, str_unavailable = None, str_classifier_missing
        else:
            classifier, str_unavailable = AIFactory._resolve_router_classifier(
                env_settings=env_settings, router_config=router_config
            )
        # Normal return with the routing client over the default route.
        return AiRouterCompletions(
            default_client=default_client,
            config=router_config,
            classifier=classifier,
            client_builder=lambda candidate: AIFactory._build_completions_client(
                candidate.engine, candidate.model, candidate.base_url
            ),
            fallback_on=router_config.fallback_on or frozenset_reasons,
            str_classifier_unavailable_reason=str_unavailable,
        )

    _LIST_LAST_MODEL_CHECK: list[AiModelCheckResult] = []

    @staticmethod
    def _model_policy(env_settings: EnvSettings) -> AIModelPolicy:
        """The allow/deny policy from models.json, or an open one."""
        config: Any = env_settings.get_models_config()
        # Normal return with the file's policy, or one that allows everything
        # (also when a test double stands in for the settings).
        return config.policy if isinstance(config, AIModelsConfig) else AIModelPolicy()

    @staticmethod
    def _require_allowed(
        policy: AIModelPolicy, str_engine: str, str_model: str, str_what: str
    ) -> None:
        """
        Raises when the policy denies a model that a call needs outright.

        Args:
            policy: The allow/deny policy.
            str_engine: Engine token.
            str_model: Model id (blank for the engine default).
            str_what: What the model is, for the message.

        Raises:
            ValueError: The policy denies the model.
        """
        str_reason: str | None = policy.denial_reason(str_engine, str_model)
        if str_reason is None:
            # Normal return: permitted.
            return None
        raise ValueError(
            f"The {str_what} {model_label(str_engine, str_model)} is {str_reason} "
            "(models.json allow/deny)."
        )

    @staticmethod
    def _log_denied(
        policy: AIModelPolicy, candidate: AIFallbackCandidate, str_what: str
    ) -> bool:
        """Logs a denied candidate loudly; returns True when it is allowed."""
        str_reason: str | None = policy.denial_reason(candidate.engine, candidate.model)
        if str_reason is None:
            # Early return: permitted.
            return True
        _LOGGER.warning(
            "MODEL POLICY: configured %s %s is %s and will be ignored.",
            str_what,
            candidate.label,
            str_reason,
            extra={
                "ai_model_policy_event": MODEL_POLICY_EVENT_DENIED,
                "engine": candidate.engine,
                "model": candidate.model,
                "reason": str_reason,
            },
        )
        # Normal return: denied.
        return False

    @staticmethod
    def _apply_model_policy(
        *,
        env_settings: EnvSettings,
        policy: AIModelPolicy,
        list_fallbacks: list[AIFallbackCandidate],
        router_config: AIRouterConfig | None,
    ) -> tuple[list[AIFallbackCandidate], AIRouterConfig | None, str | None]:
        """
        Prunes denied fallback, tier, and classifier models; never raises.

        Returns:
            (pruned chain, pruned router config or None when no tier
            survived, reason the router classifier is unusable or None).
        """
        if not policy.allow and not policy.deny:
            # Early return: an open policy changes nothing.
            return list_fallbacks, router_config, None
        list_fallbacks = [
            candidate
            for candidate in list_fallbacks
            if AIFactory._log_denied(policy, candidate, "fallback model")
        ]
        str_classifier_missing: str | None = None
        if router_config is not None:
            classifier_candidate: AIFallbackCandidate | None = (
                AIFactory._router_classifier_candidate(
                    env_settings=env_settings, router_config=router_config
                )
            )
            if classifier_candidate is not None:
                str_reason: str | None = policy.denial_reason(
                    classifier_candidate.engine, classifier_candidate.model
                )
                if str_reason is not None:
                    str_classifier_missing = (
                        f"classifier model {classifier_candidate.label} is {str_reason}"
                    )
            # Loop over tiers so each denied tier model is logged once.
            for tier in router_config.tiers:
                for candidate in tier.candidates:
                    AIFactory._log_denied(
                        policy, candidate, f"tier {tier.name!r} model"
                    )
            router_config = AIFactory._prune_router_tiers(
                router_config,
                lambda candidate: policy.is_allowed(candidate.engine, candidate.model),
                str_why="denied by policy",
            )
        # Normal return with the pruned configuration.
        return list_fallbacks, router_config, str_classifier_missing

    @staticmethod
    def _prune_router_tiers(
        router_config: AIRouterConfig,
        callable_keep: Callable[[AIFallbackCandidate], bool],
        *,
        str_why: str,
    ) -> AIRouterConfig | None:
        """Prunes tiers through prune_router_config and logs what was dropped."""
        pruned: AIRouterConfig | None = prune_router_config(
            router_config, callable_keep
        )
        if pruned is None:
            _LOGGER.warning(
                "MODEL CHECK: every router tier lost all of its models (%s); "
                "routing is off and every call goes to the default route.",
                str_why,
                extra={"ai_model_check_event": "router_disabled"},
            )
            # Early return: nothing survived.
            return None
        # Loop over newly pruned tiers so each is logged once.
        for str_tier, str_next in pruned.pruned_tiers.items():
            if str_tier in router_config.pruned_tiers:
                continue
            _LOGGER.warning(
                "MODEL CHECK: router tier %r has no available model (%s) and will "
                "be ignored; requests for it go to %s.",
                str_tier,
                str_why,
                f"tier {str_next!r}" if str_next else "the default route",
                extra={
                    "ai_model_check_event": "router_tier_pruned",
                    "tier": str_tier,
                    "escalate_to": str_next,
                },
            )
        # Normal return with the pruned config.
        return pruned

    @classmethod
    def last_model_check_report(cls) -> list[AiModelCheckResult]:
        """Results of the most recent configured-model check, in check order."""
        # Normal return with a copy of the last report.
        return list(cls._LIST_LAST_MODEL_CHECK)

    @staticmethod
    def _check_configured_models(
        *,
        env_settings: EnvSettings,
        primary_candidate: AIFallbackCandidate,
        list_fallbacks: list[AIFallbackCandidate],
        router_config: AIRouterConfig | None,
    ) -> tuple[list[AIFallbackCandidate], AIRouterConfig | None, str | None]:
        """
        Verifies every configured model once and prunes the missing ones.

        Runs when a chain or a router is configured (a lone primary fails on
        its own at build). AI_MODEL_CHECK selects the mode: "prune"
        (default) drops missing fallback and tier models with a warning,
        "strict" raises on any missing model, "off" skips the check. The
        primary cannot be pruned: when it is missing the factory raises in
        every mode but "off".

        Args:
            env_settings: Shared environment/settings accessor.
            primary_candidate: The configured completions LLM.
            list_fallbacks: The configured chain.
            router_config: The configured router, if any.

        Returns:
            (pruned chain, pruned router config or None when no tier
            survived, reason the router classifier is unusable or None).

        Raises:
            ValueError: The primary is missing, or strict mode found a
                missing model.
        """
        str_mode: str = resolve_model_check_mode(
            env_settings.get_setting(AI_MODEL_CHECK_KEY, "")
        )
        if str_mode == MODEL_CHECK_MODE_OFF:
            # Early return: the check is turned off.
            return list_fallbacks, router_config, None
        list_candidates: list[AIFallbackCandidate] = [
            primary_candidate,
            *list_fallbacks,
        ]
        classifier_candidate: AIFallbackCandidate | None = None
        if router_config is not None:
            # Loop over tiers to collect every candidate.
            for tier in router_config.tiers:
                list_candidates.extend(tier.candidates)
            classifier_candidate = AIFactory._router_classifier_candidate(
                env_settings=env_settings, router_config=router_config
            )
            if classifier_candidate is not None:
                list_candidates.append(classifier_candidate)
        raw_timeout: object = env_settings.get_setting(
            AI_MODEL_CHECK_TIMEOUT_SECONDS_KEY, ""
        )
        checker: AiModelChecker = (
            AiModelChecker(float_timeout_seconds=float(str(raw_timeout)))
            if str(raw_timeout or "").strip()
            else AiModelChecker()
        )
        list_results: list[AiModelCheckResult] = checker.check_all(list_candidates)
        AIFactory._LIST_LAST_MODEL_CHECK = list(list_results)
        dict_results: dict[tuple[str, str, str | None], AiModelCheckResult] = {
            (result.engine, result.model, result.base_url): result
            for result in list_results
        }

        def _keep(candidate: AIFallbackCandidate) -> bool:
            result: AiModelCheckResult | None = dict_results.get(
                (candidate.engine, candidate.model, candidate.base_url)
            )
            return result is None or result.keeps_model

        list_missing: list[AiModelCheckResult] = [
            result
            for result in list_results
            if result.status is AiModelCheckStatus.MISSING
        ]
        if not _keep(primary_candidate):
            raise ValueError(
                f"The configured completions model {primary_candidate.label} is not "
                "available and cannot be pruned: it is the default route. Fix "
                "COMPLETIONS_ENGINE / COMPLETIONS_MODEL_NAME."
            )
        if str_mode == MODEL_CHECK_MODE_STRICT and list_missing:
            raise ValueError(
                "AI_MODEL_CHECK=strict: configured models not available: "
                + ", ".join(f"{r.label} ({r.detail})" for r in list_missing)
            )
        list_fallbacks = prune_candidates(list_fallbacks, dict_results)
        str_classifier_missing: str | None = None
        if classifier_candidate is not None and not _keep(classifier_candidate):
            str_classifier_missing = (
                f"classifier model {classifier_candidate.label} is not available"
            )
        if router_config is not None:
            router_config = AIFactory._prune_router_tiers(
                router_config, _keep, str_why="not available"
            )
        # Normal return with the pruned configuration.
        return list_fallbacks, router_config, str_classifier_missing

    @staticmethod
    def _router_classifier_candidate(
        *, env_settings: EnvSettings, router_config: AIRouterConfig
    ) -> AIFallbackCandidate | None:
        """
        The (engine, model) the router's classifier would use, resolved the
        same way _resolve_router_classifier resolves it, so the model check
        can probe it without building it.

        Returns:
            The candidate, or None when the classifier is disabled or no
            engine can be named without a build.
        """
        str_setting: str = str(
            env_settings.get_setting(COMPLETIONS_ROUTER_CLASSIFIER_KEY, "") or ""
        ).strip()
        str_engine: str | None = None
        str_model: str | None = None
        str_base_url: str | None = None
        if str_setting:
            str_engine_part, _, str_model_part = str_setting.partition(":")
            str_engine = str_engine_part.strip().lower()
            str_model = str_model_part.strip() or None
            # The setting overrides engine and model; the config's base_url
            # still applies, as in _resolve_router_classifier.
            if router_config.classifier is not None:
                str_base_url = router_config.classifier.base_url
        elif router_config.classifier is not None:
            if router_config.classifier.disabled:
                # Early return: no classifier at all.
                return None
            str_engine = router_config.classifier.engine
            str_model = router_config.classifier.model
            str_base_url = router_config.classifier.base_url
        if not str_engine:
            str_engine = str(
                env_settings.get_setting(CLASSIFIER_ENGINE_KEY, "") or ""
            ).strip().lower() or (
                DEFAULT_CLASSIFIER_ENGINE_WITH_OPENAI_KEY
                if str(
                    env_settings.get_setting(OPENAI_API_KEY_SETTING, "") or ""
                ).strip()
                else ""
            )
        if not str_model:
            # get_ai_classifier_client reads CLASSIFIER_MODEL_NAME whenever no
            # model is passed, whichever way the engine was named.
            str_model = (
                str(
                    env_settings.get_setting(CLASSIFIER_MODEL_NAME_KEY, "") or ""
                ).strip()
                or None
            )
        if not str_engine or str_engine == "none":
            # Early return: nothing to probe.
            return None
        if str_engine == DEFAULT_CLASSIFIER_ENGINE_WITH_OPENAI_KEY and not str_model:
            # Local import: the classification module needs the openai extra.
            from .classification.ai_openai_decisions_classifier import (
                DECISIONS_DEFAULT_MODEL,
            )

            str_model = DECISIONS_DEFAULT_MODEL
        # Normal return with the probe candidate.
        return AIFallbackCandidate(
            engine=str_engine, model=str_model or "", base_url=str_base_url
        )

    @staticmethod
    def _resolve_router_config(
        *, env_settings: EnvSettings, router: AIRouterConfig | None
    ) -> AIRouterConfig | None:
        """
        Resolves the router config from the argument or COMPLETIONS_ROUTER_CONFIG.

        Args:
            env_settings: Shared environment/settings accessor.
            router: Explicit config, or None to read the setting.

        Returns:
            The config, or None when no router is configured.
        """
        if router is not None:
            # Early return with the explicit config.
            return router
        str_value: str = str(
            env_settings.get_setting(COMPLETIONS_ROUTER_CONFIG_KEY, "") or ""
        ).strip()
        if not str_value:
            # Early return: no router configured.
            return None
        # Normal return with the parsed config.
        return load_router_config(str_value)

    @staticmethod
    def _resolve_router_classifier(
        *, env_settings: EnvSettings, router_config: AIRouterConfig
    ) -> tuple[AIBaseClassifier | None, str | None]:
        """
        Builds the router's classifier, or explains why there is none.

        Order: COMPLETIONS_ROUTER_CLASSIFIER, then the config's classifier,
        then the classification factory's own resolution (CLASSIFIER_ENGINE,
        then openai-decisions when OPENAI_API_KEY is set). "none" turns the
        classifier off. A build failure never raises: the router degrades to
        its default route and logs the reason.

        Args:
            env_settings: Shared environment/settings accessor.
            router_config: The router config.

        Returns:
            (classifier or None, reason it is None or None).
        """
        str_setting: str = str(
            env_settings.get_setting(COMPLETIONS_ROUTER_CLASSIFIER_KEY, "") or ""
        ).strip()
        classifier_config: AIRouterClassifierConfig | None = router_config.classifier
        if str_setting:
            str_engine_part, _, str_model_part = str_setting.partition(":")
            classifier_config = AIRouterClassifierConfig(
                engine=str_engine_part.strip().lower() or None,
                model=str_model_part.strip() or None,
                base_url=(
                    classifier_config.base_url
                    if classifier_config is not None
                    else None
                ),
                timeout_seconds=(
                    classifier_config.timeout_seconds
                    if classifier_config is not None
                    else AIRouterClassifierConfig().timeout_seconds
                ),
                instructions=(
                    classifier_config.instructions
                    if classifier_config is not None
                    else None
                ),
            )
        if classifier_config is not None and classifier_config.disabled:
            # Early return: turned off on purpose.
            return None, "disabled"
        try:
            if classifier_config is None:
                # Early return with the factory-resolved classifier.
                return AIFactory.get_ai_classifier_client(), None
            # Normal return with the configured classifier; a None engine
            # keeps the factory's resolution while the model and base_url hold.
            return (
                AIFactory.get_ai_classifier_client(
                    classifier_engine=classifier_config.engine or None,
                    model_name=classifier_config.model,
                    base_url=classifier_config.base_url,
                ),
                None,
            )
        except (ValueError, AiProviderError) as exception:
            _LOGGER.warning(
                "Router classifier unavailable (%s: %s); routing to the default route.",
                type(exception).__name__,
                exception,
            )
            # Early return: no classifier, with the reason.
            return None, f"{type(exception).__name__}: {exception}"

    @staticmethod
    def _resolve_fallback_chain(
        *,
        env_settings: EnvSettings,
        fallbacks: Sequence[AIFallbackCandidate | tuple[str, str]] | None,
    ) -> list[AIFallbackCandidate]:
        """
        Resolves the fallback chain from the argument or configuration.

        Args:
            env_settings: Shared environment/settings accessor.
            fallbacks: Explicit chain, or None to read COMPLETIONS_FALLBACKS.

        Returns:
            Candidates in order; empty when no chain is configured.
        """
        if fallbacks is None:
            object_value: object = env_settings.get_setting(
                COMPLETIONS_FALLBACKS_KEY, ""
            )
            # Normal return with the parsed setting.
            return parse_fallback_candidates(
                str(object_value) if object_value is not None else ""
            )
        # Normal return with the explicit chain, tuples normalized.
        return [
            (
                item
                if isinstance(item, AIFallbackCandidate)
                else AIFallbackCandidate(
                    engine=str(item[0]).strip().lower(), model=str(item[1]).strip()
                )
            )
            for item in fallbacks
        ]

    @staticmethod
    def _resolve_fallback_reasons(
        *,
        env_settings: EnvSettings,
        fallback_on: Iterable[AiFallbackReason | str] | None,
    ) -> frozenset[AiFallbackReason]:
        """
        Resolves the reasons that trigger a fallback.

        Args:
            env_settings: Shared environment/settings accessor.
            fallback_on: Explicit reasons, or None to read COMPLETIONS_FALLBACK_ON.

        Returns:
            The reason set; DEFAULT_FALLBACK_REASONS when nothing is configured.
        """
        if fallback_on is not None:
            # Normal return with the explicit reasons.
            return parse_fallback_reasons(fallback_on)
        object_value: object = env_settings.get_setting(COMPLETIONS_FALLBACK_ON_KEY, "")
        str_value: str = str(object_value).strip() if object_value is not None else ""
        if not str_value:
            # Early return with the default reasons.
            return DEFAULT_FALLBACK_REASONS
        # Normal return with the configured reasons.
        return parse_fallback_reasons(str_value)

    @staticmethod
    def _build_completions_client(
        str_engine: str, str_model_name: str, base_url: str | None
    ) -> AIBaseCompletions:
        """
        Builds one completions engine client.

        Args:
            str_engine: Normalized engine token.
            str_model_name: Model name; blank means the engine default.
            base_url: Optional API base-URL override.

        Returns:
            Concrete AIBaseCompletions implementation for the engine.
        """
        try:
            # Resolve provider metadata first so engine-to-module mapping is centralized
            # in the registry and does not require hardcoded imports in the factory.
            ai_provider_spec: AiProviderSpec = get_ai_provider_spec(
                AI_PROVIDER_CAPABILITY_COMPLETIONS, str_engine
            )
            # Lazy-load and validate the provider class at runtime. This is the core
            # refactor behavior that keeps optional SDK imports out of package import-time
            # while still enforcing the AIBaseCompletions interface contract.
            class_completions_client: type[AIBaseCompletions] = load_ai_provider_class(
                ai_provider_spec,
                AIBaseCompletions,
            )
            # Instantiate the resolved class only after lazy-load validation succeeds.
            dict_client_kwargs: dict[str, Any] = {"model": str_model_name}
            if base_url is not None:
                AIFactory._require_base_url_override_support(
                    str_engine=str_engine,
                    frozenset_supported=(
                        FROZENSET_BASE_URL_OVERRIDE_COMPLETIONS_ENGINES
                    ),
                )
                dict_client_kwargs["base_url"] = base_url
            completions_client: AIBaseCompletions = class_completions_client(
                **dict_client_kwargs
            )
            # Normal return with the configured completions provider client.
            return completions_client
        except AiProviderConfigurationError as exception:
            raise AIFactory._translate_config_exception(
                exception,
                AI_PROVIDER_CAPABILITY_COMPLETIONS,
                str_engine,
            ) from exception
        except AiProviderDependencyUnavailableError as exception:
            _LOGGER.warning(str(exception))
            raise
        except AiProviderRuntimeError:
            raise

    @staticmethod
    def _require_base_url_override_support(
        *,
        str_engine: str,
        frozenset_supported: frozenset[str],
    ) -> None:
        """
        Rejects a base-URL override for engines whose SDK cannot honor it.

        Args:
            str_engine: Resolved engine selector token.
            frozenset_supported: Engines whose SDK accepts a base URL.

        Raises:
            AiProviderCapabilityUnsupportedError: When the engine cannot
                honor a base-URL override.
        """
        if str_engine in frozenset_supported:
            # Normal return because this engine honors the override.
            return None
        raise AiProviderCapabilityUnsupportedError(
            f"Engine {str_engine!r} does not support a base-URL override. "
            f"Supported engines: {', '.join(sorted(frozenset_supported))}."
        )

    @staticmethod
    def get_ai_embedding_client(
        embedding_engine: str | None = None,
        model_name: str | None = None,
    ) -> AIBaseEmbeddings:
        """
        Instantiates the configured embeddings client.

        Args:
            embedding_engine: Optional engine override; falls back to environment config.
            model_name: Optional model override; falls back to environment config.

        Returns:
            Concrete AIBaseEmbeddings implementation for the requested engine.
            Raises ValueError for unsupported engines and RuntimeError-derived
            provider exceptions for dependency/runtime loading failures.
        """
        env_settings: EnvSettings = EnvSettings()
        str_engine: str = AIFactory._resolve_required_engine(
            env_settings=env_settings,
            str_engine_key=EMBEDDING_ENGINE_KEY,
            str_engine_override=embedding_engine,
        )

        str_model_name: str
        if model_name is None:
            str_model_name = env_settings.get_setting(EMBEDDING_MODEL_NAME_KEY, "")
        else:
            str_model_name = model_name

        AIFactory._require_allowed(
            AIFactory._model_policy(env_settings),
            str_engine,
            str_model_name,
            "embeddings model",
        )
        raw_embedding_dimensions: str | int | None = env_settings.get_setting(
            EMBEDDING_DIMENSIONS_KEY, DEFAULT_EMBEDDING_DIMENSIONS
        )
        int_dimensions: int = (
            0
            if raw_embedding_dimensions in (None, "")
            else int(raw_embedding_dimensions)
        )

        try:
            # Resolve provider metadata first so engine-to-module mapping is centralized
            # in the registry and does not require hardcoded imports in the factory.
            ai_provider_spec: AiProviderSpec = get_ai_provider_spec(
                AI_PROVIDER_CAPABILITY_EMBEDDINGS, str_engine
            )
            # Lazy-load and validate the provider class at runtime. This is the core
            # refactor behavior that keeps optional SDK imports out of package import-time
            # while still enforcing the AIBaseEmbeddings interface contract.
            class_embeddings_client: type[AIBaseEmbeddings] = load_ai_provider_class(
                ai_provider_spec,
                AIBaseEmbeddings,
            )
            # Instantiate the resolved class only after lazy-load validation succeeds.
            embeddings_client: AIBaseEmbeddings = class_embeddings_client(
                model=str_model_name,
                dimensions=int_dimensions,
            )
            # Normal return with configured embeddings provider client.
            return embeddings_client
        except AiProviderConfigurationError as exception:
            raise AIFactory._translate_config_exception(
                exception,
                AI_PROVIDER_CAPABILITY_EMBEDDINGS,
                str_engine,
            ) from exception
        except AiProviderDependencyUnavailableError as exception:
            _LOGGER.warning(str(exception))
            raise
        except AiProviderRuntimeError:
            raise

    @staticmethod
    def list_completion_models(client: AIBaseCompletions) -> list[str]:
        """
        Returns all completion model names supported by the given client.

        Args:
            client: Concrete AIBaseCompletions implementation instance.

        Returns:
            list of supported completion model identifiers.
            Raises TypeError when the object is not an AIBaseCompletions instance.
        """
        if not isinstance(client, AIBaseCompletions):
            raise TypeError(f"Expected AIBaseCompletions, got {type(client).__name__}")
        # Normal return with provider model name collection.
        return client.list_model_names

    @staticmethod
    def _resolve_classifier_engine(
        *, env_settings: EnvSettings, str_engine_override: str | None
    ) -> str:
        """
        Resolves the classification engine: argument, CLASSIFIER_ENGINE, then
        the Decisions API when an OpenAI key is present.

        Args:
            env_settings: Loaded settings.
            str_engine_override: Explicit engine from the caller, if any.

        Returns:
            Normalized engine token.

        Raises:
            ValueError: Nothing names an engine and no OpenAI key is set.
        """
        if str_engine_override is not None and str_engine_override.strip():
            # Early return with the caller's choice.
            return str_engine_override.strip().lower()
        str_configured: str = (
            str(env_settings.get_setting(CLASSIFIER_ENGINE_KEY, "") or "")
            .strip()
            .lower()
        )
        if str_configured:
            # Early return with the configured engine.
            return str_configured
        if str(env_settings.get_setting(OPENAI_API_KEY_SETTING, "") or "").strip():
            _LOGGER.info(
                "%s is unset; using %s because %s is set.",
                CLASSIFIER_ENGINE_KEY,
                DEFAULT_CLASSIFIER_ENGINE_WITH_OPENAI_KEY,
                OPENAI_API_KEY_SETTING,
            )
            # Early return with the key-backed default.
            return DEFAULT_CLASSIFIER_ENGINE_WITH_OPENAI_KEY
        raise ValueError(
            f"{CLASSIFIER_ENGINE_KEY} must be configured explicitly, or set "
            f"{OPENAI_API_KEY_SETTING} to use the default "
            f"{DEFAULT_CLASSIFIER_ENGINE_WITH_OPENAI_KEY} classifier."
        )

    @staticmethod
    def get_ai_classifier_client(
        classifier_engine: str | None = None,
        model_name: str | None = None,
        base_url: str | None = None,
    ) -> AIBaseClassifier:
        """
        Builds a classification client.

        Args:
            classifier_engine: Engine token override; defaults to CLASSIFIER_ENGINE,
                then to openai-decisions when OPENAI_API_KEY is set. Tokens named
                after a completions engine (openai, claude, google-gemini, ...)
                classify through that engine's structured output.
            model_name: Model override; defaults to CLASSIFIER_MODEL_NAME, then
                COMPLETIONS_MODEL_NAME, then the engine's own default.
            base_url: Optional API base-URL override for engines that accept one.

        Returns:
            Concrete AIBaseClassifier implementation for the engine.

        Raises:
            ValueError: No engine was configured and no OpenAI key is set, or
                the registry has no classification spec for the token.
        """
        env_settings: EnvSettings = EnvSettings()
        str_engine: str = AIFactory._resolve_classifier_engine(
            env_settings=env_settings, str_engine_override=classifier_engine
        )
        str_model_name: str = (
            model_name
            if model_name is not None
            else str(env_settings.get_setting(CLASSIFIER_MODEL_NAME_KEY, "") or "")
        )
        try:
            ai_provider_spec: AiProviderSpec = get_ai_provider_spec(
                AI_PROVIDER_CAPABILITY_CLASSIFICATION, str_engine
            )
            class_classifier_client: type[AIBaseClassifier] = load_ai_provider_class(
                ai_provider_spec,
                AIBaseClassifier,
            )
            dict_client_kwargs: dict[str, Any] = {
                "model": str_model_name.strip() or None,
                "engine": str_engine,
            }
            if base_url is not None:
                dict_client_kwargs["base_url"] = base_url
            classifier_client: AIBaseClassifier = class_classifier_client(
                **dict_client_kwargs
            )
            # Normal return with the configured classification client.
            return classifier_client
        except AiProviderConfigurationError as exception:
            raise AIFactory._translate_config_exception(
                exception,
                AI_PROVIDER_CAPABILITY_CLASSIFICATION,
                str_engine,
            ) from exception
        except AiProviderDependencyUnavailableError as exception:
            _LOGGER.warning(str(exception))
            raise
        except AiProviderRuntimeError:
            raise

    @staticmethod
    def get_ai_images_client(
        image_model: str | None = None,
    ) -> AIBaseImages:
        """
        Instantiates the configured image-generation client.

        Args:
            image_model: Optional model override; falls back to environment config.

        Returns:
            Concrete AIBaseImages implementation for the requested image engine.
            Raises ValueError for unsupported engines and RuntimeError-derived
            provider exceptions for dependency/runtime loading failures.
        """
        env_settings: EnvSettings = EnvSettings()
        str_image_engine: str = AIFactory._resolve_required_engine(
            env_settings=env_settings,
            str_engine_key=IMAGE_ENGINE_KEY,
        )

        str_image_model_name: str | None
        if image_model is None:
            raw_image_model_name: str | None = env_settings.get_setting(
                IMAGE_MODEL_NAME_KEY, None
            )
            if raw_image_model_name is None or raw_image_model_name.strip() == "":
                str_image_model_name = None
            else:
                str_image_model_name = raw_image_model_name
        else:
            str_image_model_name = image_model
        AIFactory._require_allowed(
            AIFactory._model_policy(env_settings),
            str_image_engine,
            str_image_model_name or "",
            "image model",
        )
        try:
            # Resolve provider metadata first so engine-to-module mapping is centralized
            # in the registry and does not require hardcoded imports in the factory.
            ai_provider_spec: AiProviderSpec = get_ai_provider_spec(
                AI_PROVIDER_CAPABILITY_IMAGES, str_image_engine
            )
            # Lazy-load and validate the provider class at runtime. This is the core
            # refactor behavior that keeps optional SDK imports out of package import-time
            # while still enforcing the AIBaseImages interface contract.
            class_ai_images_client: type[AIBaseImages] = load_ai_provider_class(
                ai_provider_spec,
                AIBaseImages,
            )
            # Instantiate the resolved class only after lazy-load validation succeeds.
            images_client: AIBaseImages = class_ai_images_client(
                model=str_image_model_name
            )
            # Normal return with configured images provider client.
            return images_client
        except AiProviderConfigurationError as exception:
            raise AIFactory._translate_config_exception(
                exception,
                AI_PROVIDER_CAPABILITY_IMAGES,
                str_image_engine,
            ) from exception
        except AiProviderDependencyUnavailableError as exception:
            _LOGGER.warning(str(exception))
            raise
        except AiProviderRuntimeError:
            raise

    @staticmethod
    def get_ai_video_client(
        model_name: str | None = None,
        video_engine: str | None = None,
    ) -> AIBaseVideos:
        """
        Instantiates the configured video-generation client.

        Args:
            model_name: Optional model override; falls back to environment config.
            video_engine: Optional engine override; falls back to environment config.

        Returns:
            Concrete AIBaseVideos implementation for the requested video engine.
        """
        env_settings: EnvSettings = EnvSettings()
        str_video_engine: str = AIFactory._resolve_required_engine(
            env_settings=env_settings,
            str_engine_key=VIDEO_ENGINE_KEY,
            str_engine_override=video_engine,
        )

        str_model_name: str | None
        if model_name is None:
            raw_model_name: str | None = env_settings.get_setting(
                VIDEO_MODEL_NAME_KEY,
                None,
            )
            if raw_model_name is None or raw_model_name.strip() == "":
                str_model_name = None
            else:
                str_model_name = raw_model_name
        else:
            str_model_name = model_name
        AIFactory._require_allowed(
            AIFactory._model_policy(env_settings),
            str_video_engine,
            str_model_name or "",
            "video model",
        )
        try:
            ai_provider_spec: AiProviderSpec = get_ai_provider_spec(
                AI_PROVIDER_CAPABILITY_VIDEOS,
                str_video_engine,
            )
            class_ai_video_client: type[AIBaseVideos] = load_ai_provider_class(
                ai_provider_spec,
                AIBaseVideos,
            )
            return class_ai_video_client(model=str_model_name)
        except AiProviderConfigurationError as exception:
            raise AIFactory._translate_config_exception(
                exception,
                AI_PROVIDER_CAPABILITY_VIDEOS,
                str_video_engine,
            ) from exception
        except AiProviderDependencyUnavailableError as exception:
            _LOGGER.warning(str(exception))
            raise
        except AiProviderRuntimeError:
            raise

    @staticmethod
    def get_ai_voice_client(
        voice_engine: str | None = None,
        base_url: str | None = None,
        retry_policy: str | None = None,
    ) -> AIVoiceBase:
        """
        Instantiates the configured voice (text-to-speech) client.

        Delegates to AIVoiceFactory, which owns voice provider construction, so
        every capability is reachable from the same factory.

        Args:
            voice_engine: Optional engine override; falls back to AI_VOICE_ENGINE config.
            base_url: Optional API base-URL override for engines whose SDK accepts one.
            retry_policy: Optional retry policy for engines whose SDK accepts one.

        Returns:
            Concrete AIVoiceBase implementation for the requested voice engine.
        """
        env_settings: EnvSettings = EnvSettings()
        str_voice_engine: str = AIFactory._resolve_required_engine(
            env_settings=env_settings,
            str_engine_key=AI_VOICE_ENGINE_ENV_KEY,
            str_engine_override=voice_engine,
        )

        # AIVoiceFactory owns the per-engine support check, since it is the
        # layer that accepts and forwards these arguments.
        dict_provider_kwargs: dict[str, Any] = {}
        if base_url is not None:
            dict_provider_kwargs["base_url"] = base_url
        if retry_policy is not None:
            dict_provider_kwargs["retry_policy"] = retry_policy
        # Normal return with the voice client resolved by the voice factory.
        return AIVoiceFactory.create(str_voice_engine, **dict_provider_kwargs)
