# ai_model_probes.py
"""
Per-provider "does this model exist" probes for AiModelChecker.

Each probe makes at most one metadata call against the provider, never a
completion, and maps the answer to a check status:

    available     the provider returned the model's metadata
    missing       the provider said there is no such model (404, or an id
                  outside the set a fixed-catalog API serves)
    unverifiable  the probe could not ask: no credentials, no extra
                  installed, an auth rejection, or a server that does not
                  expose a model catalog

Anything else (a 5xx, a dropped connection) propagates and the checker
records it as an error, which also keeps the model.

The probes build their own thin SDK clients from the same settings the
engines use, so a check never constructs an engine (the Gemini engine, for
one, replaces an unknown model id with its default before the library's
own metadata call, which is the kind of silent swap this module exists to
catch).
"""

from __future__ import annotations

from typing import Any

from .ai_model_checker import AiModelChecker, AiModelCheckStatus
from .env_settings import EnvSettings

# Bedrock cross-region inference profile ids carry a geo prefix and live in
# a different catalog from foundation models.
TUPLE_BEDROCK_PROFILE_PREFIXES: tuple[str, ...] = ("us.", "eu.", "apac.", "global.")
FROZENSET_BEDROCK_MISSING_CODES: frozenset[str] = frozenset(
    {"ResourceNotFoundException", "ValidationException"}
)
FROZENSET_BEDROCK_UNVERIFIABLE_CODES: frozenset[str] = frozenset(
    {
        "AccessDeniedException",
        "UnrecognizedClientException",
        "InvalidSignatureException",
        "ExpiredTokenException",
    }
)
FROZENSET_BEDROCK_ENGINE_TOKENS: frozenset[str] = frozenset(
    {
        "bedrock",
        "nova",
        "anthropic",
        "llama",
        "mistral",
        "cohere",
        "ai21",
        "rerank",
        "canvas",
    }
)

TypeProbeAnswer = tuple[AiModelCheckStatus, str]


def _unverifiable(str_detail: str) -> TypeProbeAnswer:
    # Normal return with an unverifiable answer.
    return AiModelCheckStatus.UNVERIFIABLE, str_detail


def _missing(str_detail: str) -> TypeProbeAnswer:
    # Normal return with a missing answer.
    return AiModelCheckStatus.MISSING, str_detail


def _available(str_detail: str = "") -> TypeProbeAnswer:
    # Normal return with an available answer.
    return AiModelCheckStatus.AVAILABLE, str_detail


def _status_code(exception: Exception) -> int | None:
    raw: Any = getattr(exception, "status_code", None)
    # Normal return with an integer status, or None.
    return raw if isinstance(raw, int) else None


# ── OpenAI ──────────────────────────────────────────────────────────────────


def _openai_client(
    base_url: str | None, *, str_key_setting: str = "OPENAI_API_KEY"
) -> Any:
    """A bare OpenAI client, or None when the key is not set."""
    from openai import OpenAI

    env: EnvSettings = EnvSettings()
    str_key: str = str(env.get_setting(str_key_setting, "") or "").strip()
    if not str_key:
        # Early return: nothing to authenticate with.
        return None
    dict_kwargs: dict[str, Any] = {"api_key": str_key, "max_retries": 0}
    str_base: str | None = base_url or (
        str(env.get_setting("OPENAI_BASE_URL_OVERRIDE", "") or "").strip() or None
    )
    if str_base:
        dict_kwargs["base_url"] = str_base
    # Normal return with the client.
    return OpenAI(**dict_kwargs)


def _openai_retrieve(client: Any, str_model: str) -> TypeProbeAnswer:
    """models.retrieve, mapped: 404 missing; 401/403 unverifiable."""
    from openai import APIStatusError

    try:
        client.models.retrieve(str_model)
    except APIStatusError as exception:
        int_status: int | None = _status_code(exception)
        if int_status == 404:
            # Early return: the provider has no such model.
            return _missing(f"OpenAI returned 404 for model {str_model!r}")
        if int_status in (401, 403):
            # Early return: cannot ask with these credentials.
            return _unverifiable(f"OpenAI rejected the credentials ({int_status})")
        raise
    # Normal return: the model exists.
    return _available("OpenAI models.retrieve")


def probe_openai(str_model: str, base_url: str | None) -> TypeProbeAnswer:
    """OpenAI (chat and Responses share one catalog)."""
    if not str_model:
        # Early return: the engine default is not a configured id.
        return _unverifiable("engine default model")
    try:
        client: Any = _openai_client(base_url)
    except ImportError:
        # Early return: the extra is not installed.
        return _unverifiable("openai extra not installed")
    if client is None:
        # Early return: no key.
        return _unverifiable("OPENAI_API_KEY not set")
    # Normal return with the catalog answer.
    return _openai_retrieve(client, str_model)


def probe_openai_decisions(str_model: str, base_url: str | None) -> TypeProbeAnswer:
    """The Decisions API serves a fixed set; the id must also exist as a model."""
    from ..classification.ai_openai_decisions_classifier import (
        DECISIONS_DEFAULT_MODEL,
        FROZENSET_DECISIONS_MODELS,
    )

    # A blank id means the classifier's own default, not the first in the set.
    str_id: str = str_model or DECISIONS_DEFAULT_MODEL
    if str_id not in FROZENSET_DECISIONS_MODELS:
        # Early return: not a Decisions model, whatever the catalog says.
        return _missing(
            f"{str_id!r} is not served by the Decisions API "
            f"(supported: {', '.join(sorted(FROZENSET_DECISIONS_MODELS))})"
        )
    # Normal return with the catalog answer for the id.
    return probe_openai(str_id, base_url)


def probe_openai_compatible(str_model: str, base_url: str | None) -> TypeProbeAnswer:
    """
    An OpenAI-compatible server may or may not expose /models, so the probe
    lists the catalog and only calls a model missing when the list is
    non-empty and lacks it.
    """
    if not str_model:
        # Early return: the engine default is not a configured id.
        return _unverifiable("engine default model")
    try:
        from openai import OpenAI
    except ImportError:
        # Early return: the extra is not installed.
        return _unverifiable("openai extra not installed")
    env: EnvSettings = EnvSettings()
    str_base: str | None = base_url or (
        str(env.get_setting("OPENAI_COMPATIBLE_BASE_URL", "") or "").strip() or None
    )
    if not str_base:
        # Early return: no server to ask.
        return _unverifiable("OPENAI_COMPATIBLE_BASE_URL not set")
    str_key: str = str(env.get_setting("OPENAI_COMPATIBLE_API_KEY", "") or "").strip()
    client: Any = OpenAI(
        api_key=str_key or "not-needed", base_url=str_base, max_retries=0
    )
    try:
        list_ids: list[str] = [
            str(getattr(item, "id", "")) for item in client.models.list()
        ]
    except Exception as exception:
        # Early return: the server does not answer /models the standard way.
        return _unverifiable(f"server did not list models ({type(exception).__name__})")
    if not list_ids:
        # Early return: an empty catalog says nothing.
        return _unverifiable("server listed no models")
    if str_model in list_ids:
        # Early return: listed.
        return _available("listed by /models")
    # Normal return: the server lists models and this is not one of them.
    return _missing(f"{str_model!r} is not in the server's /models list")


# ── Anthropic ───────────────────────────────────────────────────────────────


def probe_anthropic(str_model: str, base_url: str | None) -> TypeProbeAnswer:
    """Anthropic models.retrieve."""
    if not str_model:
        # Early return: the engine default is not a configured id.
        return _unverifiable("engine default model")
    try:
        from anthropic import Anthropic, APIStatusError
    except ImportError:
        # Early return: the extra is not installed.
        return _unverifiable("anthropic extra not installed")
    env: EnvSettings = EnvSettings()
    str_key: str = str(env.get_setting("ANTHROPIC_API_KEY", "") or "").strip()
    if not str_key:
        # Early return: no key.
        return _unverifiable("ANTHROPIC_API_KEY not set")
    dict_kwargs: dict[str, Any] = {"api_key": str_key, "max_retries": 0}
    str_base: str | None = base_url or (
        str(env.get_setting("ANTHROPIC_BASE_URL_OVERRIDE", "") or "").strip() or None
    )
    if str_base:
        dict_kwargs["base_url"] = str_base
    client: Any = Anthropic(**dict_kwargs)
    try:
        client.models.retrieve(str_model)
    except APIStatusError as exception:
        int_status: int | None = _status_code(exception)
        if int_status == 404:
            # Early return: the provider has no such model.
            return _missing(f"Anthropic returned 404 for model {str_model!r}")
        if int_status in (401, 403):
            # Early return: cannot ask with these credentials.
            return _unverifiable(f"Anthropic rejected the credentials ({int_status})")
        raise
    # Normal return: the model exists.
    return _available("Anthropic models.retrieve")


# ── Google Gemini ───────────────────────────────────────────────────────────


def probe_google_gemini(str_model: str, base_url: str | None) -> TypeProbeAnswer:
    """Gemini models.get on the raw id (the engine would substitute its default)."""
    if not str_model:
        # Early return: the engine default is not a configured id.
        return _unverifiable("engine default model")
    try:
        from google import genai
        from google.genai import errors as gerr
    except ImportError:
        # Early return: the extra is not installed.
        return _unverifiable("google_gemini extra not installed")
    env: EnvSettings = EnvSettings()
    str_auth: str = (
        str(env.get_setting("GOOGLE_AUTH_METHOD", "api_key") or "api_key")
        .strip()
        .lower()
    )
    if str_auth != "api_key":
        # Early return: Vertex credentials are not probed in this version.
        return _unverifiable(f"GOOGLE_AUTH_METHOD={str_auth} is not probed")
    str_key: str = str(env.get_setting("GOOGLE_GEMINI_API_KEY", "") or "").strip()
    if not str_key:
        # Early return: no key.
        return _unverifiable("GOOGLE_GEMINI_API_KEY not set")
    dict_kwargs: dict[str, Any] = {"api_key": str_key}
    str_base: str | None = base_url or (
        str(env.get_setting("GOOGLE_GEMINI_BASE_URL_OVERRIDE", "") or "").strip()
        or None
    )
    if str_base:
        dict_kwargs["http_options"] = genai.types.HttpOptions(base_url=str_base)
    client: Any = genai.Client(**dict_kwargs)
    try:
        client.models.get(model=str_model)
    except gerr.APIError as exception:
        int_code: Any = getattr(exception, "code", None)
        if int_code == 404:
            # Early return: the provider has no such model.
            return _missing(f"Gemini returned 404 for model {str_model!r}")
        if int_code in (401, 403):
            # Early return: cannot ask with these credentials.
            return _unverifiable(f"Gemini rejected the credentials ({int_code})")
        raise
    # Normal return: the model exists.
    return _available("Gemini models.get")


# ── Bedrock ─────────────────────────────────────────────────────────────────


def probe_bedrock(str_model: str, base_url: str | None) -> TypeProbeAnswer:
    """
    Bedrock control plane: get_inference_profile for geo-prefixed ids,
    get_foundation_model otherwise, in the configured region.
    """
    if not str_model:
        # Early return: the engine default is not a configured id.
        return _unverifiable("engine default model")
    try:
        import boto3
        from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError
    except ImportError:
        # Early return: the extra is not installed.
        return _unverifiable("bedrock extra not installed")
    env: EnvSettings = EnvSettings()
    str_region: str = str(
        env.get_setting("AWS_REGION", "us-east-1") or "us-east-1"
    ).strip()
    try:
        client: Any = boto3.client("bedrock", region_name=str_region)
        if str_model.startswith(TUPLE_BEDROCK_PROFILE_PREFIXES):
            client.get_inference_profile(inferenceProfileIdentifier=str_model)
            str_how: str = "Bedrock get_inference_profile"
        else:
            client.get_foundation_model(modelIdentifier=str_model)
            str_how = "Bedrock get_foundation_model"
    except NoCredentialsError:
        # Early return: no AWS credentials in the chain.
        return _unverifiable("no AWS credentials")
    except ClientError as exception:
        str_code: str = str(
            (getattr(exception, "response", None) or {})
            .get("Error", {})
            .get("Code", "")
        )
        if str_code in FROZENSET_BEDROCK_MISSING_CODES:
            # Early return: the region has no such model or profile.
            return _missing(f"Bedrock {str_code} for {str_model!r} in {str_region}")
        if str_code in FROZENSET_BEDROCK_UNVERIFIABLE_CODES:
            # Early return: cannot ask with these credentials.
            return _unverifiable(f"Bedrock {str_code}")
        raise
    except BotoCoreError as exception:
        # Early return: a client-side boto failure (bad region, endpoint).
        return _unverifiable(f"{type(exception).__name__}: {exception}")
    # Normal return: the model exists in the region.
    return _available(str_how)


def register_default_probes() -> None:
    """Registers every provider probe on AiModelChecker (idempotent)."""
    AiModelChecker.register_probe("openai", probe_openai)
    AiModelChecker.register_probe("openai-responses", probe_openai)
    AiModelChecker.register_probe("openai-decisions", probe_openai_decisions)
    AiModelChecker.register_probe("openai-compatible", probe_openai_compatible)
    AiModelChecker.register_probe("claude", probe_anthropic)
    AiModelChecker.register_probe("google-gemini", probe_google_gemini)
    # Loop over the Bedrock-routed tokens; they share one control plane.
    for str_token in FROZENSET_BEDROCK_ENGINE_TOKENS:
        AiModelChecker.register_probe(str_token, probe_bedrock)


register_default_probes()
