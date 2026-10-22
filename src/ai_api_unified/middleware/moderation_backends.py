# moderation_backends.py
"""
Provider moderation services the guardrail checks can run on, beside
OpenAI Moderation (in guardrails.py) and the classifier.

| Backend | Service | Checks it serves |
| --- | --- | --- |
| `mistral_moderation` | Mistral `POST /v1/moderations` | harmful_content |
| `azure_content_safety` | Azure AI Content Safety `text:analyze`; Prompt Shields `text:shieldPrompt` | harmful_content; prompt_injection |
| `bedrock_guardrails` | Amazon Bedrock `ApplyGuardrail` against a guardrail in the account | harmful_content; prompt_injection (the PROMPT_ATTACK filter) |
| `google_moderation` | Google Cloud Natural Language `documents:moderateText` | harmful_content |

Every backend answers the same way: the highest score among the categories
the check honors, on a 0 to 1 scale, and the categories at or above the
check's threshold. Scales differ by provider and are mapped here:

- Mistral and Google report a probability per category; it is used as is.
- Azure reports a severity of 0, 2, 4, or 6; the score is severity / 6, so
  the default threshold 0.8 fires on high (6), and 0.5 on medium (4) and up.
- Bedrock reports a confidence of NONE, LOW, MEDIUM, or HIGH, mapped to 0,
  1/3, 2/3, and 1, for content filters; a detected denied topic or word
  scores 1.
- Prompt Shields reports only whether an attack was detected: 1 or 0.

Category names are normalized to lower snake case (`SelfHarm` becomes
`self_harm`, `Death, Harm & Tragedy` becomes `death_harm_and_tragedy`). A
check's `categories` setting, when given, lists the names to honor; when
empty, each backend honors its harm categories and ignores the topic labels
some providers also report (Mistral's health, financial, law, and pii;
Google's Health, Finance, Politics, Legal, and others), which describe a
subject rather than harm.

Secrets come from the environment: MISTRAL_API_KEY, AZURE_CONTENT_SAFETY_KEY
(with the endpoint in the settings or AZURE_CONTENT_SAFETY_ENDPOINT),
GOOGLE_CLOUD_LANGUAGE_API_KEY, and the usual AWS credential chain. Calls to
these services are not metered on the request cost report: they bill per
text record or unit, not per token, and the pricing registry has no such
rates.
"""

from __future__ import annotations

import re
import threading
from collections.abc import Iterable
from typing import Any

from .middleware_config import (
    GUARDRAIL_BACKEND_AZURE_CONTENT_SAFETY,
    GUARDRAIL_BACKEND_BEDROCK_GUARDRAILS,
    GUARDRAIL_BACKEND_GOOGLE_MODERATION,
    GUARDRAIL_BACKEND_MISTRAL_MODERATION,
    GUARDRAIL_CHECK_PROMPT_INJECTION,
    GuardrailModelCheckSettingsModel,
    GuardrailsSettingsModel,
)

MISTRAL_API_KEY_SETTING: str = "MISTRAL_API_KEY"
AZURE_CONTENT_SAFETY_KEY_SETTING: str = "AZURE_CONTENT_SAFETY_KEY"
AZURE_CONTENT_SAFETY_ENDPOINT_SETTING: str = "AZURE_CONTENT_SAFETY_ENDPOINT"
GOOGLE_LANGUAGE_API_KEY_SETTING: str = "GOOGLE_CLOUD_LANGUAGE_API_KEY"
AWS_REGION_SETTING: str = "AWS_REGION"

GOOGLE_MODERATE_URL: str = "https://language.googleapis.com/v2/documents:moderateText"
# Azure accepts at most 10,000 characters per text record.
INT_AZURE_CHUNK_CHARS: int = 10_000
INT_DEFAULT_CHUNK_CHARS: int = 30_000

# Harm categories each backend honors when a check names none.
FROZENSET_MISTRAL_HARM: frozenset[str] = frozenset(
    {
        "sexual",
        "hate_and_discrimination",
        "violence_and_threats",
        "dangerous_and_criminal_content",
        "selfharm",
    }
)
FROZENSET_GOOGLE_HARM: frozenset[str] = frozenset(
    {
        "toxic",
        "insult",
        "profanity",
        "derogatory",
        "sexual",
        "death_harm_and_tragedy",
        "violent",
        "firearms_and_weapons",
        "public_safety",
        "illicit_drugs",
    }
)
CATEGORY_PROMPT_ATTACK: str = "prompt_attack"
DICT_BEDROCK_CONFIDENCE: dict[str, float] = {
    "NONE": 0.0,
    "LOW": 1.0 / 3.0,
    "MEDIUM": 2.0 / 3.0,
    "HIGH": 1.0,
}

_LOCK: threading.Lock = threading.Lock()
_HTTP_CLIENT: Any = None
_BEDROCK_CLIENT: Any = None


def set_moderation_http_client(client: Any) -> None:
    """Supplies the HTTP client the REST backends use (tests, or a proxy)."""
    global _HTTP_CLIENT
    with _LOCK:
        _HTTP_CLIENT = client


def set_bedrock_guardrails_client(client: Any) -> None:
    """Supplies the bedrock-runtime client ApplyGuardrail runs on (tests)."""
    global _BEDROCK_CLIENT
    with _LOCK:
        _BEDROCK_CLIENT = client


def reset_moderation_backends() -> None:
    """Forgets the built clients."""
    global _HTTP_CLIENT, _BEDROCK_CLIENT
    with _LOCK:
        _HTTP_CLIENT = None
        _BEDROCK_CLIENT = None


def _setting(str_key: str) -> str:
    from ..util.env_settings import EnvSettings

    try:
        # Normal return with the stripped value, or "".
        return str(EnvSettings().get_setting(str_key, "") or "").strip()
    except Exception:
        # Early return: unreadable settings mean unset.
        return ""


def _http(settings: GuardrailsSettingsModel) -> Any:
    global _HTTP_CLIENT
    if _HTTP_CLIENT is not None:
        # Early return with the built or supplied client.
        return _HTTP_CLIENT
    with _LOCK:
        if _HTTP_CLIENT is None:
            import httpx

            _HTTP_CLIENT = httpx.Client(timeout=settings.backends.timeout_seconds)
    # Normal return with the client.
    return _HTTP_CLIENT


def _bedrock(settings: GuardrailsSettingsModel) -> Any:
    global _BEDROCK_CLIENT
    if _BEDROCK_CLIENT is not None:
        # Early return with the built or supplied client.
        return _BEDROCK_CLIENT
    with _LOCK:
        if _BEDROCK_CLIENT is None:
            import boto3

            _BEDROCK_CLIENT = boto3.client(
                service_name="bedrock-runtime",
                region_name=settings.backends.bedrock_guardrails.region
                or _setting(AWS_REGION_SETTING)
                or "us-east-1",
            )
    # Normal return with the client.
    return _BEDROCK_CLIENT


# ── Which backends are usable ─────────────────────────────────────────────


def azure_endpoint(settings: GuardrailsSettingsModel) -> str:
    """The Content Safety endpoint, from settings or the environment."""
    # Normal return with the endpoint without a trailing slash, or "".
    return (
        settings.backends.azure_content_safety.endpoint
        or _setting(AZURE_CONTENT_SAFETY_ENDPOINT_SETTING)
    ).rstrip("/")


def available(str_backend: str, settings: GuardrailsSettingsModel) -> bool:
    """True when a backend has what it needs to run (a key, an endpoint, an id)."""
    if str_backend == GUARDRAIL_BACKEND_MISTRAL_MODERATION:
        # Early return: a Mistral key.
        return bool(_setting(MISTRAL_API_KEY_SETTING))
    if str_backend == GUARDRAIL_BACKEND_AZURE_CONTENT_SAFETY:
        # Early return: an endpoint and a key.
        return bool(
            azure_endpoint(settings) and _setting(AZURE_CONTENT_SAFETY_KEY_SETTING)
        )
    if str_backend == GUARDRAIL_BACKEND_BEDROCK_GUARDRAILS:
        # Early return: a guardrail to apply.
        return bool(settings.backends.bedrock_guardrails.guardrail_id)
    if str_backend == GUARDRAIL_BACKEND_GOOGLE_MODERATION:
        # Early return: a Cloud API key.
        return bool(_setting(GOOGLE_LANGUAGE_API_KEY_SETTING))
    # Normal return: not one of these backends.
    return False


# ── Scoring ───────────────────────────────────────────────────────────────


def normalize_category(str_name: str) -> str:
    """`SelfHarm` -> `self_harm`; `Death, Harm & Tragedy` -> `death_harm_and_tragedy`."""
    str_value: str = str(str_name).replace("&", " and ")
    str_value = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", str_value)
    str_value = re.sub(r"[^A-Za-z0-9]+", "_", str_value).strip("_").lower()
    # Normal return with the snake-case name.
    return str_value


def _score(
    check: GuardrailModelCheckSettingsModel,
    pairs: Iterable[tuple[str, float]],
    frozenset_default: frozenset[str] | None,
) -> tuple[float, tuple[str, ...]]:
    """
    (top honored score, honored categories at or above the threshold).

    Args:
        check: The check, for its threshold and categories.
        pairs: (normalized category, score) from the provider.
        frozenset_default: Categories honored when the check names none;
            None honors every category.
    """
    set_honored: set[str] = {normalize_category(c) for c in check.categories}
    float_top: float = 0.0
    set_fired: set[str] = set()
    # Loop over the provider's scores, keeping the honored ones.
    for str_name, float_score in pairs:
        if set_honored:
            if str_name not in set_honored:
                continue
        elif frozenset_default is not None and str_name not in frozenset_default:
            continue
        float_top = max(float_top, float_score)
        if float_score >= check.threshold:
            set_fired.add(str_name)
    # Normal return with the top score and the fired categories.
    return float_top, tuple(sorted(set_fired))


def _chunks(str_text: str, int_size: int) -> list[str]:
    # Normal return with the text in pieces of at most int_size characters.
    return [str_text[i : i + int_size] for i in range(0, len(str_text), int_size)] or [
        str_text
    ]


def _raise_for_status(response: Any, str_service: str) -> Any:
    """The response's JSON, or an error naming the service and status."""
    int_status: int = int(getattr(response, "status_code", 0) or 0)
    if int_status >= 400:
        from ..ai_provider_exceptions import AiProviderRequestError

        raise AiProviderRequestError(
            f"{str_service} answered {int_status}.",
            status_code=int_status,
            provider_engine=str_service,
        )
    # Normal return with the decoded body.
    return response.json()


# ── Backends ──────────────────────────────────────────────────────────────


def _run_mistral(
    check: GuardrailModelCheckSettingsModel,
    settings: GuardrailsSettingsModel,
    str_text: str,
) -> tuple[float, tuple[str, ...]]:
    config: Any = settings.backends.mistral_moderation
    response: Any = _http(settings).post(
        f"{config.base_url.rstrip('/')}/moderations",
        headers={"Authorization": f"Bearer {_setting(MISTRAL_API_KEY_SETTING)}"},
        json={
            "model": config.model,
            "input": _chunks(str_text, INT_DEFAULT_CHUNK_CHARS),
        },
    )
    body: Any = _raise_for_status(response, "mistral_moderation")
    list_pairs: list[tuple[str, float]] = [
        (normalize_category(str_name), float(raw or 0.0))
        for result in body.get("results") or []
        for str_name, raw in (result.get("category_scores") or {}).items()
    ]
    # Normal return with the scored categories.
    return _score(check, list_pairs, FROZENSET_MISTRAL_HARM)


def _run_azure(
    str_check: str,
    check: GuardrailModelCheckSettingsModel,
    settings: GuardrailsSettingsModel,
    str_text: str,
    bool_document: bool = False,
) -> tuple[float, tuple[str, ...]]:
    config: Any = settings.backends.azure_content_safety
    str_endpoint: str = azure_endpoint(settings)
    dict_headers: dict[str, str] = {
        "Ocp-Apim-Subscription-Key": _setting(AZURE_CONTENT_SAFETY_KEY_SETTING),
        "Content-Type": "application/json",
    }
    client: Any = _http(settings)
    if str_check == GUARDRAIL_CHECK_PROMPT_INJECTION:
        bool_attack: bool = False
        # Loop over the chunks; one detected attack is enough.
        for str_chunk in _chunks(str_text, INT_AZURE_CHUNK_CHARS):
            # A tool result is a document (Prompt Shields' indirect-attack
            # analysis); the caller's own text is the user prompt.
            dict_body: dict[str, Any] = (
                {"userPrompt": "", "documents": [str_chunk]}
                if bool_document
                else {"userPrompt": str_chunk, "documents": []}
            )
            body: Any = _raise_for_status(
                client.post(
                    f"{str_endpoint}/contentsafety/text:shieldPrompt",
                    params={"api-version": config.api_version},
                    headers=dict_headers,
                    json=dict_body,
                ),
                "azure_content_safety",
            )
            list_analyses: list[Any] = (
                list(body.get("documentsAnalysis") or [])
                if bool_document
                else [body.get("userPromptAnalysis") or {}]
            )
            if any((item or {}).get("attackDetected") for item in list_analyses):
                bool_attack = True
                break
        # Early return: Prompt Shields says yes or no.
        return _score(
            check,
            [(CATEGORY_PROMPT_ATTACK, 1.0 if bool_attack else 0.0)],
            None,
        )
    list_pairs: list[tuple[str, float]] = []
    # Loop over the chunks, collecting each category's severity.
    for str_chunk in _chunks(str_text, INT_AZURE_CHUNK_CHARS):
        body = _raise_for_status(
            client.post(
                f"{str_endpoint}/contentsafety/text:analyze",
                params={"api-version": config.api_version},
                headers=dict_headers,
                json={"text": str_chunk, "outputType": "FourSeverityLevels"},
            ),
            "azure_content_safety",
        )
        list_pairs.extend(
            (
                normalize_category(item.get("category", "")),
                float(item.get("severity") or 0) / 6.0,
            )
            for item in body.get("categoriesAnalysis") or []
        )
    # Normal return with the scored categories.
    return _score(check, list_pairs, None)


def _bedrock_pairs(str_check: str, response: Any) -> list[tuple[str, float]]:
    """(category, score) from an ApplyGuardrail response, for this check."""
    bool_injection: bool = str_check == GUARDRAIL_CHECK_PROMPT_INJECTION
    list_pairs: list[tuple[str, float]] = []
    # Loop over the assessments for detected filters, topics, and words.
    for assessment in response.get("assessments") or []:
        for item in (assessment.get("contentPolicy") or {}).get("filters") or []:
            str_name: str = normalize_category(item.get("type", ""))
            if (str_name == CATEGORY_PROMPT_ATTACK) != bool_injection:
                continue
            if not item.get("detected", item.get("action") == "BLOCKED"):
                continue
            list_pairs.append(
                (
                    str_name,
                    DICT_BEDROCK_CONFIDENCE.get(str(item.get("confidence")), 0.0),
                )
            )
        if bool_injection:
            continue
        for topic in (assessment.get("topicPolicy") or {}).get("topics") or []:
            if topic.get("detected", topic.get("action") == "BLOCKED"):
                list_pairs.append(
                    (f"topic_{normalize_category(topic.get('name', ''))}", 1.0)
                )
        dict_words: Any = assessment.get("wordPolicy") or {}
        for word in dict_words.get("customWords") or []:
            if word.get("detected", word.get("action") == "BLOCKED"):
                list_pairs.append(("custom_word", 1.0))
        for word in dict_words.get("managedWordLists") or []:
            if word.get("detected", word.get("action") == "BLOCKED"):
                list_pairs.append(
                    (normalize_category(word.get("type", "profanity")), 1.0)
                )
    # Normal return with the pairs.
    return list_pairs


def _run_bedrock(
    str_check: str,
    check: GuardrailModelCheckSettingsModel,
    settings: GuardrailsSettingsModel,
    str_direction: str,
    str_text: str,
) -> tuple[float, tuple[str, ...]]:
    config: Any = settings.backends.bedrock_guardrails
    response: Any = _bedrock(settings).apply_guardrail(
        guardrailIdentifier=config.guardrail_id,
        guardrailVersion=config.guardrail_version,
        source="INPUT" if str_direction == "input" else "OUTPUT",
        content=[{"text": {"text": str_text}}],
        outputScope="FULL",
    )
    # Normal return with the scored categories.
    return _score(check, _bedrock_pairs(str_check, response), None)


def _run_google(
    check: GuardrailModelCheckSettingsModel,
    settings: GuardrailsSettingsModel,
    str_text: str,
) -> tuple[float, tuple[str, ...]]:
    list_pairs: list[tuple[str, float]] = []
    client: Any = _http(settings)
    # Loop over the chunks, collecting each category's confidence.
    for str_chunk in _chunks(str_text, INT_DEFAULT_CHUNK_CHARS):
        body: Any = _raise_for_status(
            client.post(
                GOOGLE_MODERATE_URL,
                params={"key": _setting(GOOGLE_LANGUAGE_API_KEY_SETTING)},
                json={
                    "document": {"type": "PLAIN_TEXT", "content": str_chunk},
                    "modelVersion": settings.backends.google_moderation.model_version,
                },
            ),
            "google_moderation",
        )
        list_pairs.extend(
            (
                normalize_category(item.get("name", "")),
                float(item.get("confidence") or 0.0),
            )
            for item in body.get("moderationCategories") or []
        )
    # Normal return with the scored categories.
    return _score(check, list_pairs, FROZENSET_GOOGLE_HARM)


def run_backend(
    str_backend: str,
    str_check: str,
    check: GuardrailModelCheckSettingsModel,
    settings: GuardrailsSettingsModel,
    str_direction: str,
    str_text: str,
    bool_document: bool = False,
) -> tuple[float, tuple[str, ...]]:
    """
    Runs one provider backend for one check.

    bool_document marks the text as a tool result rather than the caller's
    prompt; Azure Prompt Shields reads it as a document. Bedrock's
    PROMPT_ATTACK filter has no such distinction and reads it as input.

    Returns:
        (top honored score, fired categories).

    Raises:
        ValueError: An unknown backend.
        Whatever the service raises; the caller applies on_error.
    """
    if str_backend == GUARDRAIL_BACKEND_MISTRAL_MODERATION:
        # Early return with Mistral's answer.
        return _run_mistral(check, settings, str_text)
    if str_backend == GUARDRAIL_BACKEND_AZURE_CONTENT_SAFETY:
        # Early return with Azure's answer.
        return _run_azure(str_check, check, settings, str_text, bool_document)
    if str_backend == GUARDRAIL_BACKEND_BEDROCK_GUARDRAILS:
        # Early return with Bedrock's answer.
        return _run_bedrock(str_check, check, settings, str_direction, str_text)
    if str_backend == GUARDRAIL_BACKEND_GOOGLE_MODERATION:
        # Early return with Google's answer.
        return _run_google(check, settings, str_text)
    raise ValueError(f"Unknown moderation backend {str_backend!r}.")
