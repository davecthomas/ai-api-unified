# tests/test_moderation_backends.py
"""
The provider moderation backends for guardrails: Mistral, Azure AI Content
Safety (text analysis and Prompt Shields), Bedrock Guardrails, and Google
Cloud Natural Language. Every provider is faked; nothing leaves the process.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from ai_api_unified import AiGuardrailBlockedError, reset_guardrails
from ai_api_unified.middleware import guardrails as gr
from ai_api_unified.middleware import moderation_backends as mb
from ai_api_unified.middleware.middleware_config import (
    GuardrailModelCheckSettingsModel,
    GuardrailsSettingsModel,
)
from tests.test_request_cost import _PricedEngine

KEYS: tuple[str, ...] = (
    "OPENAI_API_KEY",
    mb.MISTRAL_API_KEY_SETTING,
    mb.AZURE_CONTENT_SAFETY_KEY_SETTING,
    mb.AZURE_CONTENT_SAFETY_ENDPOINT_SETTING,
    mb.GOOGLE_LANGUAGE_API_KEY_SETTING,
)


class _Response:
    def __init__(self, body: Any, status_code: int = 200) -> None:
        self.body = body
        self.status_code = status_code

    def json(self) -> Any:
        return self.body


class _Http:
    """Answers POSTs from a table keyed by a URL substring; records each request."""

    def __init__(self, dict_routes: dict[str, Any]) -> None:
        self.dict_routes = dict_routes
        self.requests: list[dict[str, Any]] = []

    def post(self, url: str, **kwargs: Any) -> _Response:
        self.requests.append({"url": url, **kwargs})
        for str_key, answer in self.dict_routes.items():
            if str_key in url:
                return answer if isinstance(answer, _Response) else _Response(answer)
        raise AssertionError(f"unexpected POST {url}")


class _Bedrock:
    def __init__(self, response: dict[str, Any]) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    def apply_guardrail(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        return self.response


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Credentials come only from what each test sets, never a developer .env."""
    import os

    for str_key in KEYS:
        monkeypatch.delenv(str_key, raising=False)
    monkeypatch.setattr(mb, "_setting", lambda k: os.environ.get(k, "").strip())
    monkeypatch.setattr(
        gr, "_openai_key_present", lambda: bool(os.environ.get("OPENAI_API_KEY"))
    )
    yield
    reset_guardrails()


@pytest.fixture
def guardrails_on(monkeypatch: pytest.MonkeyPatch) -> Any:
    def _enable(dict_settings: dict[str, Any]) -> None:
        entry = {"name": "guardrails", "enabled": True, "settings": dict_settings}
        monkeypatch.setenv("AI_MODELS_CONFIG", json.dumps({"middleware": [entry]}))
        from ai_api_unified.ai_models_config import clear_models_config_cache

        clear_models_config_cache()
        reset_guardrails()

    return _enable


def _check(**kwargs: Any) -> GuardrailModelCheckSettingsModel:
    return GuardrailModelCheckSettingsModel(**kwargs)


class TestSettings:
    @pytest.mark.parametrize(
        "str_backend", ["openai_moderation", "mistral_moderation", "google_moderation"]
    )
    def test_injection_rejects_backends_that_cannot_detect_it(
        self, str_backend: str
    ) -> None:
        with pytest.raises(ValueError, match="cannot use"):
            GuardrailsSettingsModel.model_validate(
                {"prompt_injection": {"backend": str_backend}}
            )

    def test_bedrock_needs_a_guardrail_id(self) -> None:
        with pytest.raises(ValueError, match="guardrail_id"):
            GuardrailsSettingsModel.model_validate(
                {"harmful_content": {"backend": "bedrock_guardrails"}}
            )
        GuardrailsSettingsModel.model_validate(
            {
                "harmful_content": {"backend": "bedrock_guardrails"},
                "backends": {"bedrock_guardrails": {"guardrail_id": "gr-1"}},
            }
        )

    def test_normalize_category(self) -> None:
        assert mb.normalize_category("SelfHarm") == "self_harm"
        assert (
            mb.normalize_category("Death, Harm & Tragedy") == "death_harm_and_tragedy"
        )
        assert mb.normalize_category("PROMPT_ATTACK") == "prompt_attack"
        assert (
            mb.normalize_category("hate_and_discrimination")
            == "hate_and_discrimination"
        )


class TestAuto:
    def _resolve(self, dict_settings: dict[str, Any] | None = None) -> tuple[str, str]:
        s = GuardrailsSettingsModel.model_validate(dict_settings or {})
        return (
            gr.resolve_backend("harmful_content", s.harmful_content, s),
            gr.resolve_backend("prompt_injection", s.prompt_injection, s),
        )

    def test_nothing_configured_is_the_classifier(self) -> None:
        assert self._resolve() == ("classifier", "classifier")

    def test_openai_leads_harmful_content(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "k")
        monkeypatch.setenv(mb.MISTRAL_API_KEY_SETTING, "k")
        assert self._resolve() == ("openai_moderation", "classifier")

    def test_mistral_then_google(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(mb.GOOGLE_LANGUAGE_API_KEY_SETTING, "k")
        assert self._resolve() == ("google_moderation", "classifier")
        monkeypatch.setenv(mb.MISTRAL_API_KEY_SETTING, "k")
        assert self._resolve() == ("mistral_moderation", "classifier")

    def test_azure_serves_both_checks_when_endpoint_and_key_exist(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(mb.AZURE_CONTENT_SAFETY_KEY_SETTING, "k")
        assert self._resolve() == ("classifier", "classifier")  # no endpoint yet
        monkeypatch.setenv(
            mb.AZURE_CONTENT_SAFETY_ENDPOINT_SETTING,
            "https://x.cognitiveservices.azure.com",
        )
        assert self._resolve() == ("azure_content_safety", "azure_content_safety")

    def test_bedrock_serves_both_checks_when_a_guardrail_is_named(self) -> None:
        assert self._resolve(
            {"backends": {"bedrock_guardrails": {"guardrail_id": "gr-1"}}}
        ) == (
            "bedrock_guardrails",
            "bedrock_guardrails",
        )


class TestMistral:
    def test_request_and_default_harm_categories(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(mb.MISTRAL_API_KEY_SETTING, "mk")
        http = _Http(
            {
                "/moderations": {
                    "results": [
                        {
                            "category_scores": {
                                "violence_and_threats": 0.91,
                                "pii": 0.99,
                                "sexual": 0.01,
                            }
                        }
                    ]
                }
            }
        )
        mb.set_moderation_http_client(http)
        s = GuardrailsSettingsModel()
        score, fired = mb.run_backend(
            "mistral_moderation",
            "harmful_content",
            _check(threshold=0.8),
            s,
            "input",
            "t",
        )
        assert fired == ("violence_and_threats",) and score == 0.91
        request = http.requests[0]
        assert request["url"] == "https://api.mistral.ai/v1/moderations"
        assert request["headers"]["Authorization"] == "Bearer mk"
        assert request["json"] == {"model": "mistral-moderation-latest", "input": ["t"]}

    def test_categories_override_the_defaults(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(mb.MISTRAL_API_KEY_SETTING, "mk")
        mb.set_moderation_http_client(
            _Http(
                {
                    "/moderations": {
                        "results": [
                            {
                                "category_scores": {
                                    "pii": 0.99,
                                    "violence_and_threats": 0.95,
                                }
                            }
                        ]
                    }
                }
            )
        )
        _, fired = mb.run_backend(
            "mistral_moderation",
            "harmful_content",
            _check(categories=["pii"]),
            GuardrailsSettingsModel(),
            "input",
            "t",
        )
        assert fired == ("pii",)


class TestAzure:
    ENDPOINT: str = "https://cs.cognitiveservices.azure.com"

    def _settings(self, monkeypatch: pytest.MonkeyPatch) -> GuardrailsSettingsModel:
        monkeypatch.setenv(mb.AZURE_CONTENT_SAFETY_KEY_SETTING, "ak")
        return GuardrailsSettingsModel.model_validate(
            {"backends": {"azure_content_safety": {"endpoint": self.ENDPOINT + "/"}}}
        )

    def test_analyze_maps_severity_to_a_score(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        s = self._settings(monkeypatch)
        http = _Http(
            {
                "text:analyze": {
                    "categoriesAnalysis": [
                        {"category": "Violence", "severity": 4},
                        {"category": "SelfHarm", "severity": 2},
                    ]
                }
            }
        )
        mb.set_moderation_http_client(http)
        score, fired = mb.run_backend(
            "azure_content_safety",
            "harmful_content",
            _check(threshold=0.5),
            s,
            "input",
            "t",
        )
        assert fired == ("violence",) and score == pytest.approx(4 / 6)
        _, fired = mb.run_backend(
            "azure_content_safety",
            "harmful_content",
            _check(threshold=0.8),
            s,
            "input",
            "t",
        )
        assert fired == ()
        request = http.requests[0]
        assert request["url"] == self.ENDPOINT + "/contentsafety/text:analyze"
        assert request["params"] == {"api-version": "2024-09-01"}
        assert request["headers"]["Ocp-Apim-Subscription-Key"] == "ak"
        assert request["json"] == {"text": "t", "outputType": "FourSeverityLevels"}

    def test_prompt_shields_for_injection(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        s = self._settings(monkeypatch)
        http = _Http(
            {
                "text:shieldPrompt": {
                    "userPromptAnalysis": {"attackDetected": True},
                    "documentsAnalysis": [],
                }
            }
        )
        mb.set_moderation_http_client(http)
        score, fired = mb.run_backend(
            "azure_content_safety",
            "prompt_injection",
            _check(),
            s,
            "input",
            "ignore it",
        )
        assert fired == ("prompt_attack",) and score == 1.0
        assert http.requests[0]["json"] == {"userPrompt": "ignore it", "documents": []}

    def test_long_text_is_chunked_at_the_service_limit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        s = self._settings(monkeypatch)
        http = _Http({"text:analyze": {"categoriesAnalysis": []}})
        mb.set_moderation_http_client(http)
        mb.run_backend(
            "azure_content_safety",
            "harmful_content",
            _check(),
            s,
            "input",
            "x" * 25_000,
        )
        assert [len(r["json"]["text"]) for r in http.requests] == [
            10_000,
            10_000,
            5_000,
        ]


class TestBedrock:
    RESPONSE: dict[str, Any] = {
        "action": "GUARDRAIL_INTERVENED",
        "assessments": [
            {
                "contentPolicy": {
                    "filters": [
                        {
                            "type": "PROMPT_ATTACK",
                            "confidence": "HIGH",
                            "action": "BLOCKED",
                            "detected": True,
                        },
                        {
                            "type": "VIOLENCE",
                            "confidence": "MEDIUM",
                            "action": "BLOCKED",
                            "detected": True,
                        },
                        {
                            "type": "HATE",
                            "confidence": "LOW",
                            "action": "NONE",
                            "detected": False,
                        },
                    ]
                },
                "topicPolicy": {
                    "topics": [
                        {
                            "name": "Investment Advice",
                            "type": "DENY",
                            "action": "BLOCKED",
                            "detected": True,
                        }
                    ]
                },
            }
        ],
    }

    def _settings(self) -> GuardrailsSettingsModel:
        return GuardrailsSettingsModel.model_validate(
            {
                "backends": {
                    "bedrock_guardrails": {
                        "guardrail_id": "gr-1",
                        "guardrail_version": "3",
                    }
                }
            }
        )

    def test_prompt_attack_serves_injection_only(self) -> None:
        client = _Bedrock(self.RESPONSE)
        mb.set_bedrock_guardrails_client(client)
        score, fired = mb.run_backend(
            "bedrock_guardrails",
            "prompt_injection",
            _check(),
            self._settings(),
            "input",
            "t",
        )
        assert fired == ("prompt_attack",) and score == 1.0
        assert client.calls[0] == {
            "guardrailIdentifier": "gr-1",
            "guardrailVersion": "3",
            "source": "INPUT",
            "content": [{"text": {"text": "t"}}],
            "outputScope": "FULL",
        }

    def test_harmful_content_reads_filters_topics_and_words(self) -> None:
        mb.set_bedrock_guardrails_client(_Bedrock(self.RESPONSE))
        score, fired = mb.run_backend(
            "bedrock_guardrails",
            "harmful_content",
            _check(threshold=0.5),
            self._settings(),
            "output",
            "t",
        )
        assert fired == ("topic_investment_advice", "violence")
        assert score == 1.0

    def test_answer_is_checked_as_output(self) -> None:
        client = _Bedrock({"assessments": []})
        mb.set_bedrock_guardrails_client(client)
        mb.run_backend(
            "bedrock_guardrails",
            "harmful_content",
            _check(),
            self._settings(),
            "output",
            "t",
        )
        assert client.calls[0]["source"] == "OUTPUT"


class TestGoogle:
    def test_confidence_and_topic_labels_ignored_by_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(mb.GOOGLE_LANGUAGE_API_KEY_SETTING, "gk")
        http = _Http(
            {
                "moderateText": {
                    "moderationCategories": [
                        {"name": "Toxic", "confidence": 0.92},
                        {"name": "Health", "confidence": 0.99},
                        {"name": "Firearms & Weapons", "confidence": 0.4},
                    ]
                }
            }
        )
        mb.set_moderation_http_client(http)
        score, fired = mb.run_backend(
            "google_moderation",
            "harmful_content",
            _check(),
            GuardrailsSettingsModel(),
            "input",
            "t",
        )
        assert fired == ("toxic",) and score == 0.92
        request = http.requests[0]
        assert request["url"] == mb.GOOGLE_MODERATE_URL
        assert request["params"] == {"key": "gk"}
        assert request["json"] == {
            "document": {"type": "PLAIN_TEXT", "content": "t"},
            "modelVersion": "MODEL_VERSION_2",
        }


class TestThroughACall:
    def test_a_provider_block_stops_the_call_before_the_answering_model(
        self, guardrails_on: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(mb.MISTRAL_API_KEY_SETTING, "mk")
        guardrails_on({"prompt_injection": {"enabled": False}})
        mb.set_moderation_http_client(
            _Http(
                {
                    "/moderations": {
                        "results": [{"category_scores": {"violence_and_threats": 0.97}}]
                    }
                }
            )
        )
        engine = _PricedEngine("p")
        with pytest.raises(AiGuardrailBlockedError) as raised:
            engine.send_prompt("a threat")
        assert raised.value.backend == "mistral_moderation"
        assert raised.value.categories == ("violence_and_threats",)
        assert engine.calls == []

    def test_a_provider_error_follows_on_error(
        self, guardrails_on: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(mb.MISTRAL_API_KEY_SETTING, "mk")
        guardrails_on({"prompt_injection": {"enabled": False}})
        mb.set_moderation_http_client(
            _Http({"/moderations": _Response({"error": "x"}, 401)})
        )
        engine = _PricedEngine("p")
        with pytest.raises(AiGuardrailBlockedError, match="could not run"):
            engine.send_prompt("hello")
        assert engine.calls == []
        guardrails_on({"prompt_injection": {"enabled": False}, "on_error": "allow"})
        mb.set_moderation_http_client(
            _Http({"/moderations": _Response({"error": "x"}, 401)})
        )
        assert engine.send_prompt("hello") == "p:ok"
