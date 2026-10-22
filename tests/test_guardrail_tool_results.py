# tests/test_guardrail_tool_results.py
"""
prompt_injection on tool results: each engine's tool-result shape is found,
only results after the model's last turn are new, each new result is judged
alone with the tool-result question, and a finding names the tool calls.

Safe by construction: every payload is neutral data plus an inert marker
word that a fake classifier keys on. No text here resembles an attack, and
no test reaches a provider or a generating model.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from ai_api_unified import (
    AIBaseClassifier,
    AIClassificationResult,
    AIClassifierCapabilitiesBase,
    AiGuardrailBlockedError,
    reset_guardrails,
    set_guardrail_classifier,
)
from ai_api_unified.middleware import guardrails as gr
from ai_api_unified.middleware import moderation_backends as mb
from ai_api_unified.middleware.middleware_config import (
    GuardrailModelCheckSettingsModel,
    GuardrailsSettingsModel,
)
from tests.test_turn_result_envelope import _TemplateEngine

from ai_api_unified.middleware.response_cache import RESPONSE_CACHE_ACTIVE


def _run_checks(*args: Any) -> list[gr.GuardrailFinding]:
    """run_checks as a guarded public call runs it: inside the call boundary,
    so the classifier's own call is not guarded again."""
    token: Any = RESPONSE_CACHE_ACTIVE.set(True)
    try:
        return gr.run_checks(*args)
    finally:
        RESPONSE_CACHE_ACTIVE.reset(token)


# The fake classifier fires on this word and nothing else.
MARKER: str = "MARKER_FIRE"
DATA_CLEAN: dict[str, Any] = {"city": "Lisbon", "temperature_c": 21}
DATA_MARKED: dict[str, Any] = {"city": "Porto", "note": MARKER}

# Injection off for the request text; only the tool-result pass and the
# checks a test names run.
INJECTION_ONLY: dict[str, Any] = {
    "prompt_injection": {"backend": "classifier"},
    "harmful_content": {"enabled": False},
}


class _MarkerJudge(AIBaseClassifier):
    """Answers 0.95 when the text holds the marker, else 0.0; records calls."""

    def __init__(self) -> None:
        super().__init__(model="marker-judge")
        self.calls: list[tuple[str, str]] = []

    @property
    def list_model_names(self) -> list[str]:
        return ["marker-judge"]

    @property
    def capabilities(self) -> AIClassifierCapabilitiesBase:
        return AIClassifierCapabilitiesBase(supports_predicate=True)

    def _classify_predicate_provider(
        self, str_text: str, *, instructions: str, **_: Any
    ) -> AIClassificationResult:
        self.calls.append((instructions, str_text))
        return AIClassificationResult(
            question_type="predicate",
            probability=0.95 if MARKER in str_text else 0.0,
        )

    def tool_calls(self) -> list[str]:
        """Texts judged with the tool-result question."""
        return [t for i, t in self.calls if i == gr.INSTRUCTIONS_TOOL_RESULT_INJECTION]

    def request_calls(self) -> list[str]:
        """Texts judged with the request question."""
        return [t for i, t in self.calls if i == gr.INSTRUCTIONS_PROMPT_INJECTION]


@pytest.fixture
def judge(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Turns guardrails on with the given settings and the marker judge."""
    from ai_api_unified.ai_models_config import clear_models_config_cache

    marker_judge: _MarkerJudge = _MarkerJudge()

    def _enable(dict_settings: dict[str, Any]) -> _MarkerJudge:
        entry: dict[str, Any] = {
            "name": "guardrails",
            "enabled": True,
            "settings": dict_settings,
        }
        monkeypatch.setenv("AI_MODELS_CONFIG", json.dumps({"middleware": [entry]}))
        monkeypatch.setattr(gr, "_openai_key_present", lambda: False)
        monkeypatch.setattr(mb, "_setting", lambda k: "")
        clear_models_config_cache()
        reset_guardrails()
        set_guardrail_classifier(marker_judge)
        return marker_judge

    yield _enable
    reset_guardrails()


# One tool result per engine shape, as build_tool_result_message makes it.
def _openai_chat(str_id: str, data: dict[str, Any]) -> dict[str, Any]:
    return {"role": "tool", "tool_call_id": str_id, "content": json.dumps(data)}


def _openai_responses(str_id: str, data: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function_call_output",
        "call_id": str_id,
        "output": json.dumps(data),
    }


def _anthropic(str_id: str, data: dict[str, Any]) -> dict[str, Any]:
    return {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": str_id,
                "content": json.dumps(data),
                "is_error": False,
            }
        ],
    }


def _gemini(str_id: str, data: dict[str, Any]) -> dict[str, Any]:
    return {
        "role": "user",
        "parts": [{"function_response": {"name": str_id, "response": data}}],
    }


def _bedrock(str_id: str, data: dict[str, Any]) -> dict[str, Any]:
    return {
        "role": "user",
        "content": [{"toolResult": {"toolUseId": str_id, "content": [{"json": data}]}}],
    }


LIST_SHAPES: list[Any] = [
    _openai_chat,
    _openai_responses,
    _anthropic,
    _gemini,
    _bedrock,
]
USER: dict[str, Any] = {"role": "user", "content": "What is the weather?"}
MODEL_TURN: dict[str, Any] = {"role": "assistant", "content": "Checking."}


class TestFindingToolResults:
    @pytest.mark.parametrize("shape", LIST_SHAPES)
    def test_every_engine_shape_is_found(self, shape: Any) -> None:
        found = gr.new_tool_results(
            {"messages": [USER, MODEL_TURN, shape("call_1", DATA_MARKED)]}
        )
        assert [t.tool_call_id for t in found] == ["call_1"]
        assert MARKER in found[0].text

    @pytest.mark.parametrize("shape", LIST_SHAPES)
    def test_tool_results_are_not_request_text(self, shape: Any) -> None:
        texts = gr.input_texts(
            {"messages": [USER, MODEL_TURN, shape("call_1", DATA_MARKED)]}
        )
        assert texts == ["What is the weather?", "Checking."]

    def test_only_results_after_the_last_model_turn_are_new(self) -> None:
        messages = [
            USER,
            MODEL_TURN,
            _openai_chat("old", DATA_MARKED),
            {"role": "assistant", "content": "One more."},
            _openai_chat("new", DATA_CLEAN),
        ]
        assert [
            t.tool_call_id for t in gr.new_tool_results({"messages": messages})
        ] == ["new"]

    def test_responses_model_items_end_the_old_results(self) -> None:
        messages = [
            _openai_responses("old", DATA_MARKED),
            {"type": "function_call", "call_id": "new", "name": "weather"},
            _openai_responses("new", DATA_CLEAN),
        ]
        assert [
            t.tool_call_id for t in gr.new_tool_results({"messages": messages})
        ] == ["new"]

    def test_gemini_text_parts_are_request_text(self) -> None:
        texts = gr.input_texts(
            {"messages": [{"role": "user", "parts": [{"text": "Weather please"}]}]}
        )
        assert texts == ["Weather please"]

    def test_bedrock_text_blocks_are_joined(self) -> None:
        message = {
            "role": "user",
            "content": [
                {
                    "toolResult": {
                        "toolUseId": "c",
                        "content": [{"text": "line one"}, {"json": {"n": 2}}],
                    }
                }
            ],
        }
        (found,) = gr.new_tool_results({"messages": [message]})
        assert found.text == 'line one\n{"n": 2}'


class TestToolResultCheck:
    def test_each_result_is_judged_alone_with_the_tool_question(
        self, judge: Any
    ) -> None:
        marker_judge = judge(
            {
                **INJECTION_ONLY,
                "prompt_injection": {
                    "backend": "classifier",
                    "tool_results": {"action": "flag"},
                },
            }
        )
        (finding,) = _run_checks(
            "send_conversation",
            "input",
            ["What is the weather?"],
            [
                gr.ToolResultText("a", json.dumps(DATA_CLEAN)),
                gr.ToolResultText("b", json.dumps(DATA_MARKED)),
            ],
        )
        assert marker_judge.request_calls() == ["What is the weather?"]
        assert marker_judge.tool_calls() == [
            json.dumps(DATA_CLEAN),
            json.dumps(DATA_MARKED),
        ]
        assert finding.tool_call_ids == ("b",)

    def test_block_names_the_tool_calls(self, judge: Any) -> None:
        judge(INJECTION_ONLY)
        with pytest.raises(AiGuardrailBlockedError) as info:
            _run_checks(
                "send_conversation",
                "input",
                [],
                [
                    gr.ToolResultText("a", json.dumps(DATA_CLEAN)),
                    gr.ToolResultText("b", json.dumps(DATA_MARKED)),
                ],
            )
        error = info.value
        assert error.check == "prompt_injection"
        assert error.source == "tool_result"
        assert error.tool_call_ids == ("b",)
        assert "tool calls b" in str(error)
        assert MARKER not in str(error)

    def test_flag_action_returns_a_finding(self, judge: Any) -> None:
        judge(
            {
                **INJECTION_ONLY,
                "prompt_injection": {
                    "backend": "classifier",
                    "tool_results": {"action": "flag"},
                },
            }
        )
        (finding,) = _run_checks(
            "send_conversation",
            "input",
            [],
            [gr.ToolResultText("b", json.dumps(DATA_MARKED))],
        )
        assert finding.source == "tool_result"
        assert finding.tool_call_ids == ("b",)
        assert finding.action == "flag"
        assert finding.score == pytest.approx(0.95)

    def test_own_threshold_wins(self, judge: Any) -> None:
        judge(
            {
                **INJECTION_ONLY,
                "prompt_injection": {
                    "backend": "classifier",
                    "tool_results": {"threshold": 0.99},
                },
            }
        )
        assert (
            _run_checks(
                "send_conversation",
                "input",
                [],
                [gr.ToolResultText("b", json.dumps(DATA_MARKED))],
            )
            == []
        )

    def test_request_text_keeps_its_own_action(self, judge: Any) -> None:
        judge(
            {
                **INJECTION_ONLY,
                "prompt_injection": {
                    "backend": "classifier",
                    "action": "flag",
                    "tool_results": {"action": "block"},
                },
            }
        )
        (finding,) = _run_checks("send_prompt", "input", [f"note {MARKER}"])
        assert finding.source == "request" and finding.action == "flag"

    def test_disabled_folds_tool_results_into_the_request_check(
        self, judge: Any
    ) -> None:
        marker_judge = judge(
            {
                **INJECTION_ONLY,
                "prompt_injection": {
                    "backend": "classifier",
                    "action": "flag",
                    "tool_results": {"enabled": False},
                },
            }
        )
        (finding,) = _run_checks(
            "send_conversation",
            "input",
            ["What is the weather?"],
            [gr.ToolResultText("b", json.dumps(DATA_MARKED))],
        )
        assert marker_judge.tool_calls() == []
        assert finding.source == "request"
        assert MARKER in marker_judge.request_calls()[0]

    def test_injection_off_skips_tool_results(self, judge: Any) -> None:
        marker_judge = judge(
            {
                "prompt_injection": {"enabled": False},
                "harmful_content": {"enabled": False},
            }
        )
        assert (
            _run_checks(
                "send_conversation",
                "input",
                [],
                [gr.ToolResultText("b", json.dumps(DATA_MARKED))],
            )
            == []
        )
        assert marker_judge.calls == []

    def test_other_checks_read_new_tool_results(self, judge: Any) -> None:
        judge(
            {
                "prompt_injection": {"enabled": False},
                "harmful_content": {"enabled": False},
                "custom": [
                    {"name": "marker_rule", "terms": [MARKER], "action": "flag"}
                ],
            }
        )
        (finding,) = _run_checks(
            "send_conversation",
            "input",
            ["What is the weather?"],
            [gr.ToolResultText("b", json.dumps(DATA_MARKED))],
        )
        assert finding.check == "marker_rule" and finding.source == "request"


class TestThroughAConversation:
    def test_flag_stamps_the_turn(self, judge: Any) -> None:
        judge(
            {
                **INJECTION_ONLY,
                "prompt_injection": {
                    "backend": "classifier",
                    "tool_results": {"action": "flag"},
                },
            }
        )
        turn = _TemplateEngine("p").send_conversation(
            "sys", [USER, MODEL_TURN, _anthropic("toolu_1", DATA_MARKED)]
        )
        assert turn.guardrail_flags == ["prompt_injection"]
        assert turn.guardrail_flagged_tool_calls == ["toolu_1"]

    def test_block_stops_before_the_provider(self, judge: Any) -> None:
        judge(INJECTION_ONLY)
        engine = _TemplateEngine("p")
        with pytest.raises(AiGuardrailBlockedError) as info:
            engine.send_conversation(
                "sys", [USER, MODEL_TURN, _gemini("weather", DATA_MARKED)]
            )
        assert info.value.tool_call_ids == ("weather",)
        assert not hasattr(engine, "last_kwargs")

    def test_old_results_are_not_checked_again(self, judge: Any) -> None:
        marker_judge = judge(INJECTION_ONLY)
        turn = _TemplateEngine("p").send_conversation(
            "sys",
            [
                USER,
                MODEL_TURN,
                _openai_chat("old", DATA_MARKED),
                {"role": "assistant", "content": "Done."},
                {"role": "user", "content": "Thanks."},
            ],
        )
        assert turn.guardrail_flagged_tool_calls == []
        assert marker_judge.tool_calls() == []

    def test_clean_results_pass(self, judge: Any) -> None:
        judge(INJECTION_ONLY)
        turn = _TemplateEngine("p").send_conversation(
            "sys", [USER, MODEL_TURN, _bedrock("tooluse_1", DATA_CLEAN)]
        )
        assert turn.guardrail_flags == []


class _Response:
    def __init__(self, body: Any) -> None:
        self.body = body
        self.status_code = 200

    def json(self) -> Any:
        return self.body


class _Http:
    def __init__(self, body: Any) -> None:
        self.body = body
        self.requests: list[dict[str, Any]] = []

    def post(self, url: str, **kwargs: Any) -> _Response:
        self.requests.append({"url": url, **kwargs})
        return _Response(self.body)


class TestAzureDocuments:
    def _run(
        self, monkeypatch: pytest.MonkeyPatch, body: dict[str, Any]
    ) -> tuple[tuple[str, ...], _Http]:
        monkeypatch.setattr(mb, "_setting", lambda k: "ak")
        http = _Http(body)
        mb.set_moderation_http_client(http)
        settings = GuardrailsSettingsModel.model_validate(
            {"backends": {"azure_content_safety": {"endpoint": "https://cs.example"}}}
        )
        try:
            _, fired = mb.run_backend(
                "azure_content_safety",
                "prompt_injection",
                GuardrailModelCheckSettingsModel(),
                settings,
                "input",
                "tool text",
                bool_document=True,
            )
        finally:
            mb.reset_moderation_backends()
        return fired, http

    def test_tool_result_is_sent_as_a_document(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fired, http = self._run(
            monkeypatch,
            {
                "userPromptAnalysis": {"attackDetected": False},
                "documentsAnalysis": [{"attackDetected": True}],
            },
        )
        assert fired == ("prompt_attack",)
        assert http.requests[0]["json"] == {
            "userPrompt": "",
            "documents": ["tool text"],
        }

    def test_clean_document_does_not_fire(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fired, _ = self._run(
            monkeypatch,
            {
                "userPromptAnalysis": {"attackDetected": False},
                "documentsAnalysis": [{"attackDetected": False}],
            },
        )
        assert fired == ()


class TestSettings:
    def test_defaults_inherit_the_check(self) -> None:
        tool_results = GuardrailsSettingsModel().prompt_injection.tool_results
        assert tool_results.enabled is True
        assert tool_results.threshold is None and tool_results.action is None

    @pytest.mark.parametrize(
        "bad",
        [
            {"harmful_content": {"tool_results": {"enabled": False}}},
            {"prompt_injection": {"tool_results": {"action": "shout"}}},
            {"prompt_injection": {"tool_results": {"threshold": 2}}},
            {"prompt_injection": {"tool_results": {"unknown": 1}}},
        ],
    )
    def test_invalid(self, bad: dict[str, Any]) -> None:
        with pytest.raises(ValueError):
            GuardrailsSettingsModel.model_validate(bad)
