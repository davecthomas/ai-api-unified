# ruff: noqa: E402

# test_openai_compatible_completions.py
"""
Tests for the OpenAI-compatible completions base and the generic
`openai-compatible` engine, against a mocked OpenAI SDK client.

Covers configuration resolution (base URL, API key, model), the vendor
settings a subclass overrides (token field, structured-output mode,
catalogue, image input, reasoning), the OpenAI-only features switched off,
factory resolution, and that the OpenAI engine itself is unchanged.
"""

import json
from typing import Any, ClassVar
from unittest.mock import Mock, patch

import pytest

pytest.importorskip("openai")

from ai_api_unified.ai_base import (
    AICompletionsPromptParamsBase,
    AIFinishReason,
    AIStructuredPrompt,
    SupportedDataType,
)
from ai_api_unified.ai_completions_exceptions import StructuredResponseTokenLimitError
from ai_api_unified.ai_factory import AIFactory
from ai_api_unified.ai_provider_exceptions import AiProviderConfigurationError
from ai_api_unified.completions.ai_openai_compatible_completions import (
    PLACEHOLDER_API_KEY,
    AiOpenAICompatibleCompletions,
)
from ai_api_unified.completions.ai_openai_completions import AiOpenAICompletions

TEST_BASE_URL: str = "http://localhost:8000/v1"
TEST_MODEL: str = "local-model"
GRAPH_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"nodes": {"type": "array", "items": {"type": "object"}}},
    "required": ["nodes"],
}


def _settings(**overrides: str) -> dict[str, str]:
    dict_settings: dict[str, str] = {
        "OPENAI_COMPATIBLE_BASE_URL": TEST_BASE_URL,
        "COMPLETIONS_MODEL_NAME": TEST_MODEL,
    }
    dict_settings.update(overrides)
    return dict_settings


def _patch_settings(dict_settings: dict[str, str]) -> Any:
    def _get_setting(str_key: str, default: Any = None) -> Any:
        return dict_settings.get(str_key, default)

    return patch(
        "ai_api_unified.util.env_settings.EnvSettings.get_setting",
        side_effect=_get_setting,
    )


def _build_client(
    cls: type[AiOpenAICompatibleCompletions] = AiOpenAICompatibleCompletions,
    **overrides: str,
) -> AiOpenAICompatibleCompletions:
    with _patch_settings(_settings(**overrides)):
        client = cls()
    client.client = Mock()
    return client


def _chat_response(content: str, finish_reason: str = "stop") -> Mock:
    message = Mock(spec=["content", "tool_calls", "refusal", "model_dump"])
    message.content = content
    message.tool_calls = None
    message.refusal = None
    message.model_dump = Mock(side_effect=TypeError("test double"))
    return Mock(
        choices=[Mock(message=message, finish_reason=finish_reason)],
        usage=Mock(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            prompt_tokens_details=Mock(cached_tokens=None),
        ),
    )


class _Graph(AIStructuredPrompt):
    nodes: list[dict[str, Any]] | None = None

    @staticmethod
    def get_prompt(input_text: str = "") -> str:
        return f"Build a graph from: {input_text}"


class _VendorCompletions(AiOpenAICompatibleCompletions):
    """A vendor subclass shaped like the DeepSeek/Qwen/Z.ai engines."""

    PROVIDER_REGISTRY_LABEL: ClassVar[str] = "vendorx"
    PROVIDER_DISPLAY_NAME: ClassVar[str] = "VendorX"
    PROVIDER_ENGINE_TOKEN: ClassVar[str] = "vendorx"
    API_KEY_SETTING: ClassVar[str] = "VENDORX_API_KEY"
    BOOL_API_KEY_REQUIRED: ClassVar[bool] = True
    BASE_URL_SETTING: ClassVar[str] = "VENDORX_BASE_URL_OVERRIDE"
    DEFAULT_BASE_URL: ClassVar[str | None] = "https://api.vendorx.example/v1"
    DEFAULT_COMPLETIONS_MODEL: ClassVar[str] = "vx-chat"
    STRUCTURED_OUTPUT_MODE: ClassVar[str] = "json_object"
    STRUCTURED_OUTPUT_MODE_SETTING: ClassVar[str | None] = None
    CONTEXT_WINDOW_SETTING: ClassVar[str | None] = None
    LIST_CATALOGUED_MODELS: ClassVar[list[str]] = ["vx-chat", "vx-reasoner"]
    DICT_CONTEXT_WINDOWS: ClassVar[dict[str, int]] = {
        "vx-chat": 128_000,
        "vx-reasoner": 128_000,
    }
    SET_IMAGE_INPUT_MODELS: ClassVar[frozenset[str]] = frozenset({"vx-chat"})
    SET_REASONING_MODELS: ClassVar[frozenset[str]] = frozenset({"vx-reasoner"})


def _build_vendor_client(**overrides: str) -> _VendorCompletions:
    dict_settings: dict[str, str] = {"VENDORX_API_KEY": "vx-key"}
    dict_settings.update(overrides)
    with _patch_settings(dict_settings):
        client = _VendorCompletions()
    client.client = Mock()
    return client


class TestConfiguration:
    def test_base_url_is_required_for_the_generic_engine(self) -> None:
        with _patch_settings({"COMPLETIONS_MODEL_NAME": TEST_MODEL}):
            with pytest.raises(
                AiProviderConfigurationError, match="OPENAI_COMPATIBLE_BASE_URL"
            ):
                AiOpenAICompatibleCompletions()

    def test_plaintext_remote_base_url_is_rejected(self) -> None:
        with _patch_settings(
            _settings(OPENAI_COMPATIBLE_BASE_URL="http://gateway.example.com/v1")
        ):
            with pytest.raises(AiProviderConfigurationError, match="https"):
                AiOpenAICompatibleCompletions()

    def test_model_is_required_for_the_generic_engine(self) -> None:
        with _patch_settings({"OPENAI_COMPATIBLE_BASE_URL": TEST_BASE_URL}):
            with pytest.raises(AiProviderConfigurationError, match="needs a model"):
                AiOpenAICompatibleCompletions()

    def test_optional_key_falls_back_to_placeholder(self) -> None:
        client = _build_client()
        assert client.api_key == PLACEHOLDER_API_KEY
        assert client.base_url == TEST_BASE_URL

    def test_caller_base_url_wins(self) -> None:
        with _patch_settings(_settings()):
            client = AiOpenAICompatibleCompletions(
                base_url="https://llm.internal.example/v1"
            )
        assert client.base_url == "https://llm.internal.example/v1"

    def test_vendor_required_key_is_enforced(self) -> None:
        with _patch_settings({}):
            with pytest.raises(ValueError, match="VENDORX_API_KEY"):
                _VendorCompletions()

    def test_vendor_defaults_apply(self) -> None:
        client = _build_vendor_client()
        assert client.base_url == "https://api.vendorx.example/v1"
        assert client.completions_model == "vx-chat"
        assert client.api_key == "vx-key"

    def test_vendor_base_url_override(self) -> None:
        client = _build_vendor_client(
            VENDORX_BASE_URL_OVERRIDE="https://proxy.example/v1"
        )
        assert client.base_url == "https://proxy.example/v1"

    def test_invalid_structured_mode_rejected(self) -> None:
        with _patch_settings(_settings(OPENAI_COMPATIBLE_STRUCTURED_OUTPUT="xml")):
            with pytest.raises(AiProviderConfigurationError, match="json_object"):
                AiOpenAICompatibleCompletions()


class TestCapabilities:
    def test_generic_context_window_from_environment(self) -> None:
        client = _build_client(OPENAI_COMPATIBLE_CONTEXT_WINDOW="32768")
        assert client.capabilities.context_window_length == 32_768
        assert client.max_context_tokens == 32_768
        assert client.capabilities.pricing is None
        assert client.list_model_names == [TEST_MODEL]

    def test_generic_context_window_unknown_skips_guard(self) -> None:
        client = _build_client()
        assert client.max_context_tokens == 0

    def test_invalid_context_window_rejected(self) -> None:
        with _patch_settings(_settings(OPENAI_COMPATIBLE_CONTEXT_WINDOW="big")):
            with pytest.raises(AiProviderConfigurationError, match="positive"):
                AiOpenAICompatibleCompletions()

    def test_vendor_catalogue_drives_capabilities(self) -> None:
        chat = _build_vendor_client()
        assert chat.list_model_names == ["vx-chat", "vx-reasoner"]
        assert chat.max_context_tokens == 128_000
        assert SupportedDataType.IMAGE in chat.capabilities.supported_data_types
        assert chat.capabilities.reasoning is False

        reasoner = _build_vendor_client(COMPLETIONS_MODEL_NAME="vx-reasoner")
        assert reasoner.capabilities.reasoning is True
        assert reasoner.capabilities.supported_data_types == [SupportedDataType.TEXT]


class TestRequestShape:
    def test_send_prompt_uses_max_tokens(self) -> None:
        client = _build_client()
        client.client.chat.completions.create.return_value = _chat_response("hi")
        assert client.send_prompt("Hello", max_response_tokens=64) == "hi"
        kwargs = client.client.chat.completions.create.call_args.kwargs
        assert kwargs["max_tokens"] == 64
        assert "max_completion_tokens" not in kwargs

    def test_structured_json_schema_mode_by_default(self) -> None:
        client = _build_client()
        client.client.chat.completions.create.return_value = _chat_response(
            json.dumps({"nodes": []})
        )
        result = client.send_structured_output("Compile.", response_schema=GRAPH_SCHEMA)
        assert result.data == {"nodes": []}
        kwargs = client.client.chat.completions.create.call_args.kwargs
        assert kwargs["response_format"]["type"] == "json_schema"
        assert "max_tokens" in kwargs

    def test_structured_json_object_mode_puts_schema_in_system_prompt(self) -> None:
        client = _build_vendor_client()
        client.client.chat.completions.create.return_value = _chat_response(
            json.dumps({"nodes": [{"kind": "task"}]})
        )
        result = client.send_structured_output(
            "Compile.",
            response_schema=GRAPH_SCHEMA,
            provider_options={"temperature": 0},
        )
        assert result.data == {"nodes": [{"kind": "task"}]}
        kwargs = client.client.chat.completions.create.call_args.kwargs
        assert kwargs["response_format"] == {"type": "json_object"}
        system_content: str = kwargs["messages"][0]["content"]
        assert json.dumps(GRAPH_SCHEMA, sort_keys=True) in system_content
        # provider_options still merge into the raw request.
        assert kwargs["temperature"] == 0

    def test_strict_schema_prompt_uses_response_format_not_functions(self) -> None:
        client = _build_client()
        client.client.chat.completions.create.return_value = _chat_response(
            json.dumps({"nodes": [{"id": 1}]})
        )
        result = client.strict_schema_prompt("Compile.", _Graph)
        assert isinstance(result, _Graph)
        assert result.nodes == [{"id": 1}]
        kwargs = client.client.chat.completions.create.call_args.kwargs
        assert "functions" not in kwargs
        assert "response_format" in kwargs

    def test_strict_schema_prompt_truncation_raises_token_limit(self) -> None:
        client = _build_client()
        client.client.chat.completions.create.return_value = _chat_response(
            '{"nodes": [', finish_reason="length"
        )
        with pytest.raises(StructuredResponseTokenLimitError):
            client.strict_schema_prompt("Compile.", _Graph)

    def test_images_rejected_for_text_only_model(self) -> None:
        client = _build_vendor_client(COMPLETIONS_MODEL_NAME="vx-reasoner")
        params = AICompletionsPromptParamsBase(
            included_types=[SupportedDataType.IMAGE],
            included_data=[b"\x89PNG\r\n\x1a\n"],
            included_mime_types=["image/png"],
        )
        with pytest.raises(ValueError, match="does not accept image input"):
            client.send_prompt("Describe", other_params=params)
        client.client.chat.completions.create.assert_not_called()

    def test_images_sent_for_image_model(self) -> None:
        client = _build_vendor_client()
        client.client.chat.completions.create.return_value = _chat_response("a cat")
        params = AICompletionsPromptParamsBase(
            included_types=[SupportedDataType.IMAGE],
            included_data=[b"\x89PNG\r\n\x1a\n"],
            included_mime_types=["image/png"],
        )
        assert client.send_prompt("Describe", other_params=params) == "a cat"
        user_content = client.client.chat.completions.create.call_args.kwargs[
            "messages"
        ][1]["content"]
        assert user_content[1]["type"] == "image_url"


class TestOpenAIOnlyFeaturesOff:
    def test_org_info_reports_none(self) -> None:
        client = _build_client()
        org_info = client.get_org_info()
        assert org_info.org_id is None
        assert client.get_org_info_capability().supports_org_id is False
        client.client.models.with_raw_response.list.assert_not_called()

    def test_observability_labels_name_the_vendor(self) -> None:
        client = _build_vendor_client()
        assert client._resolve_observability_provider_vendor() == "vendorx"
        assert client._resolve_observability_provider_engine() == "vendorx"

    def test_request_errors_name_the_vendor(self) -> None:
        from openai import APIConnectionError

        client = _build_vendor_client()
        client.client.chat.completions.create.side_effect = APIConnectionError(
            request=Mock()
        )
        with pytest.raises(Exception, match="VendorX request failed"):
            client.send_conversation("sys", [{"role": "user", "content": "hi"}])


class TestFactoryAndOpenAIUnchanged:
    def test_factory_resolves_engine_and_base_url(self) -> None:
        with _patch_settings(_settings()):
            client = AIFactory.get_ai_completions_client(
                completions_engine="openai-compatible",
                base_url="http://127.0.0.1:11434/v1",
            )
        assert isinstance(client, AiOpenAICompatibleCompletions)
        assert client.base_url == "http://127.0.0.1:11434/v1"
        assert client.completions_model == TEST_MODEL

    def test_openai_engine_still_sends_max_completion_tokens(self) -> None:
        with _patch_settings({"OPENAI_API_KEY": "test-key"}):
            client = AiOpenAICompletions(model="gpt-5.6-luna")
        client.client = Mock()
        client.client.chat.completions.create.return_value = _chat_response("hi")
        client.send_prompt("Hello", max_response_tokens=64)
        kwargs = client.client.chat.completions.create.call_args.kwargs
        assert kwargs["max_completion_tokens"] == 64
        assert client.capabilities.pricing is not None

    def test_structured_length_finish_is_reported(self) -> None:
        client = _build_client()
        client.client.chat.completions.create.return_value = _chat_response(
            '{"nodes": [', finish_reason="length"
        )
        result = client.send_structured_output("Compile.", response_schema=GRAPH_SCHEMA)
        assert result.data is None
        assert result.finish_reason is AIFinishReason.LENGTH


# ── Inheritance guard ────────────────────────────────────────────────────────

from ai_api_unified.ai_openai_base import AIOpenAIBase

# Every method the OpenAI engine classes define, with how the compatible
# engine treats it:
#   "shared"      - Chat Completions protocol code, correct for any
#                   compatible server, inherited as is.
#   "overridden"  - OpenAI-specific; the compatible engine replaces it.
#   "unreachable" - OpenAI-only helper whose only caller is overridden.
DICT_REVIEWED_OPENAI_METHODS: dict[str, str] = {
    # AIOpenAIBase
    "_fetch_org_id_via_header_probe": "unreachable",
    "_fetch_org_info_via_account_api": "unreachable",
    "_get_org_info_capability_provider": "overridden",
    "_get_org_info_provider": "overridden",
    "_resolve_api_key": "overridden",
    "async_client": "shared",
    "get_api_base_url": "overridden",
    # AiOpenAICompletions
    "_asend_conversation_provider": "shared",
    "_asend_prompt_provider": "shared",
    "_asend_structured_output_provider": "shared",
    "_async_client_for_call": "shared",
    "_build_capabilities": "overridden",
    "_build_chat_conversation_request_kwargs": "shared",
    "_build_chat_provider_tools": "shared",
    "_build_chat_structured_request_kwargs": "overridden",
    "_build_conversation_observability_metadata": "shared",
    "_build_structured_observability_metadata": "shared",
    "_build_structured_output_result_from_parts": "shared",
    "_build_tool_result_message_provider": "shared",
    "_build_turn_result_from_chat": "shared",
    "_build_user_message_content": "overridden",
    "_client_for_call": "shared",
    "_extend_messages_with_turn_provider": "shared",
    "_extract_openai_cached_tokens": "shared",
    "_extract_openai_completion_tokens": "shared",
    "_extract_openai_prompt_tokens": "shared",
    "_extract_openai_total_tokens": "shared",
    "_observed_chat_structured_result": "shared",
    "_observed_chat_turn_result": "shared",
    "_raise_request_error": "shared",
    "_sdk_option_method": "shared",
    "_send_conversation_provider": "shared",
    "_send_prompt_streaming_provider": "shared",
    "_send_structured_output_provider": "shared",
    "_serialize_chat_assistant_message": "shared",
    "_sum_optional_ints": "shared",
    "_usage_from_chat_response": "shared",
    "capabilities": "shared",
    "list_model_names": "overridden",
    "max_context_tokens": "overridden",
    "send_prompt": "shared",
    "strict_schema_prompt": "overridden",
}


def _defined_methods(cls: type) -> set[str]:
    return {
        str_name
        for str_name, value in vars(cls).items()
        if not (str_name.startswith("__") and str_name.endswith("__"))
        and (
            callable(value) or isinstance(value, (property, staticmethod, classmethod))
        )
    }


class TestInheritanceGuard:
    """
    The compatible engine inherits from the OpenAI engine, so anything added
    there flows into every vendor engine. This guard makes that a decision:
    a new or removed OpenAI method fails here until it is classified above.
    If most new additions need overriding, extract a neutral Chat Completions
    base with openai and openai-compatible as siblings instead.
    """

    def test_every_openai_method_is_reviewed(self) -> None:
        set_defined: set[str] = _defined_methods(AIOpenAIBase) | _defined_methods(
            AiOpenAICompletions
        )
        set_reviewed: set[str] = set(DICT_REVIEWED_OPENAI_METHODS)
        set_unreviewed: set[str] = set_defined - set_reviewed
        set_stale: set[str] = set_reviewed - set_defined
        assert not set_unreviewed, (
            "New OpenAI engine methods are inherited by every OpenAI-compatible "
            f"vendor engine: {sorted(set_unreviewed)}. Decide whether each is "
            "shared protocol code or OpenAI-specific (override it in "
            "AiOpenAICompatibleCompletions), then classify it in "
            "DICT_REVIEWED_OPENAI_METHODS."
        )
        assert not set_stale, (
            f"Reviewed methods no longer exist: {sorted(set_stale)}. Remove them "
            "from DICT_REVIEWED_OPENAI_METHODS."
        )

    def test_overridden_methods_are_actually_overridden(self) -> None:
        set_compatible: set[str] = set(vars(AiOpenAICompatibleCompletions))
        list_missing: list[str] = [
            str_name
            for str_name, str_status in DICT_REVIEWED_OPENAI_METHODS.items()
            if str_status == "overridden" and str_name not in set_compatible
        ]
        assert (
            not list_missing
        ), f"Marked overridden but inherited unchanged: {list_missing}."

    def test_unreachable_helpers_have_their_callers_overridden(self) -> None:
        # The org lookups run only from _get_org_info_provider.
        assert "_get_org_info_provider" in vars(AiOpenAICompatibleCompletions)
