# ai_openai_compatible_completions.py

"""
Completions for servers that speak the OpenAI Chat Completions protocol.

Many vendors and self-hosted servers (DeepSeek, Alibaba Qwen, Z.ai, vLLM,
Ollama, LiteLLM) accept the OpenAI Chat Completions request shape through
the official `openai` SDK pointed at a different base URL. They differ from
OpenAI in a small, known set of ways, and this module turns each one into a
class setting:

- the API key and base-URL settings, and whether a key is required;
- the response token field (`max_tokens` rather than OpenAI's
  `max_completion_tokens`);
- structured output (`json_schema`, or `json_object` with the schema in the
  system prompt for servers that only guarantee valid JSON);
- the model catalog, context windows, image input, and reasoning support;
- the pricing and lifecycle registry label and observability vendor label.

OpenAI-only features are switched off: the organization identity lookups
and the legacy `functions` call that `strict_schema_prompt` uses on OpenAI.

`AiOpenAICompatibleCompletions` is itself the generic `openai-compatible`
engine, configured entirely from the environment. Vendor engines subclass it
and override the class settings.
"""

from __future__ import annotations

import json
import logging
from typing import Any, ClassVar, Literal, Type

from ..ai_base import (
    AIBaseCompletions,
    AICompletionsCapabilitiesBase,
    AIFinishReason,
    AICompletionsPromptParamsBase,
    AIProviderOrgInfoBase,
    AIProviderOrgInfoCapability,
    AIStructuredOutputResult,
    AIStructuredPrompt,
    SupportedDataType,
    resolve_base_url_override,
    validate_base_url_override,
)
from ..ai_provider_exceptions import AiProviderConfigurationError
from ..pricing.pricing_registry import get_model_pricing
from .ai_openai_completions import AiOpenAICompletions

_LOGGER: logging.Logger = logging.getLogger(__name__)

TypeStructuredOutputMode = Literal["json_schema", "json_object"]
STRUCTURED_OUTPUT_MODE_JSON_SCHEMA: str = "json_schema"
STRUCTURED_OUTPUT_MODE_JSON_OBJECT: str = "json_object"
# The SDK rejects an empty api_key, but local servers such as vLLM and
# Ollama accept any value, so an unset optional key sends this placeholder.
PLACEHOLDER_API_KEY: str = "not-needed"


class AICompletionsCapabilitiesOpenAICompatible(AICompletionsCapabilitiesBase):
    """Completions capabilities for an OpenAI-compatible server."""


class AiOpenAICompatibleCompletions(AiOpenAICompletions):
    """
    Chat Completions client for any OpenAI-compatible server.

    As the generic `openai-compatible` engine it reads:
    - OPENAI_COMPATIBLE_BASE_URL (required): the server's /v1 base URL;
      https:// unless it targets a loopback host.
    - OPENAI_COMPATIBLE_API_KEY (optional): sent as the bearer token.
    - COMPLETIONS_MODEL_NAME (required): the server's model id.
    - OPENAI_COMPATIBLE_STRUCTURED_OUTPUT (optional): "json_schema"
      (default) or "json_object" for servers without schema enforcement.
    - OPENAI_COMPATIBLE_CONTEXT_WINDOW (optional): input token limit for the
      context guard; unset disables the guard.

    Vendor subclasses override the class settings below instead.
    """

    # ── Vendor settings (override in subclasses) ────────────────────────────
    PROVIDER_REGISTRY_LABEL: ClassVar[str] = "openai-compatible"
    PROVIDER_DISPLAY_NAME: ClassVar[str] = "OpenAI-compatible server"
    PROVIDER_ENGINE_TOKEN: ClassVar[str] = "openai-compatible"
    API_KEY_SETTING: ClassVar[str] = "OPENAI_COMPATIBLE_API_KEY"
    BOOL_API_KEY_REQUIRED: ClassVar[bool] = False
    BASE_URL_SETTING: ClassVar[str] = "OPENAI_COMPATIBLE_BASE_URL"
    # Vendor endpoint used when neither a caller argument nor
    # BASE_URL_SETTING supplies one. None makes the base URL required.
    DEFAULT_BASE_URL: ClassVar[str | None] = None
    # Empty makes COMPLETIONS_MODEL_NAME (or the model argument) required.
    DEFAULT_COMPLETIONS_MODEL: ClassVar[str] = ""
    MAX_TOKENS_REQUEST_FIELD: ClassVar[str] = "max_tokens"
    STRUCTURED_OUTPUT_MODE: ClassVar[str] = STRUCTURED_OUTPUT_MODE_JSON_SCHEMA
    # Environment override for STRUCTURED_OUTPUT_MODE; vendors that know
    # their server's behavior set this to None.
    STRUCTURED_OUTPUT_MODE_SETTING: ClassVar[str | None] = (
        "OPENAI_COMPATIBLE_STRUCTURED_OUTPUT"
    )
    # Environment override for the context window of an uncataloged model.
    CONTEXT_WINDOW_SETTING: ClassVar[str | None] = "OPENAI_COMPATIBLE_CONTEXT_WINDOW"
    # Cataloged models, in display order, and their input context windows.
    LIST_CATALOGED_MODELS: ClassVar[list[str]] = []
    DICT_CONTEXT_WINDOWS: ClassVar[dict[str, int]] = {}
    # Models that accept image input and that reason before answering.
    SET_IMAGE_INPUT_MODELS: ClassVar[frozenset[str]] = frozenset()
    SET_REASONING_MODELS: ClassVar[frozenset[str]] = frozenset()

    def __init__(self, model: str = "", **kwargs: Any):
        """
        Initializes the client for the configured OpenAI-compatible server.

        Args:
            model: Model id; falls back to COMPLETIONS_MODEL_NAME, then the
                class DEFAULT_COMPLETIONS_MODEL.
            **kwargs: Passed through (base_url, retry_policy, and so on).

        Raises:
            AiProviderConfigurationError: When no base URL or model resolves.
            ValueError: When a required API key is missing.
        """
        AiOpenAICompletions.__init__(self, model=model, **kwargs)
        if not str(self.completions_model or "").strip():
            raise AiProviderConfigurationError(
                f"{self.PROVIDER_DISPLAY_NAME} needs a model: pass model= or "
                "set COMPLETIONS_MODEL_NAME."
            )
        self.structured_output_mode: str = self._resolve_structured_output_mode()

    # ── Configuration resolution ────────────────────────────────────────────

    def _resolve_api_key(self) -> str:
        """
        Returns the vendor API key, or a placeholder when the key is optional.

        Raises:
            ValueError: When the key is required and unset.
        """
        raw_api_key: object = self.env.get_setting(self.API_KEY_SETTING)
        str_api_key: str = str(raw_api_key or "").strip()
        if str_api_key:
            # Early return with the configured key.
            return str_api_key
        if self.BOOL_API_KEY_REQUIRED:
            raise ValueError(
                f"{self.API_KEY_SETTING} environment variable must be set."
            )
        # Normal return with the placeholder for key-less local servers.
        return PLACEHOLDER_API_KEY

    def get_api_base_url(self, *, base_url: str | None = None) -> str:
        """
        Resolves the server base URL.

        Precedence: the caller argument, then BASE_URL_SETTING, then the
        class DEFAULT_BASE_URL. OpenAI's geo-residency routing and its
        OPENAI_BASE_URL settings do not apply.

        Raises:
            AiProviderConfigurationError: When nothing resolves, or the URL
                would send the key over plaintext to a non-loopback host.
        """
        str_resolved: str | None = resolve_base_url_override(
            self.env,
            str_env_key=self.BASE_URL_SETTING,
            str_explicit=base_url,
        )
        if str_resolved is not None:
            # Early return with the caller- or environment-supplied URL.
            return str_resolved
        if self.DEFAULT_BASE_URL is not None:
            # Early return with the vendor's default endpoint.
            return validate_base_url_override(
                self.DEFAULT_BASE_URL, str_env_key=self.BASE_URL_SETTING
            )
        raise AiProviderConfigurationError(
            f"{self.BASE_URL_SETTING} must be set to the server's base URL "
            "(for example http://localhost:8000/v1)."
        )

    def _resolve_structured_output_mode(self) -> str:
        """
        Returns the structured-output mode, honoring the environment setting.

        Raises:
            AiProviderConfigurationError: When the setting holds an unknown mode.
        """
        str_mode: str = self.STRUCTURED_OUTPUT_MODE
        if self.STRUCTURED_OUTPUT_MODE_SETTING is not None:
            raw_mode: object = self.env.get_setting(
                self.STRUCTURED_OUTPUT_MODE_SETTING, ""
            )
            str_configured: str = str(raw_mode or "").strip().lower()
            if str_configured:
                str_mode = str_configured
        if str_mode not in (
            STRUCTURED_OUTPUT_MODE_JSON_SCHEMA,
            STRUCTURED_OUTPUT_MODE_JSON_OBJECT,
        ):
            raise AiProviderConfigurationError(
                f"{self.STRUCTURED_OUTPUT_MODE_SETTING} must be "
                f"'{STRUCTURED_OUTPUT_MODE_JSON_SCHEMA}' or "
                f"'{STRUCTURED_OUTPUT_MODE_JSON_OBJECT}'; got {str_mode!r}."
            )
        # Normal return with the validated mode.
        return str_mode

    def _resolve_context_window(self) -> int:
        """
        Returns the cataloged context window, then the environment
        override, then 0 (unknown; the context guard is skipped).
        """
        int_cataloged: int | None = self.DICT_CONTEXT_WINDOWS.get(
            self.completions_model
        )
        if int_cataloged is not None:
            # Early return with the cataloged window.
            return int_cataloged
        if self.CONTEXT_WINDOW_SETTING is None:
            # Early return: no override setting for this engine.
            return 0
        raw_window: object = self.env.get_setting(self.CONTEXT_WINDOW_SETTING, "")
        str_window: str = str(raw_window or "").strip()
        if not str_window:
            # Early return: no override configured.
            return 0
        try:
            int_window: int = int(str_window)
        except ValueError as exception:
            raise AiProviderConfigurationError(
                f"{self.CONTEXT_WINDOW_SETTING} must be a positive integer; "
                f"got {str_window!r}."
            ) from exception
        if int_window <= 0:
            raise AiProviderConfigurationError(
                f"{self.CONTEXT_WINDOW_SETTING} must be a positive integer; "
                f"got {str_window!r}."
            )
        # Normal return with the configured window.
        return int_window

    # ── Catalog and capabilities ──────────────────────────────────────────

    def _build_capabilities(self) -> AICompletionsCapabilitiesBase:
        """
        Resolves capabilities from the vendor settings for the configured
        model. Every compatible server is driven through the same Chat
        Completions surface, so streaming, tool use, structured output, and
        async are declared for all models.
        """
        list_data_types: list[SupportedDataType] = [SupportedDataType.TEXT]
        if self.completions_model in self.SET_IMAGE_INPUT_MODELS:
            list_data_types.append(SupportedDataType.IMAGE)
        # Normal return with vendor-driven capabilities.
        return AICompletionsCapabilitiesOpenAICompatible(
            context_window_length=self._resolve_context_window(),
            reasoning=self.completions_model in self.SET_REASONING_MODELS,
            supported_data_types=list_data_types,
            supports_streaming=True,
            supports_tool_use=True,
            supports_structured_output=True,
            supports_async=True,
            pricing=get_model_pricing(
                self.PROVIDER_REGISTRY_LABEL, self.completions_model
            ),
        )

    @property
    def list_model_names(self) -> list[str]:
        """
        Cataloged models; the generic engine reports the configured model
        because it cannot know what the server hosts.
        """
        if self.LIST_CATALOGED_MODELS:
            # Early return with the vendor catalog.
            return list(self.LIST_CATALOGED_MODELS)
        # Normal return with the one model this client is configured for.
        return [self.completions_model]

    @property
    def max_context_tokens(self) -> int:
        """Context window for the context guard; 0 skips the guard."""
        # Normal return with the resolved window.
        return self.capabilities.context_window_length

    def _build_user_message_content(
        self,
        prompt: str,
        other_params: AICompletionsPromptParamsBase | None,
    ) -> str | list[dict[str, Any]]:
        """
        Builds the user message, rejecting images for text-only models.

        Raises:
            ValueError: When images are attached for a model without image input.
        """
        if (
            other_params is not None
            and other_params.has_included_media
            and self.completions_model not in self.SET_IMAGE_INPUT_MODELS
        ):
            raise ValueError(
                f"{self.PROVIDER_DISPLAY_NAME} model {self.completions_model!r} "
                "does not accept image input."
            )
        # Normal return with the shared OpenAI-shaped content.
        return AiOpenAICompletions._build_user_message_content(
            self, prompt, other_params
        )

    # ── Structured output ───────────────────────────────────────────────────

    def _build_chat_structured_request_kwargs(
        self,
        *,
        response_schema: dict[str, Any],
        system_prompt: str | None,
        prompt: str | None,
        messages: list[dict[str, Any]] | None,
        max_response_tokens: int,
        dict_merge_options: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Builds the structured request. In json_object mode the server only
        guarantees valid JSON, so the schema moves into the system prompt.
        """
        dict_request_kwargs: dict[str, Any] = (
            AiOpenAICompletions._build_chat_structured_request_kwargs(
                self,
                response_schema=response_schema,
                system_prompt=system_prompt,
                prompt=prompt,
                messages=messages,
                max_response_tokens=max_response_tokens,
                dict_merge_options={},
            )
        )
        if self.structured_output_mode == STRUCTURED_OUTPUT_MODE_JSON_OBJECT:
            dict_request_kwargs["response_format"] = {"type": "json_object"}
            list_messages: list[dict[str, Any]] = list(dict_request_kwargs["messages"])
            dict_system: dict[str, Any] = dict(list_messages[0])
            dict_system["content"] = (
                f"{dict_system['content']}\n\n"
                "Respond with a single JSON object that conforms to this JSON "
                f"Schema:\n{json.dumps(response_schema, sort_keys=True)}"
            )
            list_messages[0] = dict_system
            dict_request_kwargs["messages"] = list_messages
        # provider_options merge last so callers can extend the raw request.
        dict_request_kwargs.update(dict_merge_options)
        # Normal return with the mode-appropriate structured request.
        return dict_request_kwargs

    def strict_schema_prompt(
        self,
        prompt: str,
        response_model: Type[AIStructuredPrompt],
        max_response_tokens: int = AIBaseCompletions.STRUCTURED_DEFAULT_MAX_RESPONSE_TOKENS,
        *,
        other_params: AICompletionsPromptParamsBase | None = None,
    ) -> AIStructuredPrompt:
        """
        Structured prompt through send_structured_output.

        The OpenAI engine uses the legacy `functions` parameter here, which
        compatible servers generally do not implement, so this routes through
        the response_format path instead.

        Raises:
            ValueError: When images are attached (not supported here) or the
                model refused.
            StructuredResponseTokenLimitError: When the output was truncated.
        """
        if other_params is not None and other_params.has_included_media:
            raise ValueError(
                "strict_schema_prompt does not accept image attachments on "
                f"{self.PROVIDER_DISPLAY_NAME}; use send_prompt instead."
            )
        str_system_prompt: str | None = (
            other_params.system_prompt if other_params is not None else None
        )
        structured_result: AIStructuredOutputResult = self.send_structured_output(
            prompt,
            response_model=response_model,
            system_prompt=str_system_prompt,
            max_response_tokens=max_response_tokens,
        )
        if structured_result.data is None:
            if structured_result.finish_reason is AIFinishReason.LENGTH:
                self._raise_structured_token_limit_error(
                    provider_name=self.PROVIDER_REGISTRY_LABEL,
                    model_name=self.completions_model,
                    max_response_tokens=max_response_tokens,
                    finish_reason=structured_result.finish_reason.value,
                    raw_output_text=structured_result.raw_text,
                )
            raise ValueError(
                f"{self.PROVIDER_DISPLAY_NAME} returned no structured output "
                f"(finish_reason={structured_result.finish_reason.value})."
            )
        # Normal return with the validated response model.
        return response_model.model_validate(structured_result.data)

    # ── OpenAI-only features switched off ───────────────────────────────────

    def _get_org_info_provider(self) -> AIProviderOrgInfoBase:
        """Compatible servers expose no organization identity."""
        # Normal return reporting no organization identity.
        return AIProviderOrgInfoBase()

    def _get_org_info_capability_provider(self) -> AIProviderOrgInfoCapability:
        """Declares that no organization identity is resolvable."""
        # Normal return with the empty capability declaration.
        return AIProviderOrgInfoCapability()

    # ── Observability labels ────────────────────────────────────────────────

    def _resolve_observability_provider_vendor(self) -> str:
        """Reports the vendor label instead of "openai"."""
        # Normal return with the registry label.
        return self.PROVIDER_REGISTRY_LABEL

    def _resolve_observability_provider_engine(self) -> str:
        """Reports the engine token instead of "openai"."""
        # Normal return with the engine token.
        return self.PROVIDER_ENGINE_TOKEN
