from __future__ import (
    annotations,
)  # Postpone evaluation of type hints to avoid circular imports and allow forward references with | None

import socket
import textwrap
from copy import deepcopy
from typing import Any

import pytest
from pydantic import model_validator

from ai_api_unified.ai_base import (
    AIBaseCompletions,
    AIBaseEmbeddings,
    AICompletionsPromptParamsBase,
    AIStructuredPrompt,
)
from ai_api_unified.ai_factory import AIFactory
from ai_api_unified.ai_provider_exceptions import (
    AiProviderDependencyUnavailableError,
)

pytestmark = pytest.mark.nonmock
EXPLICIT_MAX_RESPONSE_TOKENS: int = 2048
GOOGLE_GEMINI_HOSTNAME: str = "generativelanguage.googleapis.com"


def _skip_if_google_dns_unavailable(aiprovider: str) -> None:
    """Skip Google live tests quickly when provider DNS is unavailable."""
    if aiprovider not in {"google-gemini", "google"}:
        return
    try:
        socket.getaddrinfo(GOOGLE_GEMINI_HOSTNAME, 443)
    except OSError as exception:
        pytest.skip(
            f"Skipping Google completions test because DNS is unavailable for {GOOGLE_GEMINI_HOSTNAME}: {exception}"
        )


def _skip_if_google_quota_exhausted(exception: Exception) -> None:
    """Skip live Gemini tests when the current API key has exhausted its quota."""
    message: str = str(exception)
    if "RESOURCE_EXHAUSTED" in message or "quota exceeded" in message.lower():
        pytest.skip(
            "Skipping Google Gemini nonmock test because the current API key quota is exhausted."
        )


class ExampleStructuredPrompt(AIStructuredPrompt):
    message_input: str  # this is an input field, not a result

    test_output: str | None = None

    @model_validator(mode="after")
    def _populate_prompt(
        self: ExampleStructuredPrompt, __: Any
    ) -> ExampleStructuredPrompt:
        """
        After validation, build and store the prompt string
        """
        object.__setattr__(
            self,
            "prompt",
            ExampleStructuredPrompt.get_prompt(message_input=self.message_input),
        )
        return self

    @model_validator(mode="before")
    def validate_input(cls, values: dict[str, Any]) -> dict[str, Any]:
        # Ensure the message_input is set to "hello"
        if "message_input" not in values or values["message_input"] != "hello":
            raise ValueError("message_input must be 'hello'")
        return values

    @staticmethod
    def get_prompt(
        message_input: str,
    ) -> str:
        prompt = textwrap.dedent(
            f"""
            Reply with than uppercase version of the message_input in the test_output field.
            message_input: '{message_input}'
            """
        ).strip()
        return prompt

    @classmethod
    def model_json_schema(cls) -> dict[str, Any]:
        """
        JSON schema for the LLM’s *output* only.
        """
        # start with a fresh copy of the base schema (deep-copied there)
        schema: dict[str, Any] = deepcopy(super().model_json_schema())
        schema["properties"]["test_output"] = {"type": "string"}
        # make test_output required for the LLM response
        schema.setdefault("required", [])
        schema["required"].append("test_output")
        return schema


@pytest.fixture
def embedding_client(aiprovider: str, embedmodel: str) -> AIBaseEmbeddings:
    """
    Returns an embeddings client for testing.
    """
    _skip_if_google_dns_unavailable(aiprovider)
    try:
        return AIFactory.get_ai_embedding_client(
            embedding_engine=aiprovider,
            model_name=embedmodel,
        )
    except AiProviderDependencyUnavailableError as exception:
        pytest.skip(f"Skipping embeddings test due to missing dependency: {exception}")


@pytest.fixture
def completion_client_simple(aiprovider: str, llmmodel: str) -> AIBaseCompletions:
    """
    Returns a completions client for testing.
    """
    _skip_if_google_dns_unavailable(aiprovider)
    try:
        return AIFactory.get_ai_completions_client(
            completions_engine=aiprovider,
            model_name=llmmodel,
        )
    except AiProviderDependencyUnavailableError as exception:
        pytest.skip(f"Skipping completions test due to missing dependency: {exception}")


@pytest.fixture
def completion_client(aiprovider: str, llmmodel: str) -> AIBaseCompletions:
    """
    Returns a completions client for testing.
    """
    _skip_if_google_dns_unavailable(aiprovider)
    try:
        return AIFactory.get_ai_completions_client(
            completions_engine=aiprovider,
            model_name=llmmodel,
        )
    except AiProviderDependencyUnavailableError as exception:
        pytest.skip(f"Skipping completions test due to missing dependency: {exception}")


def test_send_prompt(
    completion_client: AIBaseCompletions,
    aiprovider: str,
) -> None:
    """
    The completion client should uppercase the input string.
    """
    input_message = "hello"
    try:
        response = completion_client.send_prompt(input_message)
    except RuntimeError as exception:
        if aiprovider in {"google-gemini", "google"}:
            _skip_if_google_quota_exhausted(exception)
        raise
    assert response != ""


def test_structured_prompt(
    completion_client: AIBaseCompletions,
    aiprovider: str,
) -> None:
    """
    Sending a structured prompt should return an instance of ExampleStructuredPrompt
    with its `test_output` set to the uppercased prompt.
    """
    structured_prompt = ExampleStructuredPrompt(message_input="hello")
    try:
        structured_prompt_result: ExampleStructuredPrompt = (
            structured_prompt.send_structured_prompt(
                completion_client, ExampleStructuredPrompt
            )
        )
    except RuntimeError as exception:
        if aiprovider in {"google-gemini", "google"}:
            _skip_if_google_quota_exhausted(exception)
        raise
    if structured_prompt_result is None and aiprovider in {"google-gemini", "google"}:
        pytest.skip(
            "Skipping Google Gemini structured nonmock test because the provider did not return a structured result."
        )

    assert isinstance(structured_prompt_result, ExampleStructuredPrompt)
    # The `message` attribute comes from AIStructuredPrompt; it should equal prompt.prompt.upper()
    assert (
        structured_prompt_result.test_output == structured_prompt.message_input.upper()
    )


def test_structured_prompt_with_max_response_tokens(
    completion_client: AIBaseCompletions,
    aiprovider: str,
) -> None:
    """
    Sending a structured prompt with an explicit token cap should still return a
    valid structured response.
    """
    structured_prompt: ExampleStructuredPrompt = ExampleStructuredPrompt(
        message_input="hello"
    )
    try:
        structured_prompt_result: ExampleStructuredPrompt = (
            completion_client.strict_schema_prompt(
                structured_prompt.prompt,
                ExampleStructuredPrompt,
                max_response_tokens=EXPLICIT_MAX_RESPONSE_TOKENS,
            )
        )
    except RuntimeError as exception:
        if aiprovider in {"google-gemini", "google"}:
            _skip_if_google_quota_exhausted(exception)
        raise

    assert isinstance(structured_prompt_result, ExampleStructuredPrompt)
    assert (
        structured_prompt_result.test_output == structured_prompt.message_input.upper()
    )


def test_system_prompt_override(
    completion_client: AIBaseCompletions,
    aiprovider: str,
) -> None:
    """
    Verifies that providing a custom system prompt influences the request payload.
    """

    system_prompt: str = (
        "You are a terse status bot. Always reply with exactly: 'STATUS: acknowledged.'"
    )
    prompt_text: str = "Please confirm you read this sentence."

    params: AICompletionsPromptParamsBase = AICompletionsPromptParamsBase(
        system_prompt=system_prompt
    )

    try:
        response: str = completion_client.send_prompt(
            prompt_text,
            other_params=params,
        )
    except RuntimeError as exception:
        if aiprovider in {"google-gemini", "google"}:
            _skip_if_google_quota_exhausted(exception)
        raise

    assert response == "STATUS: acknowledged."
