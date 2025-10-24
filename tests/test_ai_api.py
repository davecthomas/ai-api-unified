from __future__ import (
    annotations,
)  # Postpone evaluation of type hints to avoid circular imports and allow forward references with | None

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
def embedding_client() -> AIBaseEmbeddings:
    """
    Returns an embeddings client for testing.
    """
    return AIFactory.get_ai_embedding_client()


@pytest.fixture
def completion_client_simple() -> AIBaseCompletions:
    """
    Returns a completions client for testing.
    """
    return AIFactory.get_ai_completions_client()


@pytest.fixture
def completion_client() -> AIBaseCompletions:
    """
    Returns a completions client for testing.
    """
    return AIFactory.get_ai_completions_client()


def test_send_prompt(completion_client: AIBaseCompletions) -> None:
    """
    The completion client should uppercase the input string.
    """
    input_message = "hello"
    response = completion_client.send_prompt(input_message)
    assert response != ""


def test_structured_prompt(completion_client: AIBaseCompletions) -> None:
    """
    Sending a structured prompt should return an instance of ExampleStructuredPrompt
    with its `test_output` set to the uppercased prompt.
    """
    structured_prompt = ExampleStructuredPrompt(message_input="hello")
    structured_prompt_result: ExampleStructuredPrompt = (
        structured_prompt.send_structured_prompt(
            completion_client, ExampleStructuredPrompt
        )
    )

    assert isinstance(structured_prompt_result, ExampleStructuredPrompt)
    # The `message` attribute comes from AIStructuredPrompt; it should equal prompt.prompt.upper()
    assert (
        structured_prompt_result.test_output == structured_prompt.message_input.upper()
    )


def test_system_prompt_override(completion_client: AIBaseCompletions) -> None:
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

    response: str = completion_client.send_prompt(
        prompt_text,
        other_params=params,
    )

    assert response == "STATUS: acknowledged."
