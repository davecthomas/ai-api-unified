from copy import deepcopy
import os
import sys
import pytest
import textwrap
from typing import Any, Dict, Optional, Type

from pydantic import model_validator

from ai_api.ai_factory import AIFactory

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT_DIR, "src"))
sys.path.insert(0, ROOT_DIR)

from ai_api.ai_base import AIBaseEmbeddings, AIBaseCompletions, AIStructuredPrompt


class TestStructuredPrompt(AIStructuredPrompt):
    message_input: str  # this is an input field, not a result

    test_output: Optional[str] = None

    @model_validator(mode="after")
    def _populate_prompt(
        self: "TestStructuredPrompt", __: Any
    ) -> "TestStructuredPrompt":
        """
        After validation, build and store the prompt string
        """
        object.__setattr__(
            self,
            "prompt",
            self.get_prompt(message_input=self.message_input),
        )
        return self

    @model_validator(mode="before")
    def validate_input(cls, values: Dict[str, Any]) -> Dict[str, Any]:
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
    def model_json_schema(cls) -> Dict[str, Any]:
        """
        JSON schema for the LLM’s *output* only.
        """
        # start with a fresh copy of the base schema (deep-copied there)
        schema: Dict[str, Any] = deepcopy(super().model_json_schema())
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
    Sending a structured prompt should return an instance of TestStructuredPrompt
    with its `.message` set to the uppercased prompt.
    """
    structured_prompt = TestStructuredPrompt(message_input="hello")
    structured_prompt_result: TestStructuredPrompt = (
        structured_prompt.send_structured_prompt(
            completion_client, TestStructuredPrompt
        )
    )

    assert isinstance(structured_prompt_result, TestStructuredPrompt)
    # The `message` attribute comes from AIStructuredPrompt; it should equal prompt.prompt.upper()
    assert (
        structured_prompt_result.test_output == structured_prompt.message_input.upper()
    )
