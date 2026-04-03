# tests/test_model_switch_nonmock.py
from copy import deepcopy
import os
import socket
import textwrap
from typing import Any
from pydantic import model_validator
import pytest
from ai_api_unified.ai_factory import AIFactory
from ai_api_unified.ai_base import (
    AIBaseEmbeddings,
    AIStructuredPrompt,
    AIBaseCompletions,
)

from ai_api_unified.util.utils import similarity_score

GOOGLE_GEMINI_HOSTNAME: str = "generativelanguage.googleapis.com"


def _skip_if_dns_unavailable(hostname: str) -> None:
    """Skip live provider tests quickly when DNS/network prerequisites are unavailable."""
    try:
        socket.getaddrinfo(hostname, 443)
    except OSError as exception:
        pytest.skip(f"Skipping: DNS unavailable for {hostname}: {exception}")


def _skip_if_google_quota_exhausted(exception: Exception) -> None:
    """Skip live Gemini tests when the current API key has exhausted its quota."""
    message: str = str(exception)
    if "RESOURCE_EXHAUSTED" in message or "quota exceeded" in message.lower():
        pytest.skip(
            "Skipping Google Gemini nonmock test because the current API key quota is exhausted."
        )


@pytest.fixture(autouse=True)
def require_ai_api_credentials(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Require only the credentials needed for the selected live provider."""
    aiprovider: str = request.config.getoption("aiprovider")
    if aiprovider in {"google-gemini", "google"}:
        pytest.importorskip("google.genai")
        if "GOOGLE_GEMINI_API_KEY" not in os.environ:
            pytest.skip("Skipping: GOOGLE_GEMINI_API_KEY not set")
        _skip_if_dns_unavailable(GOOGLE_GEMINI_HOSTNAME)
        monkeypatch.setenv("GOOGLE_AUTH_METHOD", "api_key")
        monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
        return

    if aiprovider == "openai" and "OPENAI_API_KEY" not in os.environ:
        pytest.skip("Skipping: OPENAI_API_KEY not set")


class GenericStructuredPromptTest(AIStructuredPrompt):
    """Example structured prompt for testing."""

    name: str | None = None
    age: int | None = None
    city: str | None = None
    input_text: str | None = None

    @model_validator(mode="after")
    def _populate_prompt(
        self: "GenericStructuredPromptTest", __: Any
    ) -> "GenericStructuredPromptTest":
        object.__setattr__(
            self,
            "prompt",
            self.get_prompt(input_text=self.input_text),
        )
        return self

    @staticmethod
    def get_prompt(input_text: str) -> str:
        prompt: str = textwrap.dedent(
            f"""
            Extract the name, age, and city from the following text:
            {input_text}
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
        # add the output field
        schema["properties"]["name"] = {"type": "string"}
        schema["properties"]["age"] = {"type": "integer"}
        schema["properties"]["city"] = {"type": "string"}
        schema.setdefault("required", [])
        schema["required"].append("name")
        schema["required"].append("age")
        schema["required"].append("city")
        return schema


_truthy_codex_env = os.environ.get("RUNNING_IN_CODEX", "0").lower()
skip_in_codex = _truthy_codex_env in ("1", "true", "yes")


@pytest.mark.skipif(
    skip_in_codex,
    reason="Skip non-mocked tests in Codex environment",
)
class TestNonMockedGenericAiModules:
    """Test that Google Gemini modules can be imported when dependencies are not mocked."""

    def test_completions_send_prompt(self, aiprovider: str, llmmodel: str) -> None:
        ai_completions: AIBaseCompletions = AIFactory.get_ai_completions_client(
            completions_engine=aiprovider, model_name=llmmodel
        )
        input_message = "What is the capital of France? Respond with one word only and no punctuation."
        try:
            response = ai_completions.send_prompt(input_message)
        except RuntimeError as exception:
            if aiprovider in {"google-gemini", "google"}:
                _skip_if_google_quota_exhausted(exception)
            raise
        print(f"Response: {response}")
        assert isinstance(response, str)
        assert "paris" in response.lower()

    def test_structured_prompt(self, aiprovider: str, llmmodel: str) -> None:
        ai_completions: AIBaseCompletions = AIFactory.get_ai_completions_client(
            completions_engine=aiprovider, model_name=llmmodel
        )
        structured_prompt = GenericStructuredPromptTest(
            input_text="My name is Alice, I am 30 years old, and I live in Paris."
        )
        try:
            structured_prompt_result: GenericStructuredPromptTest = (
                structured_prompt.send_structured_prompt(
                    ai_completions, GenericStructuredPromptTest
                )
            )
        except RuntimeError as exception:
            if aiprovider in {"google-gemini", "google"}:
                _skip_if_google_quota_exhausted(exception)
            raise
        if structured_prompt_result is None and aiprovider in {
            "google-gemini",
            "google",
        }:
            pytest.skip(
                "Skipping Google Gemini structured nonmock test because the provider did not return a structured result."
            )

        assert isinstance(structured_prompt_result, GenericStructuredPromptTest)
        assert structured_prompt_result.name.lower() == "alice"
        assert structured_prompt_result.age == 30
        assert structured_prompt_result.city.lower() == "paris"

    def test_embeddings_instance(self, aiprovider: str, embedmodel: str) -> None:
        """Ensure generate_embeddings returns a list of dicts with the right shape."""
        texts = ["Hello world", "AI embeddings test"]
        embeddings_client: AIBaseEmbeddings = AIFactory.get_ai_embedding_client(
            embedding_engine=aiprovider, model_name=embedmodel
        )
        results = embeddings_client.generate_embeddings_batch(texts)

        # basic shape checks
        assert isinstance(results, list)
        assert len(results) == len(texts)

        for res in results:
            assert isinstance(res, dict)
            assert "embedding" in res, "each result must have an 'embedding' key"
            embedding = res["embedding"]
            assert isinstance(embedding, list)
            assert len(embedding) == embeddings_client.dimensions

    def test_embeddings_empty_list_raises(
        self, aiprovider: str, embedmodel: str
    ) -> None:
        """Embedding an empty list must raise ValueError."""
        embeddings_client: AIBaseEmbeddings = AIFactory.get_ai_embedding_client(
            embedding_engine=aiprovider, model_name=embedmodel
        )
        with pytest.raises(ValueError):
            embeddings_client.generate_embeddings([])

    def test_embeddings_blank_list_raises(
        self, aiprovider: str, embedmodel: str
    ) -> None:
        """Embedding a list of blank/whitespace strings must raise ValueError."""
        embeddings_client: AIBaseEmbeddings = AIFactory.get_ai_embedding_client(
            embedding_engine=aiprovider, model_name=embedmodel
        )
        with pytest.raises(ValueError):
            embeddings_client.generate_embeddings_batch(["", "   "])

    def test_similarity_score_for_semantically_similar_phrases(
        self, aiprovider: str, embedmodel: str
    ) -> None:
        pytest.importorskip("numpy")
        embeddings_client: AIBaseEmbeddings = AIFactory.get_ai_embedding_client(
            embedding_engine=aiprovider, model_name=embedmodel
        )

        # Two near-equivalent phrases
        phrase_a = "The cat sat on the mat."
        phrase_b = "A cat is sitting on a mat."

        # Generate embeddings
        emb_a = embeddings_client.generate_embeddings(phrase_a)
        emb_b = embeddings_client.generate_embeddings(phrase_b)

        # Compute cosine similarity
        score = similarity_score(emb_a, emb_b)

        # Assertions
        assert isinstance(score, float), "Similarity score should be a float"
        assert 0.8 < score <= 1.0, f"Expected high similarity > 0.8, got {score}"
