# ruff: noqa: E402

# tests/test_google_gemini_nonmock.py
from copy import deepcopy
import os
import socket
import textwrap
from typing import Any
from pydantic import model_validator
import pytest

pytest.importorskip("google.genai")

from ai_api_unified.ai_factory import AIFactory
from ai_api_unified.ai_base import (
    AIBase,
    AIBaseEmbeddings,
    AIStructuredPrompt,
    AIBaseCompletions,
)
from ai_api_unified.embeddings.ai_google_gemini_embeddings import (
    GoogleGeminiEmbeddings,
)
from ai_api_unified.util.utils import similarity_score

GOOGLE_GEMINI_HOSTNAME: str = "generativelanguage.googleapis.com"
TEST_GEMINI_COMPLETIONS_MODEL: str = "gemini-2.5-flash"
TEST_GEMINI_EMBEDDING_MODEL: str = "gemini-embedding-001"


def _skip_if_dns_unavailable(hostname: str) -> None:
    """Skip live tests quickly when the current environment cannot resolve provider DNS."""
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
def require_google_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run Gemini nonmock tests against the API-key auth path only."""
    api_key: str | None = os.environ.get("GOOGLE_GEMINI_API_KEY")
    if not api_key:
        pytest.skip("Skipping: GOOGLE_GEMINI_API_KEY not set")

    _skip_if_dns_unavailable(GOOGLE_GEMINI_HOSTNAME)
    monkeypatch.setenv("GOOGLE_GEMINI_API_KEY", api_key)
    monkeypatch.setenv("GOOGLE_API_KEY", api_key)
    monkeypatch.setenv("GOOGLE_AUTH_METHOD", "api_key")
    monkeypatch.setenv("COMPLETIONS_ENGINE", "google-gemini")
    monkeypatch.setenv("COMPLETIONS_MODEL_NAME", TEST_GEMINI_COMPLETIONS_MODEL)
    monkeypatch.setenv("EMBEDDING_ENGINE", "google-gemini")
    monkeypatch.setenv("EMBEDDING_MODEL_NAME", TEST_GEMINI_EMBEDDING_MODEL)
    monkeypatch.delenv("EMBEDDING_DIMENSIONS", raising=False)
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)


class GeminiStructuredPromptTest(AIStructuredPrompt):
    """Example structured prompt for testing."""

    name: str | None = None
    age: int | None = None
    city: str | None = None
    input_text: str | None = None

    @model_validator(mode="after")
    def _populate_prompt(
        self: "GeminiStructuredPromptTest", __: Any
    ) -> "GeminiStructuredPromptTest":
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


@pytest.fixture
def gemini_client() -> AIBaseCompletions:
    """
    Returns a completions client for testing.
    """
    ai_completions: AIBaseCompletions = AIFactory.get_ai_completions_client(
        model_name=TEST_GEMINI_COMPLETIONS_MODEL,
        completions_engine="google-gemini",
    )
    return ai_completions


@pytest.fixture
def gemini_embeddings_client() -> AIBaseEmbeddings:
    """
    Returns an embeddings client for testing.
    """
    emb_client: AIBaseEmbeddings = AIFactory.get_ai_embedding_client(
        model_name=TEST_GEMINI_EMBEDDING_MODEL,
        embedding_engine="google-gemini",
    )
    return emb_client


@pytest.mark.skipif(
    skip_in_codex,
    reason="Skip non-mocked tests in Codex environment",
)
class TestNonMockedGoogleGeminiModules:
    """Test that Google Gemini modules can be imported when dependencies are not mocked."""

    def test_gemini_embeddings_import(self):
        """Test that Google Gemini embeddings module can be imported."""
        from ai_api_unified.embeddings import ai_google_gemini_embeddings

        assert hasattr(ai_google_gemini_embeddings, "GoogleGeminiEmbeddings")

    def test_gemini_completions_import(self):
        """Test that Google Gemini completions module can be imported."""
        from ai_api_unified.completions import ai_google_gemini_completions

        assert hasattr(ai_google_gemini_completions, "GoogleGeminiCompletions")

    def test_gemini_completions_instance(
        self, gemini_client: AIBaseCompletions
    ) -> None:
        from ai_api_unified.ai_base import AIBaseCompletions

        ai_completions: AIBaseCompletions = gemini_client
        assert isinstance(ai_completions, AIBaseCompletions)
        assert ai_completions.model_name == TEST_GEMINI_COMPLETIONS_MODEL

    def test_gemini_completions_send_prompt(
        self, gemini_client: AIBaseCompletions
    ) -> None:
        input_message = "What is the capital of France? Respond with one word only."
        try:
            response = gemini_client.send_prompt(input_message)
        except RuntimeError as exception:
            _skip_if_google_quota_exhausted(exception)
            raise
        print(f"Response: {response}")
        assert response.lower() == "paris"
        assert isinstance(response, str)

    def test_gemini_structured_prompt(self, gemini_client: AIBaseCompletions) -> None:

        structured_prompt = GeminiStructuredPromptTest(
            input_text="My name is Alice, I am 30 years old, and I live in Paris."
        )
        try:
            structured_prompt_result: GeminiStructuredPromptTest = (
                structured_prompt.send_structured_prompt(
                    gemini_client, GeminiStructuredPromptTest
                )
            )
        except RuntimeError as exception:
            _skip_if_google_quota_exhausted(exception)
            raise
        if structured_prompt_result is None:
            pytest.skip(
                "Skipping Google Gemini structured nonmock test because the provider did not return a structured result."
            )

        assert isinstance(structured_prompt_result, GeminiStructuredPromptTest)
        assert structured_prompt_result.name.lower() == "alice"
        assert structured_prompt_result.age == 30
        assert structured_prompt_result.city.lower() == "paris"

    def test_gemini_embeddings_instance(
        self, gemini_embeddings_client: AIBaseEmbeddings
    ) -> None:
        """Ensure GoogleGeminiEmbeddings.generate_embeddings returns a list of dicts with the right shape."""
        texts = ["Hello world", "AI embeddings test"]
        results = gemini_embeddings_client.generate_embeddings_batch(texts)

        # basic shape checks
        assert isinstance(results, list)
        assert len(results) == len(texts)

        for res in results:
            assert isinstance(res, dict)
            assert "embedding" in res, "each result must have an 'embedding' key"
            embedding = res["embedding"]
            assert isinstance(embedding, list)
            assert len(embedding) == gemini_embeddings_client.dimensions

    def test_gemini_embeddings_empty_list_raises(
        self, gemini_embeddings_client: AIBase
    ) -> None:
        """Embedding an empty list must raise ValueError."""
        with pytest.raises(ValueError):
            gemini_embeddings_client.generate_embeddings([])

    def test_gemini_embeddings_blank_list_raises(
        self, gemini_embeddings_client: AIBaseEmbeddings
    ) -> None:
        """Embedding a list of blank/whitespace strings must raise ValueError."""
        with pytest.raises(ValueError):
            gemini_embeddings_client.generate_embeddings_batch(["", "   "])

    def test_embedding_dimensions_override(self):
        # Pick a value less than the model’s max (3072)
        override_dim = 128
        assert (
            override_dim < GoogleGeminiEmbeddings.DEFAULT_EMBEDDING_DIMENSIONS
        ), f"Override ({override_dim}) must be < default ({GoogleGeminiEmbeddings.DEFAULT_EMBEDDING_DIMENSIONS})"

        # Construct the embeddings client with the override
        gemini_embeddings_client = GoogleGeminiEmbeddings(dimensions=override_dim)

        # Generate an embedding for a simple string
        result = gemini_embeddings_client.generate_embeddings("Hello, Vertex AI!")

        # Assert that the API returned the truncated vector
        assert isinstance(result["embedding"], list), "Embedding must be a list"
        assert (
            result["dimensions"] == override_dim
        ), f"Expected dimensions {override_dim}, got {result['dimensions']}"
        assert (
            len(result["embedding"]) == override_dim
        ), f"Expected vector length {override_dim}, got {len(result['embedding'])}"

    def test_similarity_score_for_semantically_similar_phrases(
        self, gemini_embeddings_client: AIBaseEmbeddings
    ) -> None:
        pytest.importorskip("numpy")
        # Two near-equivalent phrases
        phrase_a = "The cat sat on the mat."
        phrase_b = "A cat is sitting on a mat."

        # Generate embeddings
        emb_a = gemini_embeddings_client.generate_embeddings(phrase_a)
        emb_b = gemini_embeddings_client.generate_embeddings(phrase_b)

        # Compute cosine similarity
        score = similarity_score(emb_a, emb_b)

        # Assertions
        assert isinstance(score, float), "Similarity score should be a float"
        assert 0.8 < score <= 1.0, f"Expected high similarity > 0.8, got {score}"
