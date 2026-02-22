# test_google_gemini.py
"""
Tests for Google Gemini embeddings and completions.

Note: These tests verify the structure and imports without requiring
actual Google credentials.
"""

import pytest
from unittest.mock import Mock, patch
import sys

# Only mock the specific google.genai module to avoid conflicts
mock_genai = Mock()
mock_genai_types = Mock()
mock_genai_errors = Mock()

sys.modules["google.genai"] = mock_genai
sys.modules["google.genai.types"] = mock_genai_types
sys.modules["google.genai.errors"] = mock_genai_errors

# Don't mock google.api_core globally as it's used by other modules
# Instead, we'll mock it specifically in our Google Gemini modules

from ai_api_unified.ai_base import (
    AIStructuredPrompt,
)


class ExampleStructuredPrompt(AIStructuredPrompt):
    """Example structured prompt for testing."""

    name: str
    age: int
    city: str

    @staticmethod
    def get_prompt() -> str:
        return "Generate a person's information"


class TestGoogleGeminiIntegration:
    """Test Google Gemini integration through the factory."""

    def test_basic_integration_structure(self):
        """Test that Google Gemini integration is properly structured."""
        # This just tests that our integration doesn't crash during import
        # More specific functionality is tested in TestGoogleGeminiModules
        from ai_api_unified import ai_factory

        # Test that the factory module loads correctly
        assert hasattr(ai_factory, "AIFactory")
        assert hasattr(ai_factory, "GOOGLE_GEMINI_AVAILABLE")

        # When google-generativeai is not installed, this should be False
        # When it is installed, this could be True
        assert isinstance(ai_factory.GOOGLE_GEMINI_AVAILABLE, bool)


class TestGoogleGeminiModules:
    """Test that Google Gemini modules can be imported when dependencies are mocked."""

    def test_gemini_embeddings_module_structure(self):
        """Test that Google Gemini embeddings module has expected structure."""
        with patch(
            "ai_api_unified.embeddings.ai_google_gemini_embeddings.genai"
        ) as mock_genai:
            with patch("ai_api_unified.embeddings.ai_google_gemini_embeddings.gerr"):
                with patch("ai_api_unified.util.env_settings.EnvSettings") as mock_env:
                    mock_env_instance = Mock()
                    mock_env_instance.get_setting.side_effect = lambda key, default: {
                        "EMBEDDING_MODEL_NAME": "gemini-embedding-001",
                        "EMBEDDING_DIMENSIONS": "768",
                    }.get(key, default)
                    mock_env.return_value = mock_env_instance

                    mock_client = Mock()
                    mock_models = Mock()
                    mock_client.models = mock_models
                    mock_models.list.return_value = [
                        {"name": "models/gemini-embedding-001"}
                    ]
                    mock_genai.Client.return_value = mock_client

                    from ai_api_unified.embeddings.ai_google_gemini_embeddings import (
                        GoogleGeminiEmbeddings,
                    )

                    client = GoogleGeminiEmbeddings(
                        model="gemini-embedding-001", dimensions=768
                    )

                    assert client.model_name == "gemini-embedding-001"
                    assert isinstance(client.list_model_names, list)
                    assert "gemini-embedding-001" in client.list_model_names

                    assert hasattr(client, "generate_embeddings")
                    assert hasattr(client, "generate_embeddings_batch")

    def test_gemini_completions_module_structure(self):
        """Test that Google Gemini completions module has expected structure."""
        with patch(
            "ai_api_unified.completions.ai_google_gemini_completions.genai"
        ) as mock_genai:
            with patch("ai_api_unified.completions.ai_google_gemini_completions.gerr"):
                with patch("ai_api_unified.util.env_settings.EnvSettings") as mock_env:
                    mock_env_instance = Mock()
                    mock_env_instance.get_setting.side_effect = lambda key, default: {
                        "COMPLETIONS_MODEL_NAME": "gemini-2.0-flash-lite"
                    }.get(key, default)
                    mock_env.return_value = mock_env_instance

                    mock_client = Mock()
                    mock_models = Mock()
                    mock_client.models = mock_models
                    mock_models.get.return_value = None
                    mock_genai.Client.return_value = mock_client
                    mock_genai.types = Mock()

                    from ai_api_unified.completions.ai_google_gemini_completions import (
                        GoogleGeminiCompletions,
                    )

                    client = GoogleGeminiCompletions(model="gemini-2.0-flash-lite")

                    assert client.model_name == "gemini-2.0-flash-lite"
                    assert isinstance(client.list_model_names, list)
                    assert "gemini-2.0-flash-lite" in client.list_model_names
                    assert client.max_context_tokens > 0
                    assert client.price_per_1k_tokens > 0

                    assert hasattr(client, "capabilities")
                    capabilities = client.capabilities
                    assert capabilities.context_window_length > 0
                    assert isinstance(capabilities.reasoning, bool)
                    assert len(capabilities.supported_data_types) > 0

                    assert hasattr(client, "send_prompt")
                    assert hasattr(client, "strict_schema_prompt")

    def test_gemini_auth_method_api_key(self):
        """Test that get_client properly uses GOOGLE_AUTH_METHOD=api_key."""
        with patch("ai_api_unified.ai_google_base.genai") as mock_genai:
            with patch("ai_api_unified.util.env_settings.EnvSettings") as mock_env:
                mock_env_instance = Mock()
                mock_env_instance.get_setting.side_effect = lambda key, default: {
                    "GOOGLE_AUTH_METHOD": "api_key",
                    "GOOGLE_GEMINI_API_KEY": "fake-test-key-123",
                }.get(key, default)
                mock_env.return_value = mock_env_instance

                mock_client = Mock()
                mock_models = Mock()
                mock_client.models = mock_models
                mock_models.get.return_value = Mock(name="models/test")
                mock_genai.Client.return_value = mock_client

                from ai_api_unified.ai_google_base import AIGoogleBase

                base = AIGoogleBase()
                client = base.get_client("test-model")
                
                assert client == mock_client
                mock_genai.Client.assert_called_once_with(api_key="fake-test-key-123")

    def test_gemini_embeddings_basic_functionality(self):
        """Test basic Google Gemini embeddings functionality with mocking."""
        with patch(
            "ai_api_unified.embeddings.ai_google_gemini_embeddings.genai"
        ) as mock_genai:
            with patch("ai_api_unified.embeddings.ai_google_gemini_embeddings.gerr"):
                with patch("ai_api_unified.util.env_settings.EnvSettings") as mock_env:
                    mock_env_instance = Mock()
                    mock_env_instance.get_setting.side_effect = lambda key, default: {
                        "EMBEDDING_MODEL_NAME": "gemini-embedding-001",
                        "EMBEDDING_DIMENSIONS": "768",
                    }.get(key, default)
                    mock_env.return_value = mock_env_instance

                    mock_client = Mock()
                    mock_models = Mock()
                    mock_client.models = mock_models
                    mock_models.list.return_value = [
                        {"name": "models/gemini-embedding-001"}
                    ]
                    mock_models.embed_content.return_value = Mock(
                        embeddings=[Mock(values=[0.1, 0.2, 0.3] * 256)]
                    )
                    mock_genai.Client.return_value = mock_client

                    from ai_api_unified.embeddings.ai_google_gemini_embeddings import (
                        GoogleGeminiEmbeddings,
                    )

                    client = GoogleGeminiEmbeddings()

                    with patch.object(
                        client,
                        "_retry_with_exponential_backoff",
                        side_effect=lambda func: func(),
                    ):
                        result = client.generate_embeddings("test text")

                        assert "embedding" in result
                        assert "model" in result
                        assert result["text"] == "test text"

    def test_gemini_completions_basic_functionality(self):
        """Test basic Google Gemini completions functionality with mocking."""
        with patch(
            "ai_api_unified.completions.ai_google_gemini_completions.genai"
        ) as mock_genai:
            with patch("ai_api_unified.completions.ai_google_gemini_completions.gerr"):
                with patch("ai_api_unified.util.env_settings.EnvSettings") as mock_env:
                    mock_env_instance = Mock()
                    mock_env_instance.get_setting.side_effect = lambda key, default: {
                        "COMPLETIONS_MODEL_NAME": "gemini-2.0-flash-lite"
                    }.get(key, default)
                    mock_env.return_value = mock_env_instance

                    mock_client = Mock()
                    mock_models = Mock()
                    mock_client.models = mock_models
                    mock_response = Mock()
                    mock_response.text = "This is a test response"
                    mock_models.generate_content.return_value = mock_response
                    mock_models.get.return_value = None
                    mock_genai.Client.return_value = mock_client
                    mock_genai.types = Mock()
                    mock_genai.types.GenerateContentConfig = Mock()

                    from ai_api_unified.completions.ai_google_gemini_completions import (
                        GoogleGeminiCompletions,
                    )

                    client = GoogleGeminiCompletions()

                    with patch.object(
                        client,
                        "_retry_with_exponential_backoff",
                        side_effect=lambda func: func(),
                    ):
                        result = client.send_prompt("What is the capital of France?")
                        assert result == "This is a test response"

    def test_gemini_completions_with_prompt_params(self):
        """Test Google Gemini completions with custom prompt parameters."""
        with patch(
            "ai_api_unified.completions.ai_google_gemini_completions.genai"
        ) as mock_genai:
            with patch("ai_api_unified.completions.ai_google_gemini_completions.gerr"):
                with patch("ai_api_unified.util.env_settings.EnvSettings") as mock_env:
                    mock_env_instance = Mock()
                    mock_env_instance.get_setting.side_effect = lambda key, default: {
                        "COMPLETIONS_MODEL_NAME": "gemini-2.5-flash-lite"
                    }.get(key, default)
                    mock_env.return_value = mock_env_instance

                    mock_client = Mock()
                    mock_models = Mock()
                    mock_client.models = mock_models
                    mock_response = Mock()
                    mock_response.text = "This is a test response with custom params"
                    mock_models.generate_content.return_value = mock_response
                    mock_models.get.return_value = None
                    mock_genai.Client.return_value = mock_client
                    mock_genai.types = Mock()
                    mock_genai.types.GenerateContentConfig = Mock()

                    from ai_api_unified.completions.ai_google_gemini_completions import (
                        GoogleGeminiCompletions,
                    )
                    from ai_api_unified.completions.ai_google_gemini_capabilities import (
                        AICompletionsPromptParamsGoogle,
                    )

                    client = GoogleGeminiCompletions()

                    custom_params = AICompletionsPromptParamsGoogle(
                        temperature=0.7,
                        top_p=0.9,
                        top_k=2,
                        max_output_tokens=1024,
                    )

                    with patch.object(
                        client,
                        "_retry_with_exponential_backoff",
                        side_effect=lambda func: func(),
                    ):
                        result = client.send_prompt(
                            "What is the capital of France?", other_params=custom_params
                        )
                        assert result == "This is a test response with custom params"

                        mock_genai.types.GenerateContentConfig.assert_called_with(
                            temperature=0.7,
                            top_p=0.9,
                            top_k=2,
                            max_output_tokens=1024,
                        )

    def test_gemini_completions_structured_prompt(self):
        """Test structured prompting for Google Gemini completions."""
        with patch(
            "ai_api_unified.completions.ai_google_gemini_completions.genai"
        ) as mock_genai:
            with patch("ai_api_unified.completions.ai_google_gemini_completions.gerr"):
                with patch("ai_api_unified.util.env_settings.EnvSettings") as mock_env:
                    mock_env_instance = Mock()
                    mock_env_instance.get_setting.side_effect = lambda key, default: {
                        "COMPLETIONS_MODEL_NAME": "gemini-2.0-flash-lite"
                    }.get(key, default)
                    mock_env.return_value = mock_env_instance

                    mock_client = Mock()
                    mock_models = Mock()
                    mock_client.models = mock_models
                    mock_response = Mock()
                    mock_response.text = '{"name":"Alice","age":30,"city":"Paris"}'
                    mock_models.generate_content.return_value = mock_response
                    mock_models.get.return_value = None
                    mock_genai.Client.return_value = mock_client
                    mock_genai.types = Mock()
                    mock_genai.types.GenerateContentConfig = Mock()

                    from ai_api_unified.completions.ai_google_gemini_completions import (
                        GoogleGeminiCompletions,
                    )

                    client = GoogleGeminiCompletions()

                    with patch.object(
                        client,
                        "_retry_with_exponential_backoff",
                        side_effect=lambda func: func(),
                    ):
                        result = client.strict_schema_prompt(
                            "Describe a person", ExampleStructuredPrompt
                        )
                        assert isinstance(result, ExampleStructuredPrompt)
                        assert result.name == "Alice"

                        mock_genai.types.GenerateContentConfig.assert_called_with(
                            temperature=0.1,
                            top_p=0.8,
                            top_k=1,
                            max_output_tokens=512,
                            response_mime_type="application/json",
                            response_schema=ExampleStructuredPrompt,
                        )


if __name__ == "__main__":
    pytest.main([__file__])
