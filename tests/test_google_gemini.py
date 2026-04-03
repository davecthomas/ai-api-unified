# test_google_gemini.py
"""
Tests for Google Gemini embeddings and completions.

Note: These tests verify the structure and imports without requiring
actual Google credentials.
"""

import pytest
from unittest.mock import Mock, patch
import importlib
import sys
import types

MOCKED_GOOGLE_MODULE_NAMES: tuple[str, ...] = (
    "google",
    "google.genai",
    "google.genai.types",
    "google.genai.errors",
    "google.genai.pagers",
    "google.api_core",
    "google.api_core.exceptions",
    "google.api_core.client_options",
    "google.auth",
    "google.auth.exceptions",
    "google.cloud",
    "google.cloud.speech_v1p1beta1",
    "google.cloud.texttospeech",
)
GOOGLE_PROVIDER_MODULE_NAMES: tuple[str, ...] = (
    "ai_api_unified.ai_google_base",
    "ai_api_unified.completions.ai_google_gemini_completions",
    "ai_api_unified.embeddings.ai_google_gemini_embeddings",
    "ai_api_unified.images.ai_google_gemini_images",
    "ai_api_unified.voice.ai_voice_google",
)


def _build_mock_google_module_tree() -> dict[str, types.ModuleType]:
    """Build a minimal importable Google package tree for provider module imports."""
    mock_genai = types.ModuleType("google.genai")
    mock_genai_types = types.ModuleType("google.genai.types")
    mock_genai_errors = types.ModuleType("google.genai.errors")
    mock_genai_pagers = types.ModuleType("google.genai.pagers")
    mock_genai.Client = Mock()
    mock_genai.types = mock_genai_types
    mock_genai.errors = mock_genai_errors
    mock_genai.pagers = mock_genai_pagers

    mock_genai_types.GenerateContentResponse = type("GenerateContentResponse", (), {})
    mock_genai_types.GenerateContentConfig = type(
        "GenerateContentConfig",
        (),
        {"__init__": lambda self, **kwargs: None},
    )
    mock_genai_types.EmbedContentConfig = type(
        "EmbedContentConfig",
        (),
        {"__init__": lambda self, **kwargs: None},
    )
    mock_genai_types.GenerateImagesResponse = type("GenerateImagesResponse", (), {})
    mock_genai_types.Model = type("Model", (), {})

    google_module = types.ModuleType("google")
    google_module.genai = mock_genai

    mock_api_core = types.ModuleType("google.api_core")
    mock_api_core_exceptions = types.ModuleType("google.api_core.exceptions")
    mock_api_core_client_options = types.ModuleType("google.api_core.client_options")
    mock_api_core.exceptions = mock_api_core_exceptions
    mock_api_core.client_options = mock_api_core_client_options
    google_module.api_core = mock_api_core

    mock_api_core_exceptions.GoogleAPICallError = type(
        "GoogleAPICallError",
        (Exception,),
        {},
    )
    mock_api_core_exceptions.ResourceExhausted = type(
        "ResourceExhausted",
        (Exception,),
        {},
    )
    mock_api_core_exceptions.ServiceUnavailable = type(
        "ServiceUnavailable",
        (Exception,),
        {},
    )
    mock_api_core_exceptions.InvalidArgument = type(
        "InvalidArgument",
        (Exception,),
        {},
    )
    mock_api_core_exceptions.ClientError = type("ClientError", (Exception,), {})
    mock_api_core_client_options.ClientOptions = type(
        "ClientOptions",
        (),
        {"__init__": lambda self, **kwargs: None},
    )

    mock_auth = types.ModuleType("google.auth")
    mock_auth_exceptions = types.ModuleType("google.auth.exceptions")
    mock_auth.exceptions = mock_auth_exceptions
    mock_auth_exceptions.DefaultCredentialsError = type(
        "DefaultCredentialsError",
        (Exception,),
        {},
    )
    google_module.auth = mock_auth

    mock_cloud = types.ModuleType("google.cloud")
    mock_cloud_speech = types.ModuleType("google.cloud.speech_v1p1beta1")
    mock_cloud_texttospeech = types.ModuleType("google.cloud.texttospeech")
    mock_cloud.speech_v1p1beta1 = mock_cloud_speech
    mock_cloud.texttospeech = mock_cloud_texttospeech
    google_module.cloud = mock_cloud

    mock_cloud_speech.SpeechClient = type("SpeechClient", (), {})
    mock_cloud_speech.RecognitionConfig = type("RecognitionConfig", (), {})
    mock_cloud_speech.RecognitionAudio = type("RecognitionAudio", (), {})
    mock_cloud_speech.RecognizeResponse = type("RecognizeResponse", (), {})

    mock_cloud_texttospeech.TextToSpeechClient = type(
        "TextToSpeechClient",
        (),
        {"__init__": lambda self, *args, **kwargs: None},
    )
    mock_cloud_texttospeech.SynthesisInput = type("SynthesisInput", (), {})
    mock_cloud_texttospeech.VoiceSelectionParams = type(
        "VoiceSelectionParams",
        (),
        {},
    )
    mock_cloud_texttospeech.AudioConfig = type("AudioConfig", (), {})
    mock_cloud_texttospeech.SynthesizeSpeechResponse = type(
        "SynthesizeSpeechResponse",
        (),
        {},
    )
    mock_cloud_texttospeech.ListVoicesResponse = type("ListVoicesResponse", (), {})
    mock_cloud_texttospeech.SsmlVoiceGender = type(
        "SsmlVoiceGender",
        (),
        {"SSML_VOICE_GENDER_UNSPECIFIED": object()},
    )
    mock_cloud_texttospeech.AudioEncoding = type(
        "AudioEncoding",
        (),
        {"LINEAR16": 1},
    )

    return {
        "google": google_module,
        "google.genai": mock_genai,
        "google.genai.types": mock_genai_types,
        "google.genai.errors": mock_genai_errors,
        "google.genai.pagers": mock_genai_pagers,
        "google.api_core": mock_api_core,
        "google.api_core.exceptions": mock_api_core_exceptions,
        "google.api_core.client_options": mock_api_core_client_options,
        "google.auth": mock_auth,
        "google.auth.exceptions": mock_auth_exceptions,
        "google.cloud": mock_cloud,
        "google.cloud.speech_v1p1beta1": mock_cloud_speech,
        "google.cloud.texttospeech": mock_cloud_texttospeech,
    }


@pytest.fixture(autouse=True)
def mock_google_dependencies() -> None:
    """
    Provide isolated mock Google SDK modules for this test file only.

    The earlier fork-derived version mutated ``sys.modules`` at import time and
    left fake Google modules behind for unrelated tests. That caused later
    nonmock tests to import provider code against stubs instead of real extras.
    """
    module_names_to_restore: tuple[str, ...] = (
        *MOCKED_GOOGLE_MODULE_NAMES,
        *GOOGLE_PROVIDER_MODULE_NAMES,
    )
    original_modules: dict[str, object | None] = {
        module_name: sys.modules.get(module_name)
        for module_name in module_names_to_restore
    }

    for module_name in module_names_to_restore:
        sys.modules.pop(module_name, None)

    sys.modules.update(_build_mock_google_module_tree())
    try:
        yield
    finally:
        for module_name in module_names_to_restore:
            sys.modules.pop(module_name, None)
        for module_name, original_module in original_modules.items():
            if original_module is not None:
                sys.modules[module_name] = original_module

from ai_api_unified.ai_base import (  # noqa: E402
    AIStructuredPrompt,
)


def _import_google_gemini_embeddings_module():
    return importlib.import_module("ai_api_unified.embeddings.ai_google_gemini_embeddings")


def _import_google_gemini_completions_module():
    return importlib.import_module(
        "ai_api_unified.completions.ai_google_gemini_completions"
    )


def _import_google_base_module():
    return importlib.import_module("ai_api_unified.ai_google_base")


def _import_google_voice_module():
    return importlib.import_module("ai_api_unified.voice.ai_voice_google")


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
        embeddings_module = _import_google_gemini_embeddings_module()
        with patch.object(embeddings_module, "genai") as mock_genai:
            with patch.object(embeddings_module, "gerr"):
                with patch.object(embeddings_module, "EnvSettings") as mock_env:
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
        completions_module = _import_google_gemini_completions_module()
        with patch.object(completions_module, "genai") as mock_genai:
            with patch.object(completions_module, "gerr"):
                with patch.object(completions_module, "EnvSettings") as mock_env:
                    mock_env_instance = Mock()
                    mock_env_instance.get_setting.side_effect = lambda key, default: {
                        "COMPLETIONS_MODEL_NAME": "gemini-2.5-flash"
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

                    client = GoogleGeminiCompletions(model="gemini-2.5-flash")

                    assert client.model_name == "gemini-2.5-flash"
                    assert isinstance(client.list_model_names, list)
                    assert "gemini-2.5-flash" in client.list_model_names
                    assert client.max_context_tokens > 0
                    assert client.price_per_1k_tokens > 0

                    assert hasattr(client, "capabilities")
                    capabilities = client.capabilities
                    assert capabilities.context_window_length > 0
                    assert isinstance(capabilities.reasoning, bool)
                    assert len(capabilities.supported_data_types) > 0

                    assert hasattr(client, "send_prompt")
                    assert hasattr(client, "strict_schema_prompt")

    def test_gemini_embeddings_basic_functionality(self):
        """Test basic Google Gemini embeddings functionality with mocking."""
        embeddings_module = _import_google_gemini_embeddings_module()
        with patch.object(embeddings_module, "genai") as mock_genai:
            with patch.object(embeddings_module, "gerr"):
                with patch.object(embeddings_module, "EnvSettings") as mock_env:
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
        completions_module = _import_google_gemini_completions_module()
        with patch.object(completions_module, "genai") as mock_genai:
            with patch.object(completions_module, "gerr"):
                with patch.object(completions_module, "EnvSettings") as mock_env:
                    mock_env_instance = Mock()
                    mock_env_instance.get_setting.side_effect = lambda key, default: {
                        "COMPLETIONS_MODEL_NAME": "gemini-2.5-flash"
                    }.get(key, default)
                    mock_env_instance.get_geo_residency.return_value = None
                    mock_env.return_value = mock_env_instance

                    mock_client = Mock()
                    mock_models = Mock()
                    mock_client.models = mock_models
                    mock_response = Mock()
                    mock_response.text = "This is a test response"
                    mock_response.candidates = [Mock(finish_reason="STOP")]
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

    def test_google_base_defaults_to_api_key_auth(self):
        """Google shared client construction should prefer API-key auth when unset."""
        google_base_module = _import_google_base_module()
        with patch.object(google_base_module, "genai") as mock_genai:
            with patch.object(google_base_module, "EnvSettings") as mock_env:
                mock_env_instance = Mock()
                mock_env_instance.get_setting.side_effect = lambda key, default=None: {
                    "GOOGLE_AUTH_METHOD": "api_key",
                    "GOOGLE_GEMINI_API_KEY": "test-api-key",
                }.get(key, default)
                mock_env.return_value = mock_env_instance

                mock_client = Mock()
                mock_client.models.get.return_value = None
                mock_genai.Client.return_value = mock_client

                from ai_api_unified.ai_google_base import AIGoogleBase

                base = AIGoogleBase()
                client = base.get_client(model="gemini-2.5-flash")

                assert client is mock_client
                mock_genai.Client.assert_called_once_with(api_key="test-api-key")

    def test_google_voice_defaults_to_api_key_auth(self):
        """Google voice should also default to API-key auth for TTS client creation."""
        google_voice_module = _import_google_voice_module()
        with patch.object(
            google_voice_module.texttospeech,
            "TextToSpeechClient",
        ) as mock_tts_client:
            with patch.object(
                google_voice_module,
                "ClientOptions",
            ) as mock_client_options:
                with patch.object(google_voice_module, "EnvSettings") as mock_env:
                    mock_env_instance = Mock()
                    mock_env_instance.get_setting.side_effect = (
                        lambda key, default=None: {
                            "GOOGLE_AUTH_METHOD": "api_key",
                            "GOOGLE_GEMINI_API_KEY": "test-api-key",
                            "DEFAULT_GEMINI_TTS_MODEL": "gemini-2.5-pro-tts",
                        }.get(key, default)
                    )
                    mock_env.return_value = mock_env_instance

                    mock_client_options_instance = Mock()
                    mock_client_options.return_value = mock_client_options_instance
                    mock_tts_client.return_value = Mock()

                    from ai_api_unified.voice.ai_voice_google import AIVoiceGoogle

                    AIVoiceGoogle(engine="google")

                    mock_client_options.assert_called_once_with(api_key="test-api-key")
                    mock_tts_client.assert_called_once_with(
                        client_options=mock_client_options_instance
                    )

    def test_gemini_completions_with_prompt_params(self):
        """Test Google Gemini completions with custom prompt parameters."""
        completions_module = _import_google_gemini_completions_module()
        with patch.object(completions_module, "genai") as mock_genai:
            with patch.object(completions_module, "gerr"):
                with patch.object(completions_module, "EnvSettings") as mock_env:
                    mock_env_instance = Mock()
                    mock_env_instance.get_setting.side_effect = lambda key, default: {
                        "COMPLETIONS_MODEL_NAME": "gemini-2.5-flash-lite"
                    }.get(key, default)
                    mock_env_instance.get_geo_residency.return_value = None
                    mock_env.return_value = mock_env_instance

                    mock_client = Mock()
                    mock_models = Mock()
                    mock_client.models = mock_models
                    mock_response = Mock()
                    mock_response.text = "This is a test response with custom params"
                    mock_response.candidates = [Mock(finish_reason="STOP")]
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
        completions_module = _import_google_gemini_completions_module()
        with patch.object(completions_module, "genai") as mock_genai:
            with patch.object(completions_module, "gerr"):
                with patch.object(completions_module, "EnvSettings") as mock_env:
                    mock_env_instance = Mock()
                    mock_env_instance.get_setting.side_effect = lambda key, default: {
                        "COMPLETIONS_MODEL_NAME": "gemini-2.5-flash"
                    }.get(key, default)
                    mock_env_instance.get_geo_residency.return_value = None
                    mock_env.return_value = mock_env_instance

                    mock_client = Mock()
                    mock_models = Mock()
                    mock_client.models = mock_models
                    mock_response = Mock()
                    mock_response.text = '{"name":"Alice","age":30,"city":"Paris"}'
                    mock_response.candidates = [Mock(finish_reason="STOP")]
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
                            max_output_tokens=2048,
                            system_instruction=(
                                "Respond only with JSON following the provided schema."
                            ),
                            response_mime_type="application/json",
                            response_schema=ExampleStructuredPrompt,
                        )

    def test_gemini_completions_structured_prompt_logs_info_for_low_max_tokens(
        self,
    ) -> None:
        """Structured prompts should fail fast when max tokens is below the enforced minimum."""
        completions_module = _import_google_gemini_completions_module()
        with patch.object(completions_module, "genai") as mock_genai:
            with patch.object(completions_module, "gerr"):
                with patch.object(completions_module, "EnvSettings") as mock_env:
                    mock_env_instance = Mock()
                    mock_env_instance.get_setting.side_effect = lambda key, default: {
                        "COMPLETIONS_MODEL_NAME": "gemini-2.5-flash"
                    }.get(key, default)
                    mock_env_instance.get_geo_residency.return_value = None
                    mock_env.return_value = mock_env_instance

                    mock_client = Mock()
                    mock_models = Mock()
                    mock_client.models = mock_models
                    mock_response = Mock()
                    mock_response.text = '{"name":"Alice","age":30,"city":"Paris"}'
                    mock_response.candidates = [Mock(finish_reason="STOP")]
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
                        with pytest.raises(
                            completions_module.StructuredResponseTokenLimitError,
                            match="max_response_tokens=512",
                        ):
                            client.strict_schema_prompt(
                                "Describe a person",
                                ExampleStructuredPrompt,
                                max_response_tokens=512,
                            )


if __name__ == "__main__":
    pytest.main([__file__])
