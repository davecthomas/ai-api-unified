import os
import pytest

from ai_api.embeddings.ai_openai_embeddings import AiOpenAIEmbeddings
from ai_api.completions.ai_openai_completions import AiOpenAICompletions
from ai_api.ai_base import AIStructuredPrompt

class EchoPrompt(AIStructuredPrompt):
    text: str

    @staticmethod
    def get_prompt() -> str:
        return "Reply with a JSON object containing a 'text' field."

# Helper to skip tests without credentials
openai_key = os.getenv("OPENAI_API_KEY")

@pytest.mark.skipif(not openai_key, reason="OPENAI_API_KEY not set")
def test_generate_embeddings():
    client = AiOpenAIEmbeddings()
    result = client.generate_embeddings("hello world")
    assert "embedding" in result
    assert isinstance(result["embedding"], list)

@pytest.mark.skipif(not openai_key, reason="OPENAI_API_KEY not set")
def test_send_prompt():
    client = AiOpenAICompletions()
    response = client.send_prompt("Say hello in one sentence")
    assert isinstance(response, str)
    assert response

@pytest.mark.skipif(not openai_key, reason="OPENAI_API_KEY not set")
def test_structured_prompt():
    client = AiOpenAICompletions()
    p = EchoPrompt(prompt=EchoPrompt.get_prompt())
    result = p.send_structured_prompt(client, EchoPrompt)
    assert isinstance(result, EchoPrompt)
    assert result.text
