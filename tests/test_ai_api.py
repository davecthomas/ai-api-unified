import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT_DIR, "src"))
sys.path.insert(0, ROOT_DIR)

from ai_api.ai_base import AIBaseEmbeddings, AIBaseCompletions, AIStructuredPrompt


class DummyEmbeddings(AIBaseEmbeddings):
    def __init__(self):
        self.called = []

    @property
    def list_model_names(self):
        return ["dummy"]

    def generate_embeddings(self, text: str):
        self.called.append(text)
        return {
            "embedding": [ord(c) for c in text],
            "text": text,
            "dimensions": len(text),
        }

    def generate_embeddings_batch(self, texts):
        return [self.generate_embeddings(t) for t in texts]


class DummyCompletions(AIBaseCompletions):
    def __init__(self):
        self.prompts = []

    @property
    def list_model_names(self):
        return ["dummy"]

    @property
    def max_context_tokens(self) -> int:
        return 1024

    @property
    def price_per_1k_tokens(self) -> float:
        return 0.0

    def strict_schema_prompt(self, prompt, response_model, max_response_tokens=512):
        self.prompts.append(prompt)
        return response_model(message=prompt.upper())

    def send_prompt(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return prompt.upper()


class ExamplePrompt(AIStructuredPrompt):
    message: str | None = None

    @staticmethod
    def get_prompt() -> str:
        return "say hello in JSON with a 'message' field"


def test_generate_embeddings():
    client = DummyEmbeddings()
    result = client.generate_embeddings("hi")
    assert result["embedding"] == [104, 105]
    assert result["dimensions"] == 2


def test_send_prompt():
    client = DummyCompletions()
    response = client.send_prompt("hello")
    assert response == "HELLO"


def test_structured_prompt():
    client = DummyCompletions()
    prompt = ExamplePrompt(prompt=ExamplePrompt.get_prompt())
    result = prompt.send_structured_prompt(client, ExamplePrompt)
    assert isinstance(result, ExamplePrompt)
    assert result.message == prompt.prompt.upper()
