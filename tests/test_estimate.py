from ai_api.ai_base import AIBaseCompletions

class Dummy(AIBaseCompletions):
    def __init__(self):
        pass

    @property
    def list_model_names(self):
        return ["dummy"]

    def max_context_tokens(self) -> int:
        return 256

    @property
    def price_per_1k_tokens(self) -> float:
        return 0.0

    def strict_schema_prompt(self, prompt, response_model, max_response_tokens=512):
        raise NotImplementedError

    def send_prompt(self, prompt: str) -> str:
        return ""

def test_estimate_max_tokens_rounds_up():
    result = Dummy.estimate_max_tokens(5)
    assert isinstance(result, int)
    assert result % 16 == 0
