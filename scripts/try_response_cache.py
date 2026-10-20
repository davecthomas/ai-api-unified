# scripts/try_response_cache.py
"""
Live demonstration of the response cache against the configured provider.

Copies ./models.json (or the file AI_MODELS_CONFIG names) with the
response_cache middleware turned on, then sends the same prompt three
times: a miss, a hit, and a call with cache="off". Prints what each cost.

    poetry run python scripts/try_response_cache.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path


def main() -> int:
    str_source: str = os.environ.get("AI_MODELS_CONFIG") or "models.json"
    path_source: Path = Path(str_source)
    if not path_source.is_file():
        print(f"No models config at {path_source}; set AI_MODELS_CONFIG.")
        return 1
    dict_config: dict = json.loads(path_source.read_text(encoding="utf-8"))
    list_middleware: list = [
        entry
        for entry in (dict_config.get("middleware") or [])
        if entry.get("name") != "response_cache"
    ]
    list_middleware.append({"name": "response_cache", "enabled": True})
    dict_config["middleware"] = list_middleware
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
        json.dump(dict_config, handle)
        str_temp: str = handle.name
    os.environ["AI_MODELS_CONFIG"] = str_temp

    from ai_api_unified import AIFactory
    from ai_api_unified.ai_base import AICompletionsPromptParamsBase

    class _Params(AICompletionsPromptParamsBase):
        pass

    client = AIFactory.get_ai_completions_client()
    str_prompt: str = (
        "Name one prime number between 10 and 20. Answer with the number only."
    )

    def _show(str_label: str, result) -> None:
        print(
            f"{str_label:<14} text={result.text.strip()!r:<8} model={result.model_name} "
            f"cache_hit={result.cache_hit} cost=${result.cost.total_usd} "
            f"usd_saved={result.usd_saved}"
        )

    _show("1 first call", client.send_prompt_with_result(str_prompt))
    _show("2 same call", client.send_prompt_with_result(str_prompt))
    _show(
        "3 cache=off",
        client.send_prompt_with_result(str_prompt, other_params=_Params(cache="off")),
    )
    _show("4 same again", client.send_prompt_with_result(str_prompt))
    Path(str_temp).unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
