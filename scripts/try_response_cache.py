# scripts/try_response_cache.py
"""
Live demonstration of the response cache against the configured provider.

Copies ./models.json (or the file AI_MODELS_CONFIG names) with the
response_cache middleware turned on, then runs the same prompts twice:
round 1 against the empty cache (every call reaches the provider), round 2
against the filled cache (every call is a hit). Prints what each call cost,
the counters after each round, and every live entry (metadata only).

    poetry run python scripts/try_response_cache.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
from pathlib import Path

LIST_PROMPTS: list[str] = [
    "Name one prime number between 10 and 20. Answer with the number only.",
    "Name one even number below 10. Answer with the number only.",
    "What is the capital of France? Answer with the city only.",
]


def main() -> int:
    logging.basicConfig(level=logging.WARNING)
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

    from ai_api_unified import (
        AIFactory,
        response_cache_entries,
        response_cache_stats,
    )

    client = AIFactory.get_ai_completions_client()

    def _round(str_title: str) -> None:
        print(f"\n== {str_title}")
        # Loop over the prompts, printing what each call cost.
        for int_index, str_prompt in enumerate(LIST_PROMPTS, start=1):
            result = client.send_prompt_with_result(str_prompt)
            print(
                f"  {int_index}. text={result.text.strip()!r:<8} "
                f"model={result.model_name} cache_hit={result.cache_hit} "
                f"cost=${result.cost.total_usd} usd_saved={result.usd_saved}"
            )
        dict_stats: dict = response_cache_stats()
        print(
            f"  counters: entries={dict_stats['entries']} hits={dict_stats['hits']} "
            f"misses={dict_stats['misses']} usd_saved=${dict_stats['usd_saved']}"
        )

    _round("Round 1: empty cache, every call reaches the provider")
    _round("Round 2: same prompts, every call is a hit")

    print("\n== Cache contents (metadata only; never the prompt or the answer)")
    print(
        f"  {'key':<34} {'operation':<12} {'model':<24} {'expires':>8} "
        f"{'usd/hit':>11} {'bytes':>6}"
    )
    # Loop over the live entries, most recently used first.
    for entry in response_cache_entries():
        print(
            f"  {entry['key'][:34]:<34} {entry['operation'] or '-':<12} "
            f"{(entry['model'] or '-')[:24]:<24} "
            f"{entry['expires_in_seconds']:>7.0f}s "
            f"{str(entry['usd']) if entry['usd'] is not None else '-':>11} "
            f"{entry['bytes']:>6}"
        )
    Path(str_temp).unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
