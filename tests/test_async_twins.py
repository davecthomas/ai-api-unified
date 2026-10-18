# tests/test_async_twins.py
"""
Async twins for the calls whose SDKs have no async client: streaming,
batches, token counting, previews, images, videos, voice, and multimodal
embeddings. They run the blocking call on a worker thread in the caller's
context, so cost scopes, tags, and fallback behave as in the sync call.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any
from unittest.mock import patch

import pytest

from ai_api_unified import AiFallbackReason as R, request_cost_scope
from ai_api_unified.ai_base import (
    AIBaseCompletions,
    AIBaseImageProperties,
    AIBaseImages,
    AIBaseVideos,
    aiter_in_thread,
)
from ai_api_unified.middleware.observability_runtime import (
    push_observability_tags,
    reset_observability_context,
)
from ai_api_unified.voice.ai_voice_base import AIVoiceBase
from tests.test_request_cost import GPT_COST, _chain, _err, _PricedEngine


def _concrete(base: type) -> Any:
    """An instance of an abstract base with its abstract methods waived."""
    cls: type = type(f"_Concrete{base.__name__}", (base,), {})
    cls.__abstractmethods__ = frozenset()  # type: ignore[attr-defined]
    return cls.__new__(cls)


def _collect(agen: Any) -> list[Any]:
    async def _run() -> list[Any]:
        return [item async for item in agen]

    return asyncio.run(_run())


class TestStreaming:
    def test_chunks_arrive_in_order_and_the_request_is_metered(self) -> None:
        engine = _PricedEngine("p")
        assert _collect(engine.asend_prompt_streaming("hi")) == ["p-1", "p-2"]
        assert engine.last_request_cost is not None
        assert engine.last_request_cost.total_usd == GPT_COST

    def test_fallback_before_the_first_chunk(self) -> None:
        chain = _chain(
            _PricedEngine("p", fail=_err(R.UNAVAILABLE, 529)), _PricedEngine("f")
        )
        assert _collect(chain.asend_prompt_streaming("hi")) == ["f-1", "f-2"]
        assert chain.last_request_cost.failed_calls == 1

    def test_caller_scope_and_tags_reach_the_worker(self) -> None:
        engine = _PricedEngine("p")

        async def _run() -> Any:
            token = push_observability_tags({"tenant": "acme"})
            try:
                with request_cost_scope("step") as scope:
                    _ = [c async for c in engine.asend_prompt_streaming("hi")]
            finally:
                reset_observability_context(token)
            return scope

        scope = asyncio.run(_run())
        assert scope.report.total_usd == GPT_COST
        assert engine.tags_seen[0]["tenant"] == "acme"

    def test_errors_propagate(self) -> None:
        engine = _PricedEngine("p", fail=_err(R.UNAVAILABLE, 529))
        with pytest.raises(Exception, match="fail"):
            _collect(engine.asend_prompt_streaming("hi"))

    def test_breaking_early_closes_the_stream(self) -> None:
        closed: list[bool] = []

        def _gen() -> Any:
            try:
                for int_index in range(1000):
                    yield int_index
            finally:
                closed.append(True)

        async def _first() -> Any:
            async for item in aiter_in_thread(_gen()):
                return item

        assert asyncio.run(_first()) == 0
        assert closed == [True]


class TestThreadedTwins:
    def _runs_off_the_loop(self, coro: Any, target: Any) -> Any:
        """Awaits coro and reports the thread target ran on."""
        seen: dict[str, int] = {}

        def _record(*args: Any, **kwargs: Any) -> str:
            seen["thread"] = threading.get_ident()
            seen["args"] = args  # type: ignore[assignment]
            seen["kwargs"] = kwargs  # type: ignore[assignment]
            return "done"

        target.side_effect = _record

        async def _run() -> Any:
            seen["loop"] = threading.get_ident()
            return await coro

        assert asyncio.run(_run()) == "done"
        assert seen["thread"] != seen["loop"]
        return seen

    def test_batch_twins(self) -> None:
        engine = _PricedEngine("p")
        with patch.object(_PricedEngine, "run_batch") as run_batch:
            seen = self._runs_off_the_loop(
                engine.arun_batch([], timeout_seconds=5), run_batch
            )
        assert seen["kwargs"]["timeout_seconds"] == 5
        for str_name in (
            "submit_batch",
            "get_batch",
            "get_batch_results",
            "cancel_batch",
        ):
            with patch.object(_PricedEngine, str_name) as target:
                self._runs_off_the_loop(getattr(engine, f"a{str_name}")("b1"), target)

    def test_count_tokens_and_preview(self) -> None:
        engine = _PricedEngine("p")
        with patch.object(_PricedEngine, "count_tokens") as target:
            self._runs_off_the_loop(engine.acount_tokens("hi"), target)
        with patch.object(AIBaseCompletions, "preview_route") as target:
            seen = self._runs_off_the_loop(
                engine.apreview_route("hi", max_response_tokens=9), target
            )
        assert seen["kwargs"]["max_response_tokens"] == 9

    def test_images(self) -> None:
        images = _concrete(AIBaseImages)
        with patch.object(AIBaseImages, "generate_images") as target:
            seen = self._runs_off_the_loop(
                images.agenerate_images("a cat", AIBaseImageProperties()), target
            )
        assert seen["args"][0] == "a cat"

    def test_videos(self) -> None:
        videos = _concrete(AIBaseVideos)
        for str_name, args in (
            ("generate_video", ("a wave",)),
            ("get_video_generation_job", ("job-1",)),
            ("download_video_result", ("job-1",)),
            ("wait_for_video_generation", ("job-1",)),
        ):
            with patch.object(AIBaseVideos, str_name) as target:
                str_async = (
                    "await_for_video_generation"
                    if str_name == "wait_for_video_generation"
                    else f"a{str_name}"
                )
                self._runs_off_the_loop(getattr(videos, str_async)(*args), target)

    def test_voice(self) -> None:
        voice = _concrete(AIVoiceBase)
        with patch.object(AIVoiceBase, "text_to_voice") as target:
            seen = self._runs_off_the_loop(voice.atext_to_voice("hello"), target)
        assert seen["args"][0] == "hello"
        with patch.object(AIVoiceBase, "speech_to_text") as target:
            self._runs_off_the_loop(voice.aspeech_to_text(b"x"), target)
