# tests/test_route_ceiling.py
"""
provider_options["route_ceiling"]: the most demanding tier a call may use.
A pick above it is lowered to it (method `ceiling`), and escalation stops
there.
"""

from __future__ import annotations

from typing import Any

import pytest

from ai_api_unified import AiFallbackReason as R
from ai_api_unified.completions.ai_router_completions import ROUTE_CEILING_INDEX
from tests.test_model_router import _engines, _FakeClassifier, _router, _tier
from tests.test_request_cost import _err

MESSAGES: list[dict[str, Any]] = [{"role": "user", "content": "hi"}]
TIERS = [
    _tier("easy", "claude:e"),
    _tier("medium", "claude:m"),
    _tier("hard", "claude:h"),
]


def _turn(router: Any, **options: Any) -> Any:
    return router.send_conversation("sys", MESSAGES, provider_options=options or None)


class TestCeiling:
    def test_a_pick_above_the_ceiling_is_lowered(self) -> None:
        engines = _engines("e", "m", "h")
        router = _router(engines, tiers=TIERS, classifier=_FakeClassifier("hard"))
        turn = _turn(router, route_ceiling="medium")
        assert turn.text == "m:turn" and engines["claude:h"].calls == []
        assert turn.route_tier == "medium" and turn.route_method == "ceiling"
        # The reserved key never reaches the engine.
        dict_seen = engines["claude:m"].last_kwargs.get("provider_options") or {}
        assert "route_ceiling" not in dict_seen

    def test_a_pick_at_or_below_the_ceiling_is_unchanged(self) -> None:
        router = _router(
            _engines("e", "m", "h"), tiers=TIERS, classifier=_FakeClassifier("easy")
        )
        turn = _turn(router, route_ceiling="medium")
        assert turn.route_tier == "easy" and turn.route_method == "classifier"

    def test_escalation_stops_at_the_ceiling(self) -> None:
        engines = _engines(
            "e", "m", "h", e=_err(R.UNAVAILABLE, 529), m=_err(R.UNAVAILABLE, 529)
        )
        router = _router(engines, tiers=TIERS, classifier=_FakeClassifier("easy"))
        with pytest.raises(Exception):
            _turn(router, route_ceiling="medium")
        assert engines["claude:m"].calls and engines["claude:h"].calls == []

    def test_the_ceiling_does_not_carry_over_to_the_next_call(self) -> None:
        engines = _engines(
            "e", "m", "h", e=_err(R.UNAVAILABLE, 529), m=_err(R.UNAVAILABLE, 529)
        )
        router = _router(engines, tiers=TIERS, classifier=_FakeClassifier("easy"))
        with pytest.raises(Exception):
            _turn(router, route_ceiling="medium")
        assert _turn(router).text == "h:turn"
        assert ROUTE_CEILING_INDEX.get() is None

    def test_an_override_is_not_moved(self) -> None:
        router = _router(
            _engines("e", "m", "h"), tiers=TIERS, classifier=_FakeClassifier("easy")
        )
        turn = _turn(router, route="hard", route_ceiling="easy")
        assert turn.route_tier == "hard" and turn.route_method == "override"

    def test_floor_and_ceiling_together(self) -> None:
        router = _router(
            _engines("e", "m", "h"), tiers=TIERS, classifier=_FakeClassifier("easy")
        )
        turn = _turn(router, route_floor="medium", route_ceiling="medium")
        assert turn.route_tier == "medium"

    def test_a_floor_above_the_ceiling_raises(self) -> None:
        router = _router(_engines("e", "m", "h"), tiers=TIERS)
        with pytest.raises(ValueError, match="above"):
            _turn(router, route_floor="hard", route_ceiling="easy")

    def test_an_unknown_ceiling_raises(self) -> None:
        router = _router(_engines("e", "m", "h"), tiers=TIERS)
        with pytest.raises(ValueError, match="route_ceiling"):
            _turn(router, route_ceiling="extreme")

    def test_structured_output_honors_the_ceiling(self) -> None:
        engines = _engines("e", "m", "h")
        router = _router(engines, tiers=TIERS, classifier=_FakeClassifier("hard"))
        result = router.send_structured_output(
            "q",
            response_schema={"type": "object"},
            provider_options={"route_ceiling": "easy"},
        )
        assert result.data == {"answer": "e"}

    def test_preview_honors_the_ceiling(self) -> None:
        router = _router(
            _engines("e", "m", "h"), tiers=TIERS, classifier=_FakeClassifier("hard")
        )
        preview = router.preview_route("q", provider_options={"route_ceiling": "easy"})
        assert preview.tier == "easy" and preview.method == "ceiling"
