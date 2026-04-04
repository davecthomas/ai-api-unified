from __future__ import annotations

import pytest

pytest.importorskip("boto3")

from ai_api_unified.videos.ai_bedrock_videos import AINovaReelVideoProperties


def test_nova_reel_rejects_multi_video_requests() -> None:
    """Nova Reel should fail fast instead of silently downgrading multi-video requests."""

    with pytest.raises(
        ValueError,
        match="supports exactly one generated video per request",
    ):
        AINovaReelVideoProperties(num_videos=2)
