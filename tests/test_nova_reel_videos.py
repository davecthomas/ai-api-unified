from __future__ import annotations

import pytest

pytest.importorskip("boto3")

from ai_api_unified.ai_base import AIMediaReference
from ai_api_unified.videos.ai_bedrock_videos import (
    AINovaReelVideoProperties,
    AINovaReelVideos,
)


class _InspectableNovaReelVideos(AINovaReelVideos):
    """Nova Reel test double exposing helper methods without bootstrapping Bedrock."""

    def __init__(self) -> None:
        pass


def test_nova_reel_rejects_multi_video_requests() -> None:
    """Nova Reel should fail fast instead of silently downgrading multi-video requests."""

    with pytest.raises(
        ValueError,
        match="supports exactly one generated video per request",
    ):
        AINovaReelVideoProperties(num_videos=2)


def test_nova_reel_rejects_non_s3_remote_image_inputs() -> None:
    """Nova Reel remote image references should fail fast unless they are on S3."""

    provider: _InspectableNovaReelVideos = _InspectableNovaReelVideos()

    with pytest.raises(
        ValueError,
        match="remote_uri must use an s3:// URI",
    ):
        provider._to_bedrock_image(
            AIMediaReference(remote_uri="https://example.com/image.png")
        )
