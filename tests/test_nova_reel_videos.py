from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("boto3")

from ai_api_unified.ai_base import AIMediaReference, AIBaseVideoProperties
from ai_api_unified.videos.ai_bedrock_videos import (
    AINovaReelVideoProperties,
    AINovaReelVideos,
)


class _InspectableNovaReelVideos(AINovaReelVideos):
    """Nova Reel test double exposing helper methods without bootstrapping Bedrock."""

    def __init__(self) -> None:
        pass


class _FakePaginator:
    """Simple paginator double that records prefix scans for S3 listing tests."""

    def __init__(self, pages_by_prefix: dict[str, list[dict[str, Any]]]) -> None:
        self.pages_by_prefix: dict[str, list[dict[str, Any]]] = pages_by_prefix
        self.calls: list[dict[str, str]] = []

    def paginate(self, *, Bucket: str, Prefix: str) -> list[dict[str, Any]]:
        self.calls.append({"Bucket": Bucket, "Prefix": Prefix})
        return self.pages_by_prefix.get(Prefix, [])


class _FakeS3Client:
    """Small S3 client double exposing only the paginator used by the provider."""

    def __init__(self, paginator: _FakePaginator) -> None:
        self.paginator: _FakePaginator = paginator

    def get_paginator(self, operation_name: str) -> _FakePaginator:
        assert operation_name == "list_objects_v2"
        return self.paginator


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


def test_nova_reel_coerce_properties_applies_provider_defaults_for_base_inputs() -> (
    None
):
    """Base video properties should not suppress Nova Reel defaults."""

    provider: _InspectableNovaReelVideos = _InspectableNovaReelVideos()

    coerced_properties: AINovaReelVideoProperties = provider._coerce_properties(
        AIBaseVideoProperties()
    )

    assert coerced_properties.duration_seconds == 6
    assert coerced_properties.resolution == "1280x720"
    assert coerced_properties.fps == 24
    assert coerced_properties.timeout_seconds == 1800
    assert coerced_properties.poll_interval_seconds == 10


def test_nova_reel_output_lookup_paginates_and_skips_empty_base_prefix() -> None:
    """Nova Reel output discovery should paginate within the invocation prefix only."""

    provider: _InspectableNovaReelVideos = _InspectableNovaReelVideos()
    paginator: _FakePaginator = _FakePaginator(
        {
            "job-123/": [
                {"Contents": []},
                {"Contents": [{"Key": "job-123/output.mp4"}]},
            ]
        }
    )
    provider.s3_client = _FakeS3Client(paginator)

    output_uri: str = provider._find_output_video_s3_uri(
        output_s3_uri="s3://test-bucket",
        provider_job_id="arn:aws:bedrock:us-east-1:123456789012:async-invoke/job-123",
    )

    assert output_uri == "s3://test-bucket/job-123/output.mp4"
    assert paginator.calls == [{"Bucket": "test-bucket", "Prefix": "job-123/"}]
