from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from ai_api_unified.videos import frame_helpers


class _FakeReader:
    """Small imageio reader double used to exercise frame extraction logic."""

    def __init__(self) -> None:
        self.closed: bool = False

    def get_meta_data(self) -> dict[str, Any]:
        return {"fps": 24}

    def get_data(self, frame_index: int) -> dict[str, int]:
        return {"frame_index": frame_index}

    def close(self) -> None:
        self.closed = True


class _FakeImageIOModule:
    """imageio test double that records the temp video path handed to ffmpeg."""

    def __init__(self, expected_video_bytes: bytes) -> None:
        self.expected_video_bytes: bytes = expected_video_bytes
        self.last_path: Path | None = None
        self.last_format: str | None = None
        self.last_reader: _FakeReader | None = None

    def get_reader(self, path: str, format: str) -> _FakeReader:
        self.last_path = Path(path)
        self.last_format = format
        assert self.last_path.read_bytes() == self.expected_video_bytes
        self.last_reader = _FakeReader()
        return self.last_reader


class _FakePILFrame:
    """Simple Pillow frame double that serializes the save format into bytes."""

    def __init__(self, frame_index: int, saved_formats: list[str]) -> None:
        self.frame_index: int = frame_index
        self.saved_formats: list[str] = saved_formats

    def save(self, buffer_stream: Any, *, format: str) -> None:
        self.saved_formats.append(format)
        buffer_stream.write(f"{format}:{self.frame_index}".encode("utf-8"))


class _FakePILImageModule:
    """Pillow module double that records requested save formats."""

    def __init__(self) -> None:
        self.saved_formats: list[str] = []

    def fromarray(self, frame_array: dict[str, int]) -> _FakePILFrame:
        return _FakePILFrame(frame_array["frame_index"], self.saved_formats)


def test_extract_image_frames_reads_from_temp_path_and_closes_reader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Frame extraction should materialize the video buffer on disk before opening ffmpeg."""

    video_buffer: bytes = b"synthetic-video"
    fake_imageio: _FakeImageIOModule = _FakeImageIOModule(video_buffer)
    fake_pil_image: _FakePILImageModule = _FakePILImageModule()
    monkeypatch.setattr(
        frame_helpers,
        "_load_frame_dependencies",
        lambda: (fake_imageio, fake_pil_image),
    )

    image_buffers: list[bytes] = frame_helpers.extract_image_frames_from_video_buffer(
        video_buffer,
        frame_indices=[0],
    )

    assert image_buffers == [b"PNG:0"]
    assert fake_imageio.last_path is not None
    assert fake_imageio.last_format == "ffmpeg"
    assert fake_imageio.last_reader is not None
    assert fake_imageio.last_reader.closed is True


def test_extract_image_frames_maps_jpg_alias_to_jpeg(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Common jpg aliases should be normalized to Pillow's JPEG format key."""

    fake_imageio: _FakeImageIOModule = _FakeImageIOModule(b"synthetic-video")
    fake_pil_image: _FakePILImageModule = _FakePILImageModule()
    monkeypatch.setattr(
        frame_helpers,
        "_load_frame_dependencies",
        lambda: (fake_imageio, fake_pil_image),
    )

    image_buffers: list[bytes] = frame_helpers.extract_image_frames_from_video_buffer(
        b"synthetic-video",
        frame_indices=[0],
        image_format="jpg",
    )

    assert image_buffers == [b"JPEG:0"]
    assert fake_pil_image.saved_formats == ["JPEG"]


def test_extract_image_frames_rejects_unsupported_image_formats() -> None:
    """Unsupported output image formats should fail fast with a clear error."""

    with pytest.raises(ValueError, match="image_format must be one of"):
        frame_helpers.extract_image_frames_from_video_buffer(
            b"synthetic-video",
            frame_indices=[0],
            image_format="svg",
        )
