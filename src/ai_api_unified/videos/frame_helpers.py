from __future__ import annotations

import importlib
import io
import tempfile
from pathlib import Path
from typing import Any

FRAME_EXTRA_INSTALL_MESSAGE: str = (
    "Video frame extraction requires the optional 'video_frames' dependencies. "
    'Install them with `poetry install --extras "video_frames" --with dev` for local development '
    "or `poetry add 'ai-api-unified[video_frames]'` in downstream projects."
)


def _load_frame_dependencies() -> tuple[Any, Any]:
    """Lazily import the optional frame-extraction dependencies."""

    try:
        imageio = importlib.import_module("imageio")
        pil_image = importlib.import_module("PIL.Image")
    except ImportError as exception:
        raise RuntimeError(FRAME_EXTRA_INSTALL_MESSAGE) from exception
    return imageio, pil_image


def extract_image_frames_from_video_buffer(
    video_buffer: bytes,
    *,
    time_offsets_seconds: list[float] | None = None,
    frame_indices: list[int] | None = None,
    image_format: str = "png",
) -> list[bytes]:
    """
    Extract one or more frame images from a video buffer.
    """

    if not video_buffer:
        raise ValueError("video_buffer must be a non-empty byte string.")
    provided_time_offsets: bool = bool(time_offsets_seconds)
    provided_frame_indices: bool = bool(frame_indices)
    if provided_time_offsets == provided_frame_indices:
        raise ValueError(
            "Provide exactly one of time_offsets_seconds or frame_indices."
        )

    imageio, pil_image = _load_frame_dependencies()
    normalized_image_format: str = image_format.strip().lower()
    if normalized_image_format == "":
        raise ValueError("image_format must be a non-empty string.")

    with tempfile.NamedTemporaryFile(suffix=".mp4") as temp_video_file:
        temp_video_file.write(video_buffer)
        temp_video_file.flush()
        reader = imageio.get_reader(temp_video_file.name, format="ffmpeg")
        try:
            metadata: dict[str, Any] = reader.get_meta_data()
            resolved_frame_indices: list[int]
            if time_offsets_seconds is not None:
                fps_value: Any = metadata.get("fps")
                if fps_value in (None, 0):
                    raise ValueError(
                        "Unable to resolve FPS metadata for time-based frame extraction."
                    )
                fps: float = float(fps_value)
                resolved_frame_indices = []
                for offset_seconds in time_offsets_seconds:
                    if offset_seconds < 0:
                        raise ValueError(
                            "time_offsets_seconds values must be zero or greater."
                        )
                    resolved_frame_indices.append(
                        max(0, int(round(offset_seconds * fps)))
                    )
            else:
                assert frame_indices is not None
                resolved_frame_indices = []
                for frame_index in frame_indices:
                    if frame_index < 0:
                        raise ValueError(
                            "frame_indices values must be zero or greater."
                        )
                    resolved_frame_indices.append(frame_index)

            frame_cache: dict[int, bytes] = {}
            image_buffers: list[bytes] = []
            for frame_index in resolved_frame_indices:
                cached_image_buffer: bytes | None = frame_cache.get(frame_index)
                if cached_image_buffer is not None:
                    image_buffers.append(cached_image_buffer)
                    continue
                try:
                    frame_array: Any = reader.get_data(frame_index)
                except IndexError as exception:
                    raise ValueError(
                        f"Requested frame index {frame_index} is outside the video bounds."
                    ) from exception
                pil_frame = pil_image.fromarray(frame_array)
                buffer_stream: io.BytesIO = io.BytesIO()
                pil_frame.save(buffer_stream, format=normalized_image_format.upper())
                frame_bytes: bytes = buffer_stream.getvalue()
                frame_cache[frame_index] = frame_bytes
                image_buffers.append(frame_bytes)
            return image_buffers
        finally:
            reader.close()
