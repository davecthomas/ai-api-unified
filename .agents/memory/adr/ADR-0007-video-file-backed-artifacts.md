---
adr: "0007"
title: File-backed video artifacts as default output representation
status: Accepted
date: "2026-04-04"
tags: [architecture, video, artifacts, memory-management]
must_read: false
supersedes: ~
superseded_by: ~
---

# ADR-0007: File-backed video artifacts as default output representation

## Status
Accepted

## Context
The existing image generation interface returns `list[bytes]`, which works for images but creates memory pressure for video payloads (tens to hundreds of MB). Additionally, Bedrock naturally delivers results to S3, and forcing all artifacts into in-memory bytes would add unnecessary download-to-memory steps.

## Decision
`AIVideoArtifact` is file-first: `file_path: Path | None` is the primary accessor. A convenience `read_bytes()` method is available for callers who need in-memory access (e.g., frame extraction). When `download_outputs=True` (the default), providers download artifacts to `output_dir` or a deterministic temporary directory. When `download_outputs=False`, `file_path` may be `None` but `remote_uri` is still populated when available.

## Consequences
- Avoids surprising memory spikes for large video files.
- Matches Bedrock's natural S3 delivery model.
- Frame extraction helpers compose cleanly via `artifact.read_bytes()` -> `extract_image_frames_from_video_buffer(...)`.
- Downstream callers inspect `artifact.file_path` to copy/upload; no accidental multi-hundred-MB allocations.

## Evidence
`docs/video_generation_design.md` (§ Why File-Backed Artifacts Are The Default, § Normalized Result Types).
