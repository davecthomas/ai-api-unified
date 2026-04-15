"""Tests for built package metadata validation helpers."""

from __future__ import annotations

import io
from pathlib import Path
import tarfile
import zipfile

from ai_api_unified.util.package_metadata_validation import (
    collect_distribution_paths,
    find_direct_url_requirements,
    main,
)

DIRECT_URL_REQUIREMENT: str = (
    "en-core-web-sm @ "
    "https://github.com/explosion/spacy-models/releases/download/"
    "en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl ; "
    'extra == "middleware-pii-redaction-small"'
)


def _build_metadata_text(list_str_requirements: list[str]) -> str:
    """
    Builds minimal package metadata text for a synthetic distribution artifact.

    Args:
        list_str_requirements: ``Requires-Dist`` values to encode.

    Returns:
        Metadata text suitable for METADATA or PKG-INFO payloads.
    """
    list_str_lines: list[str] = [
        "Metadata-Version: 2.3",
        "Name: ai_api_unified",
        "Version: 2.5.1",
    ]
    list_str_lines.extend(
        f"Requires-Dist: {str_requirement}" for str_requirement in list_str_requirements
    )
    return "\n".join(list_str_lines) + "\n"


def _write_wheel(path_distribution: Path, str_metadata_text: str) -> None:
    """
    Writes a synthetic wheel containing package metadata.

    Args:
        path_distribution: Output wheel path.
        str_metadata_text: Metadata payload to embed.

    Returns:
        None
    """
    with zipfile.ZipFile(path_distribution, "w") as zip_distribution:
        zip_distribution.writestr(
            "ai_api_unified-2.5.1.dist-info/METADATA",
            str_metadata_text,
        )


def _write_sdist(path_distribution: Path, str_metadata_text: str) -> None:
    """
    Writes a synthetic source distribution containing PKG-INFO metadata.

    Args:
        path_distribution: Output source distribution path.
        str_metadata_text: Metadata payload to embed.

    Returns:
        None
    """
    bytes_metadata: bytes = str_metadata_text.encode("utf-8")
    tar_info: tarfile.TarInfo = tarfile.TarInfo("ai_api_unified-2.5.1/PKG-INFO")
    tar_info.size = len(bytes_metadata)
    with tarfile.open(path_distribution, "w:gz") as tar_distribution:
        tar_distribution.addfile(tar_info, io.BytesIO(bytes_metadata))


def test_collect_distribution_paths_expands_dist_directory(tmp_path: Path) -> None:
    """
    Verifies supported wheel and sdist artifacts are discovered from a directory.

    Args:
        tmp_path: Temporary pytest directory fixture.

    Returns:
        None
    """
    path_wheel: Path = tmp_path / "ai_api_unified-2.5.1-py3-none-any.whl"
    path_sdist: Path = tmp_path / "ai_api_unified-2.5.1.tar.gz"
    path_notes: Path = tmp_path / "notes.txt"

    _write_wheel(path_wheel, _build_metadata_text([]))
    _write_sdist(path_sdist, _build_metadata_text([]))
    path_notes.write_text("ignore me", encoding="utf-8")

    list_path_distributions: list[Path] = collect_distribution_paths([str(tmp_path)])

    assert path_sdist in list_path_distributions
    assert path_wheel in list_path_distributions
    assert path_notes not in list_path_distributions


def test_find_direct_url_requirements_flags_invalid_wheel_and_sdist(
    tmp_path: Path,
) -> None:
    """
    Verifies direct URL requirements are detected in both wheel and sdist metadata.

    Args:
        tmp_path: Temporary pytest directory fixture.

    Returns:
        None
    """
    str_metadata_text: str = _build_metadata_text(
        [
            "presidio-analyzer>=2.2.35",
            DIRECT_URL_REQUIREMENT,
        ]
    )
    path_wheel: Path = tmp_path / "ai_api_unified-2.5.1-py3-none-any.whl"
    path_sdist: Path = tmp_path / "ai_api_unified-2.5.1.tar.gz"

    _write_wheel(path_wheel, str_metadata_text)
    _write_sdist(path_sdist, str_metadata_text)

    dict_path_to_invalid_requirements: dict[Path, list[str]] = (
        find_direct_url_requirements([path_wheel, path_sdist])
    )

    assert dict_path_to_invalid_requirements[path_wheel] == [DIRECT_URL_REQUIREMENT]
    assert dict_path_to_invalid_requirements[path_sdist] == [DIRECT_URL_REQUIREMENT]


def test_main_returns_success_for_publish_safe_metadata(
    tmp_path: Path,
    capsys,
) -> None:
    """
    Verifies the CLI entrypoint accepts metadata without direct URL requirements.

    Args:
        tmp_path: Temporary pytest directory fixture.
        capsys: Pytest stdout/stderr capture fixture.

    Returns:
        None
    """
    path_wheel: Path = tmp_path / "ai_api_unified-2.5.1-py3-none-any.whl"
    _write_wheel(
        path_wheel,
        _build_metadata_text(
            [
                "presidio-analyzer>=2.2.35",
                "spacy<4.0.0,>=3.4.4,!=3.7.0 ; extra == 'middleware-pii-redaction'",
            ]
        ),
    )

    int_exit_code: int = main([str(tmp_path)])
    captured_output = capsys.readouterr()

    assert int_exit_code == 0
    assert "no direct URL requirements found" in captured_output.out
    assert captured_output.err == ""
