# test_version_sync.py
"""
Tests that every version literal in the repository agrees.

The version lives in exactly three files (see CLAUDE.md):
    1. pyproject.toml       — [project] version
    2. src/ai_api_unified/__version__.py — __version__ constant
    3. README.md            — the title heading on line 1

These tests fail whenever a bump touches one location and misses another,
which is how the README title drifted to 2.5.0 while the package shipped 2.6.0.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

PATH_REPO_ROOT: Path = Path(__file__).resolve().parent.parent
REGEX_SEMVER: str = r"\d+\.\d+\.\d+"


def _read_pyproject_version() -> str:
    """Returns the [project] version from pyproject.toml."""
    dict_pyproject = tomllib.loads(
        (PATH_REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    # Normal return with the packaging source-of-truth version.
    return dict_pyproject["project"]["version"]


def _read_module_version() -> str:
    """Returns the __version__ constant from src/ai_api_unified/__version__.py."""
    str_module_text: str = (
        PATH_REPO_ROOT / "src" / "ai_api_unified" / "__version__.py"
    ).read_text(encoding="utf-8")
    match_version = re.search(
        rf'^__version__: str = "({REGEX_SEMVER})"', str_module_text, re.MULTILINE
    )
    assert match_version is not None, "__version__.py is missing the version constant"
    # Normal return with the runtime fallback version.
    return match_version.group(1)


def _read_readme_title_version() -> str:
    """Returns the version embedded in README.md's title heading."""
    str_first_line: str = (
        (PATH_REPO_ROOT / "README.md").read_text(encoding="utf-8").splitlines()[0]
    )
    match_version = re.search(rf"^# ai-api-unified ({REGEX_SEMVER})$", str_first_line)
    assert match_version is not None, (
        "README.md line 1 must be '# ai-api-unified X.Y.Z' "
        f"but was: {str_first_line!r}"
    )
    # Normal return with the README title version.
    return match_version.group(1)


def test_version_locations_agree() -> None:
    """All three version literals must be identical."""
    str_pyproject_version: str = _read_pyproject_version()
    str_module_version: str = _read_module_version()
    str_readme_version: str = _read_readme_title_version()

    assert str_pyproject_version == str_module_version, (
        f"pyproject.toml ({str_pyproject_version}) and __version__.py "
        f"({str_module_version}) disagree"
    )
    assert str_pyproject_version == str_readme_version, (
        f"pyproject.toml ({str_pyproject_version}) and the README.md title "
        f"({str_readme_version}) disagree"
    )


def test_version_is_semver_shaped() -> None:
    """The version must be plain MAJOR.MINOR.PATCH."""
    assert re.fullmatch(REGEX_SEMVER, _read_pyproject_version())
