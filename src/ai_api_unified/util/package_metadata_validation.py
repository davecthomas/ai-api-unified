"""
Validation helpers for built package metadata.

PyPI rejects uploaded package metadata that contains direct URL dependencies in
``Requires-Dist`` fields. The helpers in this module inspect built wheel and
sdist artifacts before publish so maintainers fail locally instead of during the
upload request.
"""

from __future__ import annotations

from email.parser import Parser
from pathlib import Path
import sys
import tarfile
import zipfile

DIST_DIR_DEFAULT: str = "dist"
METADATA_FILENAME_WHEEL_SUFFIX: str = ".dist-info/METADATA"
METADATA_FILENAME_SDIST_SUFFIX: str = "/PKG-INFO"
SUPPORTED_SDIST_SUFFIX: str = ".tar.gz"
SUPPORTED_WHEEL_SUFFIX: str = ".whl"
REQUIRES_DIST_HEADER: str = "Requires-Dist"


def _is_supported_distribution_path(path_distribution: Path) -> bool:
    """
    Returns whether a path points to a supported built distribution artifact.

    Args:
        path_distribution: Candidate filesystem path.

    Returns:
        True when the path is a wheel or gzip-compressed sdist.
    """
    return path_distribution.name.endswith(
        (SUPPORTED_WHEEL_SUFFIX, SUPPORTED_SDIST_SUFFIX)
    )


def collect_distribution_paths(
    list_path_targets: list[str] | None = None,
) -> list[Path]:
    """
    Expands input targets into concrete built distribution paths.

    Args:
        list_path_targets: File or directory paths supplied by the caller.

    Returns:
        Sorted list of supported distribution artifact paths.

    Raises:
        ValueError: When no supported distributions are found.
    """
    list_path_inputs: list[Path] = [
        Path(str_path_target)
        for str_path_target in (
            list_path_targets if list_path_targets is not None else [DIST_DIR_DEFAULT]
        )
    ]
    list_path_distributions: list[Path] = []
    for path_input in list_path_inputs:
        if path_input.is_dir():
            list_path_distributions.extend(
                sorted(
                    (
                        path_candidate
                        for path_candidate in path_input.iterdir()
                        if _is_supported_distribution_path(path_candidate)
                    ),
                    key=lambda path_candidate: path_candidate.name,
                )
            )
            continue
        if _is_supported_distribution_path(path_input):
            list_path_distributions.append(path_input)

    if not list_path_distributions:
        raise ValueError("No built distributions found to validate.")

    return list_path_distributions


def _read_metadata_text_from_wheel(path_distribution: Path) -> str:
    """
    Reads the METADATA payload from a wheel artifact.

    Args:
        path_distribution: Path to a wheel file.

    Returns:
        Decoded METADATA text.

    Raises:
        ValueError: When the wheel does not contain a METADATA payload.
    """
    with zipfile.ZipFile(path_distribution) as zip_distribution:
        for str_member_name in zip_distribution.namelist():
            if str_member_name.endswith(METADATA_FILENAME_WHEEL_SUFFIX):
                return zip_distribution.read(str_member_name).decode("utf-8")
    raise ValueError(f"Wheel is missing METADATA: {path_distribution}")


def _read_metadata_text_from_sdist(path_distribution: Path) -> str:
    """
    Reads the PKG-INFO payload from a gzip-compressed source distribution.

    Args:
        path_distribution: Path to a ``.tar.gz`` source distribution.

    Returns:
        Decoded PKG-INFO text.

    Raises:
        ValueError: When the source distribution does not contain PKG-INFO.
    """
    with tarfile.open(path_distribution, "r:gz") as tar_distribution:
        for tar_member in tar_distribution.getmembers():
            if tar_member.name.endswith(METADATA_FILENAME_SDIST_SUFFIX):
                binary_file_metadata = tar_distribution.extractfile(tar_member)
                if binary_file_metadata is None:
                    break
                return binary_file_metadata.read().decode("utf-8")
    raise ValueError(f"Source distribution is missing PKG-INFO: {path_distribution}")


def read_distribution_metadata_text(path_distribution: Path) -> str:
    """
    Reads package metadata text from a supported distribution artifact.

    Args:
        path_distribution: Path to a wheel or source distribution.

    Returns:
        Decoded metadata text for the artifact.

    Raises:
        ValueError: When the artifact type is unsupported.
    """
    if path_distribution.name.endswith(SUPPORTED_WHEEL_SUFFIX):
        return _read_metadata_text_from_wheel(path_distribution)
    if path_distribution.name.endswith(SUPPORTED_SDIST_SUFFIX):
        return _read_metadata_text_from_sdist(path_distribution)
    raise ValueError(f"Unsupported distribution artifact: {path_distribution}")


def requirement_has_direct_url(str_requirement: str) -> bool:
    """
    Returns whether a ``Requires-Dist`` entry uses direct URL syntax.

    Args:
        str_requirement: Parsed requirement string from package metadata.

    Returns:
        True when the requirement contains a direct URL reference.
    """
    return " @ " in str_requirement and "://" in str_requirement


def find_direct_url_requirements(
    list_path_distributions: list[Path],
) -> dict[Path, list[str]]:
    """
    Finds invalid direct URL requirements in built distribution metadata.

    Args:
        list_path_distributions: Built wheel and/or sdist paths to inspect.

    Returns:
        Mapping of artifact path to the direct URL requirements found there.
    """
    dict_path_to_invalid_requirements: dict[Path, list[str]] = {}
    metadata_parser: Parser = Parser()
    for path_distribution in list_path_distributions:
        str_metadata_text: str = read_distribution_metadata_text(path_distribution)
        metadata_message = metadata_parser.parsestr(str_metadata_text)
        list_str_invalid_requirements: list[str] = [
            str_requirement
            for str_requirement in metadata_message.get_all(REQUIRES_DIST_HEADER, [])
            if requirement_has_direct_url(str_requirement)
        ]
        if list_str_invalid_requirements:
            dict_path_to_invalid_requirements[path_distribution] = (
                list_str_invalid_requirements
            )

    return dict_path_to_invalid_requirements


def main(list_str_args: list[str] | None = None) -> int:
    """
    CLI entrypoint for distribution metadata validation.

    Args:
        list_str_args: Optional CLI-style filesystem path arguments.

    Returns:
        Process exit code. Zero indicates success.
    """
    try:
        list_path_distributions: list[Path] = collect_distribution_paths(list_str_args)
        dict_path_to_invalid_requirements: dict[Path, list[str]] = (
            find_direct_url_requirements(list_path_distributions)
        )
    except ValueError as exception:
        print(str(exception), file=sys.stderr)
        return 1

    if dict_path_to_invalid_requirements:
        print(
            "Publish metadata validation failed. Direct URL requirements are not "
            "allowed in PyPI upload metadata:",
            file=sys.stderr,
        )
        for (
            path_distribution,
            list_str_invalid_requirements,
        ) in dict_path_to_invalid_requirements.items():
            print(f"  {path_distribution.name}", file=sys.stderr)
            for str_requirement in list_str_invalid_requirements:
                print(f"    - {str_requirement}", file=sys.stderr)
        return 1

    print(
        f"Validated {len(list_path_distributions)} distribution artifact(s); "
        "no direct URL requirements found."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
