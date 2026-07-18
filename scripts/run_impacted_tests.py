#!/usr/bin/env python3
# ============================================================================
# AI AGENT REMINDER — TEST SELECTION POLICY
#
# This is the development-loop test runner: it maps your git diff onto test
# areas (tests/area_map.py) and runs only the impacted tests. Use it instead
# of the full suite while iterating:
#
#     poetry run python scripts/run_impacted_tests.py              # diff vs main
#     poetry run python scripts/run_impacted_tests.py --dry-run    # show plan
#     poetry run python scripts/run_impacted_tests.py --base HEAD~1
#
# The FULL mocked regression suite is REQUIRED before tagging or publishing
# a release (publish.sh enforces it):
#
#     poetry run pytest -q -m "not nonmock"
# ============================================================================
"""
Impact-based test selection: git diff -> areas -> pytest marker expression.

Load-bearing changes (ai_base.py, the factory/registry, conftest, pyproject)
escalate to the full suite. Changed test files always run directly. Paths
with no test impact (docs, scripts, ADRs, memory shards) trigger nothing.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))

from area_map import (  # noqa: E402
    ALL_AREAS,
    AREA_MARKER_PREFIX,
    areas_for_test_file,
    classify_changed_path,
)


def _resolve_default_base() -> str:
    """
    Resolves the default diff base: the first remote's primary branch.
    """
    str_remote: str = (
        subprocess.run(["git", "remote"], capture_output=True, text=True, check=True)
        .stdout.split("\n")[0]
        .strip()
    )
    if not str_remote:
        # Early return with a local fallback when the repo has no remote.
        return "main"
    completed = subprocess.run(
        ["git", "symbolic-ref", "--short", f"refs/remotes/{str_remote}/HEAD"],
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        # Early return with the conventional primary branch name.
        return f"{str_remote}/main"
    # Normal return with the remote primary branch ref.
    return completed.stdout.strip()


def _changed_paths(str_base: str) -> list[str]:
    """
    Returns changed paths versus the base, including uncommitted changes.
    """
    set_paths: set[str] = set()
    for list_command in (
        ["git", "diff", "--name-only", f"{str_base}...HEAD"],
        ["git", "diff", "--name-only", "HEAD"],
        ["git", "diff", "--name-only", "--cached"],
    ):
        completed = subprocess.run(
            list_command, capture_output=True, text=True, check=True
        )
        set_paths.update(
            str_line.strip()
            for str_line in completed.stdout.split("\n")
            if str_line.strip()
        )
    # Normal return with the sorted union of committed and working-tree changes.
    return sorted(set_paths)


def build_plan(
    list_changed_paths: list[str],
) -> tuple[bool, set[str], list[str], list[str]]:
    """
    Classifies changed paths into a test plan.

    Returns:
        Tuple of (run_full_suite, impacted areas, changed test files,
        paths with no test impact).
    """
    bool_full_suite: bool = False
    set_areas: set[str] = set()
    list_test_files: list[str] = []
    list_no_impact: list[str] = []
    # Loop over changed paths so each contributes its mapped impact.
    for str_path in list_changed_paths:
        impact = classify_changed_path(str_path)
        if impact is None:
            list_no_impact.append(str_path)
        elif impact == ALL_AREAS:
            bool_full_suite = True
        elif isinstance(impact, str):
            # A changed test file: fold its own areas into the marker
            # expression (positional paths would restrict collection and
            # silently drop area selection). An unmapped file runs directly
            # so collection surfaces the map-this-file error.
            str_basename: str = impact.rsplit("/", 1)[-1]
            tuple_file_areas = areas_for_test_file(str_basename)
            if tuple_file_areas is not None:
                set_areas.update(tuple_file_areas)
            else:
                list_test_files.append(impact)
        else:
            set_areas.update(impact)
    # Normal return with the classified plan.
    return bool_full_suite, set_areas, list_test_files, list_no_impact


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run only the tests impacted by the current git diff."
    )
    parser.add_argument(
        "--base",
        default=None,
        help="Diff base (default: the remote primary branch).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the selection plan without running pytest.",
    )
    args = parser.parse_args()

    str_base: str = args.base or _resolve_default_base()
    list_changed_paths: list[str] = _changed_paths(str_base)
    if not list_changed_paths:
        print(f"No changes versus {str_base}; nothing to run.")
        return 0

    bool_full_suite, set_areas, list_test_files, list_no_impact = build_plan(
        list_changed_paths
    )

    print(f"Diff base: {str_base}")
    print(f"Changed paths: {len(list_changed_paths)}")
    if list_no_impact:
        print(f"No test impact ({len(list_no_impact)}): {', '.join(list_no_impact)}")

    list_pytest_args: list[str]
    if bool_full_suite:
        print(
            "Load-bearing change detected -> FULL mocked regression suite. "
            "(Also required before any release; see publish.sh.)"
        )
        list_pytest_args = ["-m", "not nonmock"]
    elif not set_areas and not list_test_files:
        print("No impacted test areas; nothing to run.")
        return 0
    elif list_test_files and set_areas:
        # Unmapped test files alongside area impacts: escalate to the full
        # suite so the map-this-file collection error is not lost.
        print(
            "Unmapped test files changed alongside source areas "
            f"({', '.join(list_test_files)}) -> FULL mocked regression suite."
        )
        list_pytest_args = ["-m", "not nonmock"]
    elif list_test_files:
        # Unmapped changed test files run directly; collection will raise the
        # add-to-area-map error for them.
        list_pytest_args = ["-m", "not nonmock", *list_test_files]
        print(f"Selection -> unmapped test files: {', '.join(list_test_files)}")
    else:
        str_marker_expression: str = " or ".join(
            f"{AREA_MARKER_PREFIX}{str_area}" for str_area in sorted(set_areas)
        )
        list_pytest_args = ["-m", f"({str_marker_expression}) and not nonmock"]
        print(f"Selection -> areas: {', '.join(sorted(set_areas))}")

    list_command: list[str] = ["pytest", "-q", *list_pytest_args]
    print("Running:", " ".join(list_command))
    if args.dry_run:
        return 0
    completed = subprocess.run(["poetry", "run", *list_command])
    # Normal return with pytest's exit status.
    return completed.returncode


if __name__ == "__main__":
    sys.exit(main())
