# tests/conftest.py
#
# Pytest auto-discovers this file; keep it alongside your tests:
#
# project-root/
# ├── pyproject.toml
# ├── src/
# │   └── ai_api_unified/
# └── tests/
#     ├── conftest.py         ← pytest auto-loads this
#     ├── test_ai_api.py
#     ├── test_google_gemini.py
#     ├── test_google_gemini_nonmock.py
#     ├── test_model_switch_nonmock.py
#     ├── test_openai_us_endpoint.py
#     └── test_voice_nonmock.py  ← live TTS; requires provider credentials

import os
import sys

import pytest
from dotenv import load_dotenv

# Load .env into process environment so poetry-run pytest sees it
load_dotenv()

AWS_PROFILE_ENV_VAR: str = "AWS_PROFILE"
DEFAULT_AWS_PROFILE_NAME: str = "radlibs-awspower"
AWS_SDK_LOAD_CONFIG_ENV_VAR: str = "AWS_SDK_LOAD_CONFIG"
AWS_SDK_LOAD_CONFIG_ENABLED: str = "1"
# Placeholders only. They let botocore build a client for the mocked suite on a
# machine with no AWS configuration; no request is ever signed with them.
DICT_DUMMY_AWS_CREDENTIALS: dict[str, str] = {
    "AWS_ACCESS_KEY_ID": "testing",
    "AWS_SECRET_ACCESS_KEY": "testing",
    "AWS_SESSION_TOKEN": "testing",
    "AWS_DEFAULT_REGION": "us-east-1",
}
# Placeholders only, for engines that gate on a key being present before the
# patched SDK client is built. Never used to authenticate.
DICT_PLACEHOLDER_PROVIDER_KEYS: dict[str, str] = {
    "GOOGLE_GEMINI_API_KEY": "testing-not-a-real-key",
}
# Applied per test file, never process-wide. Some tests in *_nonmock.py carry
# no `nonmock` marker and reach the live API; they skip only because no key is
# configured, so exporting a placeholder globally would switch them on and send
# real requests with an invalid key. Only files listed here get the placeholder.
FROZENSET_FILES_NEEDING_PLACEHOLDER_KEYS: frozenset[str] = frozenset(
    {"test_google_gemini.py"}
)


def _aws_profile_is_configured(profile_name: str) -> bool:
    """
    Reports whether the named profile exists in this machine's AWS configuration.

    Args:
        profile_name: Profile to look for, e.g. the repository default.

    Returns:
        True when botocore can see the profile, False when it cannot or when
        boto3 is not installed.
    """
    try:
        import botocore.session
    except ImportError:
        # Normal return: without the bedrock extra there is no boto3 to configure.
        return False
    try:
        return profile_name in botocore.session.Session().available_profiles
    except Exception:
        # A malformed or unreadable AWS config is not this test suite's problem.
        return False


def ensure_aws_test_environment() -> None:
    """
    Ensure pytest sessions inherit AWS configuration the Bedrock tests can use.

    Prefers an explicit AWS_PROFILE, then the repository default when that
    profile actually exists on this machine. Falls back to static dummy
    credentials so the mocked suite stays hermetic: CI and a fresh clone have
    no AWS config, and forcing a profile name there makes botocore raise
    ProfileNotFound before a mocked test can run. Live tests in *_nonmock.py
    are excluded from mocked runs and supply their own real credentials.
    """
    if not os.environ.get(AWS_SDK_LOAD_CONFIG_ENV_VAR):
        os.environ[AWS_SDK_LOAD_CONFIG_ENV_VAR] = AWS_SDK_LOAD_CONFIG_ENABLED
    if (
        AWS_PROFILE_ENV_VAR in os.environ
        and not os.environ[AWS_PROFILE_ENV_VAR].strip()
    ):
        # An empty AWS_PROFILE is worse than an absent one: botocore reads it as
        # a profile named "" and raises ProfileNotFound before any test runs.
        del os.environ[AWS_PROFILE_ENV_VAR]
    if os.environ.get(AWS_PROFILE_ENV_VAR):
        # Normal return: the caller chose a profile and it wins.
        return
    if _aws_profile_is_configured(DEFAULT_AWS_PROFILE_NAME):
        os.environ[AWS_PROFILE_ENV_VAR] = DEFAULT_AWS_PROFILE_NAME
        # Normal return with the repository default profile selected.
        return
    for str_var_name, str_placeholder in DICT_DUMMY_AWS_CREDENTIALS.items():
        os.environ.setdefault(str_var_name, str_placeholder)


ensure_aws_test_environment()


# 1. Determine the project root (one level up from this tests/ directory)
project_root: str = os.path.dirname(os.path.dirname(__file__))

# 2. Compute the absolute path to the src/ directory containing our package
src_dir: str = os.path.join(project_root, "src")

# 3. If src/ isn’t already on sys.path, insert it at the front.
#    That way, `import ai_api_unified` resolves to your local code.
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)


def configure_presidio_test_log_noise_filter() -> None:
    """
    Installs Presidio warning-noise filtering for the full pytest process.

    Args:
        None

    Returns:
        None after ensuring test logging uses the shared Presidio noise filter.
    """
    # Import lazily to satisfy lint import-order requirements in pytest bootstrap.
    from ai_api_unified.middleware.impl.presidio_log_control import (
        configure_presidio_log_noise_filter,
    )

    configure_presidio_log_noise_filter()
    # Normal return after test log-noise filter setup.
    return None


configure_presidio_test_log_noise_filter()


# ============================================================================
# AI AGENT REMINDER — TEST SELECTION POLICY
#
# 483+ tests is too many to run on every edit. During development, run only
# the areas impacted by your change:
#
#     poetry run python scripts/run_impacted_tests.py        # auto from git diff
#     poetry run pytest -m "area_engine_openai and not nonmock"   # by hand
#
# The FULL mocked regression suite (poetry run pytest -q -m "not nonmock")
# is REQUIRED before tagging or publishing a release; publish.sh enforces it.
# Area markers are applied automatically from tests/area_map.py — every new
# test file MUST be mapped there or collection fails below.
# ============================================================================
tests_dir: str = os.path.dirname(__file__)
if tests_dir not in sys.path:
    sys.path.insert(0, tests_dir)

from area_map import (  # noqa: E402
    AREA_MARKER_PREFIX,
    AREAS,
    areas_for_test_file,
)


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "nonmock: tests that call live provider APIs and require credentials",
    )
    # Loop over areas so every area marker is registered for strict marker use.
    for str_area in AREAS:
        config.addinivalue_line(
            "markers",
            f"{AREA_MARKER_PREFIX}{str_area}: tests exercising the "
            f"'{str_area}' code area (auto-applied from tests/area_map.py)",
        )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """
    Auto-applies area markers from tests/area_map.py to every collected test.

    Fails collection for unmapped test files so impact-based selection can
    never silently skip a new test file.
    """
    list_unmapped_files: list[str] = []
    # Loop over collected items so each carries its file's area markers.
    for item in items:
        str_basename: str = os.path.basename(str(item.fspath))
        tuple_areas = areas_for_test_file(str_basename)
        if tuple_areas is None:
            if str_basename not in list_unmapped_files:
                list_unmapped_files.append(str_basename)
            continue
        for str_area in tuple_areas:
            item.add_marker(getattr(pytest.mark, f"{AREA_MARKER_PREFIX}{str_area}"))
    if list_unmapped_files:
        raise pytest.UsageError(
            "Test files missing from tests/area_map.py DICT_TEST_FILE_AREAS: "
            f"{', '.join(sorted(list_unmapped_files))}. Add each file to the "
            "map (this keeps impact-based test selection complete)."
        )


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--aiprovider",
        action="store",
        # default="openai",
        default="google-gemini",
        help="AI provider to use (default: google-gemini)",
    )
    parser.addoption(
        "--embedmodel",
        action="store",
        # default="text-embedding-3-small",
        default="gemini-embedding-001",
        help="Embedding model to use (default: gemini-embedding-001)",
    )
    parser.addoption(
        "--llmmodel",
        action="store",
        default="gemini-3.5-flash",
        help="LLM model to use (default: gemini-2.5-flash)",
    )


@pytest.fixture(scope="session")
def aiprovider(request: pytest.FixtureRequest) -> str:
    # Either form works: "aiprovider" or "--aiprovider"
    return request.config.getoption("aiprovider")


@pytest.fixture(scope="session")
def embedmodel(request: pytest.FixtureRequest) -> str:
    return request.config.getoption("embedmodel")


@pytest.fixture(scope="session")
def llmmodel(request: pytest.FixtureRequest) -> str:
    return request.config.getoption("llmmodel")


@pytest.fixture(autouse=True)
def placeholder_provider_keys(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Supplies placeholder provider keys to the mocked test files that need them.

    `ai_google_base` reads GOOGLE_GEMINI_API_KEY through its own EnvSettings,
    which a test patching EnvSettings in the completions module does not cover.
    Seven fully mocked tests therefore failed on any machine without a real key,
    CI and a fresh clone included, despite patching the SDK and sending nothing.

    Scoped per file on purpose. monkeypatch reverts after each test, and a real
    key already in the environment always wins.
    """
    if os.path.basename(str(request.node.fspath)) not in (
        FROZENSET_FILES_NEEDING_PLACEHOLDER_KEYS
    ):
        # Normal return: this file reaches the network or needs no key.
        return None
    for str_var_name, str_placeholder in DICT_PLACEHOLDER_PROVIDER_KEYS.items():
        if not os.environ.get(str_var_name):
            monkeypatch.setenv(str_var_name, str_placeholder)
    # Normal return after seeding placeholders for this test.
    return None
