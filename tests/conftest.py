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


def ensure_aws_test_environment() -> None:
    """
    Ensure pytest sessions inherit AWS SSO configuration commonly required by Bedrock tests.

    Running pytest without these settings can launch boto3 with anonymous credentials,
    reproducing NoCredentialsError in local debug sessions.
    """
    if not os.environ.get(AWS_PROFILE_ENV_VAR):
        os.environ[AWS_PROFILE_ENV_VAR] = DEFAULT_AWS_PROFILE_NAME
    if not os.environ.get(AWS_SDK_LOAD_CONFIG_ENV_VAR):
        os.environ[AWS_SDK_LOAD_CONFIG_ENV_VAR] = AWS_SDK_LOAD_CONFIG_ENABLED


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


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "nonmock: tests that call live provider APIs and require credentials",
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
        default="gemini-2.5-flash",
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
