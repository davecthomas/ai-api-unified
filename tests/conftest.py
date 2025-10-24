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
import pytest
import sys
from dotenv import load_dotenv

# Load .env into process environment so poetry-run pytest sees it
load_dotenv()

BEDROCK_DEPENDENCIES_AVAILABLE: bool = False
try:
    from ai_api_unified.ai_bedrock_base import (
        AIBedrockBase,
    )

    BEDROCK_DEPENDENCIES_AVAILABLE = True
except ImportError:
    BEDROCK_DEPENDENCIES_AVAILABLE = False

if BEDROCK_DEPENDENCIES_AVAILABLE:
    AWS_PROFILE_ENV_VAR: str = "AWS_PROFILE"
    DEFAULT_AWS_PROFILE_NAME: str = "not-configured"
    AWS_SDK_LOAD_CONFIG_ENV_VAR: str = "AWS_SDK_LOAD_CONFIG"
    AWS_SDK_LOAD_CONFIG_ENABLED: str = "1"

    def ensure_aws_test_environment() -> None:
        """
        Ensure pytest sessions inherit the AWS SSO configuration that Bedrock clients require.

        Running pytest without these settings launches boto3 with anonymous credentials which
        reproduces the NoCredentialsError seen in VS Code's pytest debugger. Explicitly populate
        the values so every test invocation matches the working VS Code configuration.
        """
        if not os.environ.get(AWS_PROFILE_ENV_VAR):
            os.environ[AWS_PROFILE_ENV_VAR] = DEFAULT_AWS_PROFILE_NAME
        if not os.environ.get(AWS_SDK_LOAD_CONFIG_ENV_VAR):
            os.environ[AWS_SDK_LOAD_CONFIG_ENV_VAR] = AWS_SDK_LOAD_CONFIG_ENABLED

    ensure_aws_test_environment()


# 1. Determine the project root (one level up from this tests/ directory)
project_root = os.path.dirname(os.path.dirname(__file__))

# 2. Compute the absolute path to the src/ directory containing our package
src_dir = os.path.join(project_root, "src")

# 3. If src/ isn’t already on sys.path, insert it at the front.
#    That way, `import ai_api_unified` resolves to your local code.
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "nonmock: tests that call live provider APIs and require credentials",
    )


def pytest_addoption(parser):
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
        default="gpt-4o-mini",
        # default="gemini-2.5-flash",
        help="LLM model to use (default: gpt-4o-mini)",
    )


@pytest.fixture(scope="session")
def aiprovider(request):
    # Either form works: "aiprovider" or "--aiprovider"
    return request.config.getoption("aiprovider")


@pytest.fixture(scope="session")
def embedmodel(request):
    return request.config.getoption("embedmodel")


@pytest.fixture(scope="session")
def llmmodel(request):
    return request.config.getoption("llmmodel")
