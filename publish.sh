#!/bin/bash

# ==============================================================================
# Publish Script for ai_api_unified
# ==============================================================================
# This script automates the publishing process to PyPI.
# It ensures that:
# 1. The git working directory is clean.
# 2. The user is reminded of the current version and prompts for confirmation.
# 3. Old build artifacts are removed.
# 4. The package is built locally and validated for publish-safe metadata.
# 5. The prevalidated package is published securely using Poetry.
# ==============================================================================

# Exit immediately if a command exits with a non-zero status
set -e
# Exit on undefined variables
set -u
# Pipeline returns the exit status of the first failed command
set -o pipefail

echo "========================================"
echo "   ai_api_unified Publishing Script"
echo "========================================"

# 1. Check for uncommitted changes
if ! git diff-index --quiet HEAD --; then
    echo "❌ Error: You have uncommitted changes in your git repository."
    echo "Please commit or stash your changes before publishing."
    exit 1
fi

# 2. Extract current version from pyproject.toml
CURRENT_VERSION=$(grep -E '^version = ' pyproject.toml | awk -F'"' '{print $2}')
echo "📦 Current version in pyproject.toml is: ${CURRENT_VERSION}"

echo ""
echo "Have you bumped the version in both 'pyproject.toml' and 'src/ai_api_unified/__version__.py'?"
read -p "Continue with publishing version ${CURRENT_VERSION}? (y/N): " confirm
if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
    echo "Publishing aborted by user."
    exit 0
fi

echo ""
echo "🧹 Cleaning up previous builds..."
rm -rf dist/ build/ *.egg-info/

echo "🚀 Building the package via Poetry..."
poetry build

echo "🔎 Validating built package metadata..."
PYTHONPATH=src poetry run python -m ai_api_unified.util.package_metadata_validation dist

echo "🚀 Publishing the validated package via Poetry..."
# This requires the PyPI token to be configured beforehand:
# poetry config pypi-token.pypi <your-pypi-token>
poetry publish

echo "✅ Successfully published version ${CURRENT_VERSION}!"
echo ""
echo "Don't forget to push a git tag for this release:"
echo "  git tag v${CURRENT_VERSION}"
echo "  git push origin v${CURRENT_VERSION}"
