#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "Cleaning up previous builds..."
rm -rf dist/ build/ *.egg-info/

echo "Building and publishing the package using poetry..."
poetry publish --build

echo "Successfully published!"
