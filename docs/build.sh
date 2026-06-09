#!/bin/bash
# Build the docs.
#   English -> _build/html
set -e
cd "$(dirname "$0")"
sphinx-build -b html . _build/html
echo "Built: _build/html"
