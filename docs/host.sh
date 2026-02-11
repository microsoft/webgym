#!/bin/bash
# Auto-rebuild and serve docs with live reload
cd "$(dirname "$0")"
sphinx-autobuild . ./_build/html --host 0.0.0.0 --port 8000 --open-browser