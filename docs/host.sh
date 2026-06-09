#!/bin/bash
# Build the docs and serve them locally.
#   http://localhost:8000/        English
set -e
cd "$(dirname "$0")"
./build.sh
cd _build/html
python3 -m http.server 8000 --bind 0.0.0.0
