#!/bin/bash
# Clean and rebuild documentation
cd "$(dirname "$0")"
make clean && make html
