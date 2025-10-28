#!/bin/bash
# Build Sphinx documentation using the virtual environment

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Check if venv exists
if [ ! -d "$PROJECT_ROOT/venv" ]; then
    echo "Error: Virtual environment not found at $PROJECT_ROOT/venv"
    echo "Please run: python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Use the venv's sphinx-build
SPHINX_BUILD="$PROJECT_ROOT/venv/bin/sphinx-build"

if [ ! -f "$SPHINX_BUILD" ]; then
    echo "Error: sphinx-build not found in venv"
    echo "Please run: source venv/bin/activate && pip install -r docs/sphinx-requirements.txt"
    exit 1
fi

# Build the documentation
cd "$SCRIPT_DIR"
"$SPHINX_BUILD" -b html . _build/html

echo ""
echo "Documentation built successfully!"
echo "Open: $SCRIPT_DIR/_build/html/index.html"
