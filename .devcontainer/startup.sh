#!/bin/bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv
uv sync -vv --frozen
uv pip install --no-cache-dir marimo
