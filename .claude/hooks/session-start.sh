#!/bin/bash
set -euo pipefail

# Only run in remote (Claude Code on the web) environments
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

# Enable async mode so the session starts immediately while deps install
echo '{"async": true, "asyncTimeout": 300000}'

XORTRON_PKG_DIR="${CLAUDE_PROJECT_DIR}/llama-index-integrations/llms/llama-index-llms-xortron"

# Install the Xortron package in editable mode with runtime deps
pip install -e "${XORTRON_PKG_DIR}" --quiet

# Install dev/test/lint tooling
pip install pytest pytest-asyncio ruff black mypy httpx --quiet

# Expose the package dir for convenience
echo "export XORTRON_PKG_DIR=\"${XORTRON_PKG_DIR}\"" >> "$CLAUDE_ENV_FILE"
