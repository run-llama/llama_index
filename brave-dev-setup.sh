#!/usr/bin/env bash
# Setup script for Brave browser development environment
# Requirements:
#   - Node.js >= 24.11.1 (use nvm to install: nvm install 24)
#   - Git
#   - Network access to github.com and storage.googleapis.com
#
# The Chromium source download (npm run init) is very large (multiple GB)
# and requires access to Google's infrastructure endpoints.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BRAVE_DIR="${SCRIPT_DIR}/src/brave"

# Check Node.js version
REQUIRED_NODE_MAJOR=24
CURRENT_NODE_MAJOR=$(node -v | cut -d. -f1 | tr -d 'v')
if [ "$CURRENT_NODE_MAJOR" -lt "$REQUIRED_NODE_MAJOR" ]; then
  echo "Error: Node.js >= 24.11.1 is required (current: $(node -v))"
  echo "Install it with: nvm install 24 && nvm use 24"
  exit 1
fi

# Clone brave-core if not present
if [ ! -d "$BRAVE_DIR" ]; then
  echo "Cloning brave-core..."
  mkdir -p "${SCRIPT_DIR}/src"
  git clone https://github.com/brave/brave-core.git "$BRAVE_DIR"
else
  echo "brave-core already cloned at $BRAVE_DIR"
fi

# Install npm dependencies
echo "Installing npm dependencies..."
cd "$BRAVE_DIR"
npm install

# Initialize (downloads Chromium source - this is very large)
echo "Running npm run init (downloads Chromium source)..."
echo "This step requires network access to storage.googleapis.com"
npm run init
