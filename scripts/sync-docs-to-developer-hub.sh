#!/usr/bin/env bash
set -euo pipefail

# Syncs docs and API reference from this repo to the developer hub (docs site) repo.
# Can be run locally or from CI.
#
# Usage:
#   ./scripts/sync-docs-to-developer-hub.sh /path/to/developer-hub-repo [--skip-api-docs]
#
# What it syncs:
#   1. Framework docs (markdown):
#      docs/src/content/docs/framework/**/*.{md,mdx,yml,png,jpg,jpeg}
#        -> <docs-repo>/src/content/docs/python/framework/
#
#   2. API reference (built HTML via mkdocs):
#      Builds mkdocs, then syncs output
#        -> <docs-repo>/api-reference/python/framework/
#
# Excluded from framework docs:
#   examples/**, api_reference/**, CONTRIBUTING.md, DOCS_README.md

DOCS_REPO="${1:?Usage: $0 /path/to/developer-hub-repo [--skip-api-docs]}"
SKIP_API_DOCS=false
for arg in "${@:2}"; do
  case "$arg" in
    --skip-api-docs) SKIP_API_DOCS=true ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# --- 1. Sync framework docs (markdown) ---

SOURCE_DIR="$REPO_ROOT/docs/src/content/docs/framework"
DEST_DIR="$DOCS_REPO/src/content/docs/python/framework"

if [ ! -d "$SOURCE_DIR" ]; then
  echo "Error: Source docs directory not found: $SOURCE_DIR"
  exit 1
fi

if [ ! -d "$DOCS_REPO" ]; then
  echo "Error: Docs repo not found: $DOCS_REPO"
  exit 1
fi

echo "=== Syncing framework docs ==="
echo "  from: $SOURCE_DIR"
echo "  to:   $DEST_DIR"

mkdir -p "$DEST_DIR"

# rsync processes rules in order — specific excludes before generic includes
rsync -av --delete \
  --exclude='examples/***' \
  --exclude='api_reference/***' \
  --exclude='CONTRIBUTING.md' \
  --exclude='DOCS_README.md' \
  --exclude='_static/***' \
  --include='*/' \
  --include='*.md' \
  --include='*.mdx' \
  --include='*.yml' \
  --include='*.png' \
  --include='*.jpg' \
  --include='*.jpeg' \
  --exclude='*' \
  "$SOURCE_DIR/" "$DEST_DIR/"

# _static may contain non-standard file types (svg, js, css, etc.)
if [ -d "$SOURCE_DIR/_static" ]; then
  echo "Syncing _static directory..."
  rsync -av --delete "$SOURCE_DIR/_static/" "$DEST_DIR/_static/"
fi

echo "Framework docs sync complete."

# --- 2. Build and sync API reference (HTML) ---

if [ "$SKIP_API_DOCS" = true ]; then
  echo "=== Skipping API docs (--skip-api-docs) ==="
  exit 0
fi

echo ""
echo "=== Building API reference ==="

MKDOCS_CONFIG="$REPO_ROOT/docs/api_reference/mkdocs.yml"
API_DOCS_BUILD_DIR="$REPO_ROOT/.build/api-docs-output"
API_DOCS_DEST_DIR="$DOCS_REPO/api-reference/python/framework"

if [ ! -f "$MKDOCS_CONFIG" ]; then
  echo "Error: mkdocs config not found: $MKDOCS_CONFIG"
  exit 1
fi

# Build mkdocs — use uv run --with to install mkdocs and plugins on-the-fly
# without polluting the main project dependencies
MKDOCS_DEPS="mkdocs>=1.6.1,mkdocs-material>=9.6.14,mkdocstrings[python]>=0.29.1,mkdocs-click>=0.9.0,mkdocs-include-dir-to-nav>=1.2.0,mkdocs-render-swagger-plugin>=0.1.2,mkdocs-github-admonitions-plugin>=0.0.3,griffe-fieldz>=0.2.1"

echo "Running mkdocs build..."
cd "$REPO_ROOT"
uv run \
  --with "$MKDOCS_DEPS" \
  mkdocs build -f "$MKDOCS_CONFIG" -d "$API_DOCS_BUILD_DIR"

echo "Syncing API docs to $API_DOCS_DEST_DIR"
mkdir -p "$API_DOCS_DEST_DIR"
rsync -av --delete "$API_DOCS_BUILD_DIR/" "$API_DOCS_DEST_DIR/"

echo "API reference sync complete."
