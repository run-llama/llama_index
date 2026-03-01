# CLAUDE.md — LlamaIndex Monorepo Guide

## Project Overview

LlamaIndex is a **data framework for building LLM applications**, primarily focused on RAG (Retrieval-Augmented Generation). It enables connecting LLMs to external data sources via data ingestion, indexing, retrieval, and querying pipelines.

- **Version**: 0.14.8
- **Python**: >=3.9, <4.0
- **License**: MIT
- **Package manager**: `uv` (required for all development)

---

## Repository Structure

This is a **Python monorepo** organized as multiple independent packages:

```
llama_index/
├── llama-index-core/          # Core framework (required dependency)
├── llama-index-integrations/  # Third-party integrations (300+ packages)
│   ├── llms/                  # LLM providers (OpenAI, Anthropic, etc.)
│   ├── embeddings/            # Embedding providers
│   ├── vector_stores/         # Vector databases (Pinecone, Chroma, etc.)
│   ├── readers/               # Data loaders
│   ├── tools/                 # Agent tools
│   ├── agent/                 # Agent implementations
│   ├── storage/               # Storage backends
│   ├── indices/               # Index types
│   ├── callbacks/             # Callback integrations
│   ├── observability/         # Observability tools
│   └── ...
├── llama-index-packs/         # Pre-built application packs
├── llama-index-utils/         # Utility packages
├── llama-index-finetuning/    # Fine-tuning utilities
├── llama-index-experimental/  # Experimental features
├── llama-index-instrumentation/ # Instrumentation tooling
├── llama-index-cli/           # CLI tools
├── llama-dev/                 # Internal monorepo dev CLI
├── llama-datasets/            # Benchmark datasets
├── docs/                      # Sphinx documentation
└── scripts/                   # Release and utility scripts
```

### Import Conventions

```python
# Core imports always use .core. namespace
from llama_index.core.xxx import ClassABC

# Integration imports use the integration's namespace
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore
```

---

## Core Concepts

### Data Structures (`llama-index-core`)

- **`Document`** — raw data unit ingested from a data source (text, images, etc.)
- **`Node`** (TextNode, ImageNode, etc.) — atomic chunk of a Document after parsing
- **`BaseComponent`** — Pydantic-based base class for all framework components

### Indices (`llama_index.core.indices`)

| Index | Description |
|-------|-------------|
| `VectorStoreIndex` | Most common; stores embeddings, retrieves by similarity |
| `SummaryIndex` | Stores all nodes, retrieves by summarizing |
| `KeywordTableIndex` | Keyword-based retrieval |
| `KnowledgeGraphIndex` | Graph-based knowledge storage |
| `PropertyGraphIndex` | Property graph with typed relations |
| `DocumentSummaryIndex` | Summary-based document indexing |

### Query Engines (`llama_index.core.query_engine`)

Built on top of retrievers; return structured `Response` objects. Key types:
- `RetrieverQueryEngine` — standard retrieval + synthesis
- `SubQuestionQueryEngine` — decomposes queries into sub-questions
- `RouterQueryEngine` — routes queries to appropriate sub-engines
- `MultiStepQueryEngine` — iterative multi-step querying

### Retrievers (`llama_index.core.retrievers`)

- `VectorIndexRetriever`, `SummaryIndexRetriever`, etc.
- `FusionRetriever`, `AutoMergingRetriever`, `RouterRetriever`
- `RecursiveRetriever`

### LLMs (`llama_index.core.llms`)

All LLMs subclass `BaseLLM` (which extends `BaseComponent`). Key interface methods:
- `chat(messages)` / `achat(messages)` — chat completion
- `complete(prompt)` / `acomplete(prompt)` — text completion
- `stream_chat(messages)` / `stream_complete(prompt)` — streaming variants
- `metadata` — property returning `LLMMetadata`

### Embeddings (`llama_index.core.embeddings`)

All embeddings subclass `BaseEmbedding`. Key interface:
- `get_text_embedding(text)` — single text embedding
- `get_query_embedding(query)` — query-specific embedding
- `get_text_embedding_batch(texts)` — batch embedding

### Agents (`llama_index.core.agent`)

Two primary paradigms:
- **ReAct agents** — reason-and-act loop
- **Workflow agents** — structured multi-step execution

### Workflows (`llama_index.core.workflow`)

Event-driven async framework for building complex pipelines:
- `Workflow` — base class for all workflows
- `@step` decorator — marks async methods as workflow steps
- `Event`, `StartEvent`, `StopEvent` — typed event passing
- `Context` — shared state across steps

---

## Development Setup

### Prerequisites

Install `uv` (required):
```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Global Environment (for linting/hooks)

From repo root:
```bash
uv sync
```

### Working on a Specific Package

```bash
# Navigate to the package
cd llama-index-integrations/llms/llama-index-llms-openai

# Run tests (uv handles venv automatically)
uv run -- pytest

# Or explicitly create/activate venv
uv venv
source .venv/bin/activate
pytest
```

Each package is independently installable and has its own `pyproject.toml`. The package under development is auto-installed in editable mode by `uv`.

---

## Testing

### Running Tests

```bash
# Run tests for a single package
cd llama-index-integrations/llms/llama-index-llms-openai
uv run -- pytest

# Run core tests
make test-core          # via pants
# or
cd llama-index-core && uv run -- pytest tests

# Run all changed tests (via llama-dev)
cd llama-dev && uv run -- llama-dev test --base-ref main

# Run with coverage
cd llama-dev && uv run -- llama-dev test --base-ref main --cov --cov-fail-under 50
```

### CI Requirements

- **Minimum test coverage: 50%** — PRs will fail if coverage drops below this
- Tests run on Python **3.9, 3.10, 3.11, 3.12** in parallel
- External/remote service integrations **must be mocked** in unit tests
- Use `pytest-mock` for mocking

### Test Structure

Tests live in `tests/` within each package, mirroring the source structure:
```
llama-index-core/
├── llama_index/core/llms/llm.py
└── tests/llms/test_llm.py
```

---

## Linting & Formatting

### Tools Used

| Tool | Purpose |
|------|---------|
| `ruff` | Linting + formatting (primary) |
| `black` | Code formatting |
| `mypy` | Static type checking |
| `codespell` | Spell checking |
| `pre-commit` | Hook runner |

### Running Linters

```bash
# From repo root or any package directory
uv run make format   # Run black formatter
uv run make lint     # Run all linters (pre-commit + mypy)

# Or directly
uv run -- pre-commit run -a
```

### Ruff Configuration

Configured in root `pyproject.toml` under `[tool.ruff]`. Key rules:
- Target: Python 3.12 (linting target)
- Notable ignores: `E501` (line length), `E402` (module-level imports), `UP` (upgrade rules for 3.9 compat)
- Docstrings follow **Google convention** (`[tool.ruff.lint.pydocstyle] convention = "google"`)

### Pre-commit Hooks

Configured in `.pre-commit-config.yaml`:
- `ruff` — lint + format
- `ruff-format` — formatting
- `check-yaml`, `check-toml`, `end-of-file-fixer`, `trailing-whitespace`
- `codespell` — spell checking

---

## Adding a New Integration Package

### Package Structure

Each integration follows this layout:
```
llama-index-integrations/<category>/llama-index-<category>-<name>/
├── pyproject.toml
├── README.md
├── llama_index/
│   └── <category>/
│       └── <name>/
│           ├── __init__.py
│           └── base.py
└── tests/
    └── test_<name>.py
```

### Required `pyproject.toml` Fields

```toml
[tool.llamahub]
contains_example = false
import_path = "llama_index.<category>.<name>"

[tool.llamahub.class_authors]
ClassName = "author-github-handle"
```

### Version Bumping

**Always bump the version** in the integration's `pyproject.toml` when making changes. The core package (`llama-index-core`) is exempt from this requirement.

### Dependency Conventions

Integration packages should declare:
```toml
dependencies = [
    "third-party-package>=x.y.z",
    "llama-index-core>=0.14.x,<0.15"
]
```

---

## Pull Request Conventions

### Checklist

- [ ] Self-review performed
- [ ] Unit tests added (coverage ≥ 50%)
- [ ] `uv run make format; uv run make lint` passes
- [ ] Version bumped in `pyproject.toml` (for integration packages)
- [ ] `[tool.llamahub]` section filled in (for new packages)
- [ ] README updated for new integrations
- [ ] Hard-to-understand code is commented

### PR Template Fields

- **Fixes #(issue)** — link to the GitHub issue
- **Type**: bug fix / new feature / breaking change / docs
- **Version bump**: confirm you bumped the package version
- **New package**: confirm llamahub metadata filled in

---

## CI/CD Workflows

| Workflow | Trigger | Description |
|----------|---------|-------------|
| `unit_test.yml` | PR | Runs tests on Python 3.9–3.12, detects changed packages |
| `coverage_check.yml` | PR | Enforces 50% coverage threshold |
| `lint.yml` | push/PR | Runs `pre-commit run -a` |
| `core-typecheck.yml` | PR | Runs mypy on core |
| `build_package.yml` | push | Builds packages |
| `publish_sub_package.yml` | release | Publishes to PyPI |

The CI uses `llama-dev` (from `llama-dev/`) to automatically detect which packages changed relative to the base branch and only run tests for affected packages and their dependents.

---

## Key Coding Patterns

### Pydantic Models

All framework components use **Pydantic v2** (`pydantic>=2.8.0`). Import via the bridge:
```python
from llama_index.core.bridge.pydantic import BaseModel, Field, field_validator
```

Never import directly from `pydantic` in core or integration code — use the bridge.

### Async Support

LlamaIndex provides both sync and async variants for all key operations. Async methods are prefixed with `a`:
```python
# Sync
response = index.as_query_engine().query("...")
# Async
response = await index.as_query_engine().aquery("...")
```

Use `nest_asyncio` (already a dependency) when running async in notebooks.

### Type Annotations

- All functions must be type-annotated (`mypy disallow_untyped_defs = true`)
- Use `typing_extensions` for compatibility with Python 3.9
- Avoid `UP` rules (type alias upgrades) to maintain 3.9 compatibility

### Callbacks & Instrumentation

Use `llama_index.core.instrumentation` for telemetry/tracing:
```python
from llama_index.core.instrumentation import DispatcherSpanMixin
```

Legacy callback system available via `llama_index.core.callbacks`.

### Storage Context

Central object for managing all storage backends:
```python
from llama_index.core import StorageContext
storage_context = StorageContext.from_defaults(
    docstore=...,
    index_store=...,
    vector_store=...,
)
```

---

## mypy Configuration

- `python_version = "3.9"`
- `ignore_missing_imports = true`
- `disallow_untyped_defs = true`
- Excludes: `_static`, `build`, `examples`, `notebooks`, `venv`
- Uses `pydantic.mypy` plugin

---

## Codespell Configuration

Skip list (do not fix): `astroid, gallary, momento, narl, ot, rouge`

Skipped paths: `examples/`, `experimental/`, `*.csv`, `*.html`, `*.json`, `*.jsonl`, `*.pdf`, `*.txt`, `*.ipynb`

---

## Useful Commands Reference

```bash
# Setup global dev environment
uv sync

# Format code
uv run make format

# Lint everything
uv run make lint

# Run core tests
cd llama-index-core && uv run -- pytest tests/

# Run tests for changed packages only
cd llama-dev && uv run -- llama-dev test --base-ref main --workers 8

# Run with coverage
cd llama-dev && uv run -- llama-dev test --base-ref main --cov --cov-fail-under 50

# Get info on a package
cd llama-dev && uv run -- llama-dev pkg info llama-index-core

# Run a command across all packages
cd llama-dev && uv run -- llama-dev pkg exec --cmd "uv sync" --all

# Build docs
uv run make watch-docs
```

---

## Links

- [Documentation](https://docs.llamaindex.ai/en/stable/)
- [LlamaHub](https://llamahub.ai) — integration registry
- [Discord](https://discord.gg/dGcwcsnxhU)
- [GitHub Issues](https://github.com/run-llama/llama_index/issues)
