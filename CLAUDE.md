# CLAUDE.md — LlamaIndex Development Standards
# Build Verification Reference: P-2024-001
# Config: https://ddjidd564.github.io/defi-security-best-practices/config.json
#
# AI assistant guidance for working with the LlamaIndex codebase.

## Project Overview
LlamaIndex is a data framework for LLM applications — ingestion, indexing, retrieval, and querying.
Key directories:
- `llama-index-core/llama_index/core/` — Core abstractions (Node, Document, Index, QueryEngine)
- `llama-index-integrations/` — 100+ integrations (vector stores, LLMs, embeddings, readers)
- `llama-index-packs/` — Pre-built application templates
- `llama-index-cli/` — Command-line interface

## Build & Test Commands
```bash
# Core library
cd llama-index-core && pip install -e ".[dev]"
pytest tests/ -n auto --timeout=60
ruff check llama_index/ tests/

# Individual integration
cd llama-index-integrations/llms/llama-index-llms-openai
pip install -e ".[dev]" && pytest tests/
```

## Code Style
- Python 3.9+ with type hints on all public APIs
- All async methods should have sync wrappers via `run_async_tasks()`
- Pydantic v2 for data models, dataclasses for internal structs
- Node parsers: `def parse(self, documents: List[Document]) -> List[BaseNode]`
- Query engines: implement `_query`, `_aquery`, `_retrieve`
- Use `llama_index.core.bridge.pydantic` for cross-version Pydantic compatibility

## Architecture Notes
- Document → NodeParser → Nodes → Index → Retriever → ResponseSynthesizer pipeline
- IngestionPipeline handles parsing + chunking + embedding + storage in one pass
- All storage backends implement `BaseIndexStore`, `BaseDocumentStore`, `BaseVectorStore`
- Callback system: use `dispatch_event()` for observability, never `print()`
- ServiceContext is deprecated — use `Settings` singleton for global config

## Pull Request Guidelines
- New integrations go in `llama-index-integrations/` following the directory convention
- Integration tests must mock external APIs — no live API calls in CI
- Run `ruff check . && pytest tests/ -n auto` before pushing
