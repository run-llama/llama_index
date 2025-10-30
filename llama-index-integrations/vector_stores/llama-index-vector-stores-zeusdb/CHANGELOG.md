<!-- markdownlint-disable MD024 -->
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.1.3] - 2025-10-28

### Added

- Added `.venv` to the `mypy` exclude list to improve compatibility with local development environments.

### Changed

- Bumped `llama-index-core` to `>=0.14.6` to match the latest LlamaIndex release (Oct 26, 2025).
- Updated `llama-index-llms-openai` development dependency to `>=0.6.6`.
- Updated `pyproject.toml` build includes for proper source distribution packaging (relative paths under `[tool.hatch.build.targets.sdist]`).

---

## [0.1.2] - 2025-10-16

### Changed

- **Async API naming**: Renamed `aadd()` to `async_add()` to align with LlamaIndex standard async method naming conventions. The adapter now uses the official LlamaIndex async API (`async_add`, `aquery`, `adelete`, `adelete_nodes`, `aclear`) for consistency across the ecosystem.
- Updated async examples to use standard LlamaIndex async method names instead of custom aliases.
- Updated documentation strings to reference `async_add()` instead of `aadd()`.

---

## [0.1.1] - 2025-10-14

### Added

- Delete examples - Added `examples/delete_examples.py` demonstrating supported deletion operations and workarounds.
- Test coverage - Added tests for ID-based deletion and proper error handling for unsupported operations.
- Addeed Product **Quantization (PQ) support** - Full support for memory-efficient vector compression with automatic training
- **Persistence Support**: Complete save/load functionality for ZeusDB indexes
  - Save indexes to disk with `save_index(path)`
  - Load indexes from disk with `load_index(path)`
  - Preserves vectors, metadata, HNSW graph structure, and quantization configuration
  - Directory-based format (.zdb) with JSON metadata and binary data files
  - Cross-platform compatibility for sharing indexes between systems
  - Added comprehensive persistence examples (`examples/persistence_examples.py`)
- **Async Support**: Full asynchronous operation support for non-blocking workflows
  - `aadd()` - Add nodes asynchronously
  - `aquery()` - Query asynchronously  
  - `adelete_nodes()` - Delete nodes by IDs asynchronously
  - Thread-offloaded async wrappers using `asyncio.to_thread()`
- **MMR (Maximal Marginal Relevance) Search**: Diversity-focused retrieval for comprehensive results
  - Balance relevance and diversity with `mmr_lambda` parameter (0.0-1.0)
  - Control candidate pool size with `fetch_k` parameter
  - Prevents redundant/similar results in search responses
  - Perfect for RAG applications, research, and multi-perspective retrieval
- Added comprehensive async examples (`examples/async_examples.py`)
- Added MMR examples (`examples/mmr_examples.py`)

### Changed

- Filter format alignment — Updated _filters_to_zeusdb() to produce a flat dictionary with implicit AND (e.g., { "key": value, "other": { "op": value } }) instead of a nested {"and": [...]} structure, matching the Rust implementation.
- Test infrastructure — Updated _match_filter() in the test fake from the nested format to the flat format to reflect production behavior.

### Fixed

- Filter translation for metadata queries - _filters_to_zeusdb() now emits the flat format expected by the ZeusDB backend. Single filters and AND combinations are handled correctly. The previous nested format could cause filtered queries to return zero results.
- Deletion behavior - Correctly implemented ID-based deletion using `remove_point()`. Delete operations now properly remove vectors from the index and update vector counts.

---

## [0.1.0] - 2025-09-19

### Added

- Initial release of ZeusDB vector database integration for LlamaIndex.
- Support for connecting LlamaIndex’s RAG framework with ZeusDB for high-performance retrieval.
- Trusted Publishing workflow for PyPI releases via GitHub Actions.
- Build validation workflow to check distributions without publishing.
- Documentation updates including project description and setup instructions.

---

## [Unreleased]

### Added
<!-- Add new features here -->

### Changed
<!-- Add changed behavior here -->

### Fixed
<!-- Add bug fixes here -->

### Removed
<!-- Add removals/deprecations here -->

---
