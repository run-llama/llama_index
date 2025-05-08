# CHANGELOG

## [0.1.2] - 2024-02-13

- Add maintainers and keywords from library.json (llamahub)

## [x.x.x] - 2025-04-27

### Added

- **Extended `DatabaseReader`**
  - `schema` parameter supported for all connection patterns except when passing an existing `SQLDatabase`.
  - `metadata_cols` for per-column metadata extraction and optional key-renaming.
  - `exclude_text_cols` to omit selected columns from the `Document.text_resource` (replaces the deprecated `Document.text`).
  - `resource_id` callback for deterministic `id_` generation.
  - `lazy_load_data` generator for memory-efficient row streaming.
  - `aload_data` async wrapper (runs sync work in a thread with `asyncio.to_thread`).
- **Test suite**
  - New pytest module (`tests/test_database_reader.py`) covering:
    - basic loading,
    - metadata/exclusion behaviour,
    - custom `id_` callback,
    - lazy generator,
    - async wrapper.

### Changed

- Improved docstrings, adopting new standards for `Document`, noting the deprecated fields.
- `load_data` now delegates to `lazy_load_data`, reducing duplicate logic.
- Improved logging; warnings are emitted once per missing/duplicate column.
