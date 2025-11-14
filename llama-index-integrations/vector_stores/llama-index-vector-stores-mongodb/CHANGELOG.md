# Changelog

## [v0.1.8]

### Changed

- Refactored `filters_to_mql` function for handling metadata key and error handling
- Improved `map_lc_mql_filter_operators` for better type safety

### Fixed

- Edge case handling in `filters_to_mql` function
- Potential issues with filter operator mapping

### Improved

- Type safety and static analysis support

## [v0.1.9] - Unreleased

### Performance

- Filters are now applied at the database query stage (`$vectorSearch` filter parameter, `$search` compound filter) instead of post-processing results in Python.
- Skips building advanced clauses when they would be no-ops (no values and missing disabled).

### Added

- Implemented `FilterOperator.IS_EMPTY` semantics for MongoDB integration (Vector + Atlas Search). Field is treated as matching when missing, `null`, or an empty array. MQL translation now expands to a structural `$or`; Atlas Search translation builds nested compound clauses.
- Added helper `_mql_clause` to encapsulate single-filter MQL construction.
- Added `search_filter` parameter to `fulltext_search_stage()` for clarity (deprecates `filter` parameter for Atlas Search context).

### Tests

- Added unit tests covering `IS_EMPTY` standalone, combined with `IN` under OR conditions, and Atlas Search compound translation.
- Added integration tests for DEFAULT mode filter pre-application: EQ operator, IN operator, compound AND filters, compound OR filters.
- Added integration tests for hybrid search with filters (vector bias, text bias, OR combinations, AND contradictions).

### Fixed

- Corrected fulltext search index definition (removed incorrect array wrapping in field type mapping).
- Empty filter dictionaries are now properly omitted from query stages instead of being passed as empty objects.

### Changed

- Unified filter construction in `base.py` - both MQL and Atlas Search filters are now built once before mode-specific query branching.
- `filters_to_search_filter()` renamed to `filters_to_atlas_search_compound()` for clarity about its Atlas Search-specific purpose.

### Internal

- Reordered operator handling in `filters_to_atlas_search_compound` to group negatives, structural emptiness, then range/in/equals for consistency.
- Clarified that `IS_EMPTY` is structurally handled and intentionally omitted from `map_lc_mql_filter_operators`.
