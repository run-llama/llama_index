# Changelog

## [v0.8.0]

### Changed

- Changed the default sparse embedding function from `BGEM3SparseEmbeddingFunction` to `BM25BuiltInFunction`.

## [v0.2.1]

### Changed

- Changed the default consistency level from "Strong" to "Session".

## [v0.1.23]

### Added

- Added a new parameter `collection_properties` to set collection properties.

## [v0.1.22]

### Added

- Introduced a new `IndexManagement` enum to control index creation behavior:
  - `NO_VALIDATION`: Skips index validation and creation.
  - `CREATE_IF_NOT_EXISTS`: Creates the index only if it doesn't exist (default behavior).
- Added a new `index_management` parameter to the `MilvusVectorStore` constructor, allowing users to specify the desired index management strategy.

### Changed

- Updated the collection creation and index management logic in `_create_hybrid_index` to ensure proper sequence of operations.
- Refactored index creation logic to respect the new `index_management` setting.

### Fixed

- Resolved an issue where the collection object was potentially being recreated after index creation, which could lead to inconsistencies.
- Ensured that the collection is created before any index operations are attempted in hybrid mode.

### Improved

- Streamlined the process of collection creation and index management for hybrid (dense and sparse) vector operations.
- Provided more control over index creation behavior through the `index_management` parameter.

### Developer Notes

- The `_create_index_if_required` method now checks the `index_management` setting before proceeding with index creation or validation.
- The `_create_index_if_required` method now passes `self.collection_name` to `_create_hybrid_index` when in sparse mode.
- No changes to the existing public API of `MilvusVectorStore` were made; the `index_management` parameter is an addition to the constructor.

### Upgrade Notes

- This change is backwards compatible. Existing code using `MilvusVectorStore` without specifying `index_management` will default to the previous behavior (`CREATE_IF_NOT_EXISTS`).
- Users can now optionally specify an `index_management` strategy when initializing `MilvusVectorStore` to control index creation behavior.
