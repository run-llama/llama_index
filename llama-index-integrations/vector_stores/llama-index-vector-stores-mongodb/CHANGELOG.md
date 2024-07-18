## [0.1.7]

### Added

    - New feature in `MongoDBAtlasVectorSearch` to add `pipeline_id` at the top level of documents:
    - New constructor parameter `add_pipeline_id_top_level` (boolean) to enable/disable the feature.
    - New constructor parameter `pipeline_id_key` (string) to customize the field name for the top-level pipeline ID.

### Changed

    - Modified `add` method in `MongoDBAtlasVectorSearch` to support adding `pipeline_id` at the top level when enabled.

### Migration

    - Existing usage of `MongoDBAtlasVectorSearch` remains unchanged. To use the new feature, initialize the class with `add_pipeline_id_top_level=True`.
    - If you want to customize the field name for the top-level pipeline ID, use the `pipeline_id_key` parameter during initialization.
