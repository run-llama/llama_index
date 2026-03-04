# CHANGELOG — llama-index-vector-stores-opensearch

## [1.0.0]

- Changed engine default from deprecated `nmslib` (since version 3.0.0) to `faiss`
- Allow custom index configuration for Opensearch
- Updated opensearch-py to 3.x
- Bugfix: Use same event loop to close async client

## [0.5.5]

- Added support for IS_EMPTY Filter Operator in case OpensearchVectorClient

## [0.4.1]

- Added ability to create OpensearchVectorClient with custom os_async_client (like os_client)

## [0.2.2]

- Fixed issue where Opensearch Serverless does not support painless scripting so handling the case where is_aoss is set and using knn_score script instead.

## [0.2.1]

- Refresh Opensearch index after delete operation to reflect the change for future searches

## [0.1.14]

- Adds support for full MetadataFilters (all operators and nested filters)
- Removes necessity to prefix filter keys with "metadata."

## [0.1.2]

- Adds OpensearchVectorClient as top-level import

## [0.1.1]

- Fixes strict equality in dependency of llama-index-core
