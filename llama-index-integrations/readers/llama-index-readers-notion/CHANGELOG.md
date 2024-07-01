# CHANGELOG

## [0.1.8] - 2024-07-01

### New Features

1. **Request with Retry Mechanism**: Added `_request_with_retry` method for retrying requests and handling rate limits.
2. **List Databases**: Added `list_databases` method to list all databases in the Notion workspace.

### Modifications

1. **\_read_block Method**: Now uses `_request_with_retry` for API requests.
2. **query_database Method**: Updated to use `_request_with_retry` and handle pagination.
3. **search Method**: Updated to use `_request_with_retry` and handle pagination.
4. **load_data Method**: Now supports loading from multiple databases via a list of `database_ids`.

### Bug Fixes

1. **Rate Limit Handling**: Improved error handling with exponential backoff in `_request_with_retry`.
2. **Pagination Handling**: Fixed issues in `query_database` and `search` methods for better pagination handling.

### Example Usage

1. **Main Block**: Added examples for listing databases within a workplace.

## [0.1.2] - 2024-02-13

- Add maintainers and keywords from library.json (llamahub)
