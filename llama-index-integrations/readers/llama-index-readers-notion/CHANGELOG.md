# CHANGELOG

## [0.7.1]

### Bug Fixes

- **`_read_block` pagination**: the next page of a block's children is now requested from the same block with the `start_cursor` query parameter. The `next_cursor` value was previously used as a block ID in the URL, which requested that block's own children instead, so pages and blocks with more than 100 children were silently truncated.
- **Table rows**: `table_row` blocks are now read (cell text joined with ` | `). Their text is in `cells`, not `rich_text`, so tables were dropped.
- **Mentions and equations**: rich text is read from `plain_text`, so page, user and date mentions and inline equations are no longer skipped.

## [0.1.9] - 2024-07-02

### Bug Fixes

- **Fixed `load_data` method**: Removed reassignment of `database_ids`, allowing the function to correctly check the else condition when `database_ids` is `None`.

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
