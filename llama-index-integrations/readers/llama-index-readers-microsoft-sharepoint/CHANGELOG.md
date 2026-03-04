# CHANGELOG

## [0.8.0] - 2026-02-15

### Added

- Add pagination support for all Microsoft Graph API calls to handle large result sets

## [0.7.0] - 2026-01-28

### Added

- **SharePoint Site Pages Support**: New `SharePointType.PAGE` option to load SharePoint Site Pages as documents
  - `load_pages_data()`: Load all pages or a specific page from a SharePoint site
  - `list_pages()`: List all pages in a SharePoint site
  - `get_page_text()`: Get the text content of a specific page
  - `get_page_id_by_name()`: Find a page ID by its name
- **Instrumentation Events**: Added LlamaIndex instrumentation support with `DispatcherSpanMixin`
  - `TotalPagesToProcessEvent`: Emitted when total page count is determined
  - `PageDataFetchStartedEvent`: Emitted when page processing begins
  - `PageDataFetchCompletedEvent`: Emitted when a page is successfully processed
  - `PageSkippedEvent`: Emitted when a page is skipped (via callback)
  - `PageFailedEvent`: Emitted when page processing fails
- **New Parameters**:
  - `sharepoint_type`: Choose between `DRIVE` (files) or `PAGE` (site pages)
  - `sharepoint_host_name`: Host name for accessing sites with `Sites.Selected` permission
  - `sharepoint_relative_url`: Relative URL for accessing sites with `Sites.Selected` permission
  - `process_document_callback`: Callback function to filter pages before processing
  - `fail_on_error`: Control error handling behavior (raise or log and continue)
- **Sites.Selected Permission Support**: Use `sharepoint_host_name` and `sharepoint_relative_url` to access specific sites without tenant-wide permissions

## [0.5.1] - 2025-04-02

- Fix issue with folder path encoding when a file path contains special characters

## [0.1.7] - 2024-04-03

- Use recursive strategy by default for reading from a folder

## [0.1.6] - 2024-04-01

- Allow passing arguments for sitename and folder path during construction of reader

## [0.1.4] - 2024-03-26

- Make the reader serializable by inheriting from `BasePydanticReader` instead of `BaseReader`.

## [0.1.2] - 2024-02-13

- Add maintainers and keywords from library.json (llamahub)
