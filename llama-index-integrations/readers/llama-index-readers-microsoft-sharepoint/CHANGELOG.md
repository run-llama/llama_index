# CHANGELOG

## [0.7.0] - 2025-07-22

- Add support for custom file readers via `custom_parsers` and `custom_folder` arguments.
- Add event system for page/document processing (see `event.py` for event classes).
- Add `process_document_callback` for filtering/skipping documents.
- Add support for SharePoint page reading (`sharepoint_type="page"` and `page_name`).
- Improve error handling with `fail_on_error` option.
- Refactor and document all new features in README.
- Add comprehensive tests for all new features.
- BREAKING: Some constructor arguments and behaviors have changed to support new features.

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
