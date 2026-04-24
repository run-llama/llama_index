# CHANGELOG

## [0.8.0] - 2026-04-10

### Added

- New `default_parsers.py` module with `Default*Parser` classes for all supported file types and a `get_default_parsers()` factory function
- `FileType` enum: new values `PAGE_HTML`, `XLSB`, `MSG`, `SVG`
- `CustomParserManager` is now always initialized - default parsers are merged with any `custom_parsers` overrides; `custom_parser_manager` is never `None`
- `custom_folder` is now accepted without `custom_parsers` (previously raised `ValueError`)
- `fail_on_error` is now enforced for attachment processing (was previously only enforced for page processing)
- New `[all]` optional install extra that installs all attachment parser dependencies at once

### Changed

- Confluence page HTML body now routed through `FileType.PAGE_HTML` via `CustomParserManager`; `PAGE_HTML` and `HTML` are independent and can be overridden separately via `custom_parsers`
- `process_attachment_callback` signature extended to 3 args: `(media_type: str, file_size: int, attachment_title: str)`
- `process_page()` no longer accepts a `text_maker` argument and always parses page HTML through `FileType.PAGE_HTML`
- All `process_*()` attachment methods now delegate to `custom_parser_manager`
- Attachment parser dependencies (`pytesseract`, `pdf2image`, `Pillow`, `docx2txt`, `python-pptx`, `pandas`, `openpyxl`, `pyxlsb`, `extract-msg`, `beautifulsoup4`, `svglib`, `reportlab`) moved from core to the `[all]` optional extra
- `pyproject.toml` dependency layout updated: explicit core dependency declarations and parser dependency grouping under `[all]`; `xlrd` removed and replaced by `openpyxl` (in `[all]`)
- `load_data()` error log messages now include the page ID
- `CustomParserManager.process_with_custom_parser` now returns `tuple[str, dict]` instead of `str`, propagating `metadata` from the parser's returned `Document`
- `process_page()` merges parser-supplied `metadata` with the default metadata keys (`title`, `page_id`, `status`, `url`); default keys always take precedence on conflicts, while any additional keys set by a custom parser are preserved in the final `Document`

### Removed

- `html_parser.py` / `HtmlTextParser` — replaced by `DefaultPageHtmlParser` in `default_parsers.py`
- `ValueError` that prevented using `custom_folder` without `custom_parsers`
- Core dependency on `xlrd`

### Fixed

- README: corrected `FileType` import path in custom parsers example (`readers.confluence_reader` → `llama_index.readers.confluence.event`)
- README: fixed syntax error in `page_ids` example (`"<page_id_3"` → `"<page_id_3>"`)
- README: removed invalid `include_children=True` from the `space_key` pagination example
- README: documented `folder_id` as the 5th mutually exclusive `load_data` query option
- README: documented `cloud` and `client_args` constructor parameters
- README: added Document Metadata section describing the `metadata` dict fields returned by `load_data`

## [0.1.8] - 2024-08-20

- Added observability events for ConfluenceReader
- Added support for custom parser for attachments
- Added callback for page and attachment processing

## [0.1.7] - 2024-07-23

- Sort order of authentication parameters and environment variables as:
  1. `oauth2`
  2. `api_token`
  3. `user_name` and `password`
  4. Environment variable `CONFLUENCE_API_TOKEN`
  5. Environment variable `CONFLUENCE_USERNAME` and `CONFLUENCE_PASSWORD`

## [0.1.2] - 2024-02-13

- Add maintainers and keywords from library.json (llamahub)

## [1.1.5] - 2024-06-18

- Add constructor args for auth parameters (@cmpaul)
