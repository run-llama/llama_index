# CHANGELOG

## [0.7.1] - 2026-04-04

- Added an injectable HTML parser interface for page content processing.
- Added support for supplying a custom HTML parser to `ConfluenceReader` during initialization.
- Exported the HTML parser base type (`HtmlParserBase`) as part of the public package API.

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
