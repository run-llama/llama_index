# CHANGELOG — llama-index-llms-genai

## [v0.9.4]

### Changed

- Bump `google-genai` dependency to support v2.0 SDKs

## [v0.9.3]

### Fixed

- Pass `DocumentBlock.title` as `display_name` when uploading files via the Google GenAI File API, fixing title mismatches between prompts and DocumentBlocks
