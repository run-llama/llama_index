# CHANGELOG — llama-index-llms-genai

## [v0.9.6]

### Fixed

- Forward `thoughts_token_count` and `cached_content_token_count` from `usage_metadata` into `ChatResponse.additional_kwargs` so Gemini 2.5 thinking-token and prompt-cache usage is visible to `TokenCountingHandler` and OpenInference instrumentation (#19293).

## [v0.9.4]

### Changed

- Bump `google-genai` dependency to support v2.0 SDKs

## [v0.9.3]

### Fixed

- Pass `DocumentBlock.title` as `display_name` when uploading files via the Google GenAI File API, fixing title mismatches between prompts and DocumentBlocks
