# CHANGELOG — llama-index-llms-genai

## [v0.9.4]

### Fixed

- Raise a clear `ValueError` in `create_file_part` when a Vertex AI client is used with a file path going through the Gemini File API, replacing the cryptic `'This method is only supported in the Gemini Developer client.'` error from `google-genai` for `file_mode='fileapi'` and for `file_mode='hybrid'` with files larger than 20MB.

## [v0.9.3]

### Fixed

- Pass `DocumentBlock.title` as `display_name` when uploading files via the Google GenAI File API, fixing title mismatches between prompts and DocumentBlocks
