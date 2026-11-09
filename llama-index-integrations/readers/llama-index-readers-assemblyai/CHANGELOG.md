# CHANGELOG

## [0.7.0] - 2026-11-09

- Add `lazy_load_data` generator support and `extra_info` metadata propagation to `load_data`.
- Add support for loading transcripts by `transcript_id` using `assemblyai.Transcript.get_by_id`.
- Add `CHAPTERS` and `UTTERANCES` formats to `TranscriptFormat`.
- Preserve transcript metadata in `SUBTITLES_SRT` and `SUBTITLES_VTT`.
- Guard against `None` transcript text for silent or empty audio.
- Add unit test suite with mock coverage for all formats and transcript ID loading.

## [0.1.2] - 2024-02-13

- Add maintainers and keywords from library.json (llamahub)
