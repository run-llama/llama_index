# CHANGELOG

## [0.1.5] - 2024-06-28

### Added

- Added `get_channel_ids` method to `SlackReader` class to fetch channel IDs based on names and regex patterns.
- Implemented `_is_regex` method to determine if a pattern is a valid regex.
- Implemented `_list_channels` method to fetch all channels (public and private) from Slack.
- Implemented `_filter_channels` method to filter channels based on provided names and regex patterns.

### Fixed

- Improved error handling for `not_in_channel` error in `SlackReader`. The reader now properly logs the error and stops retrying, preventing indefinite loops.
- Enhanced `rate-limiting` handling by implementing a default retry interval, ensuring better control over retries.

## [0.1.2] - 2024-02-13

- Add maintainers and keywords from library.json (llamahub)
