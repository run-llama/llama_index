# CHANGELOG

## [0.1.8]

### Added

- Implemented ResourcesReaderMixin and FileSystemReaderMixin for extended functionality
- New methods: list_resources, get_resource_info, load_resource, read_file_content
- Comprehensive logging for better debugging and operational insights
- Detailed exception handling for all GCSReader methods
- Comprehensive docstrings for GCSReader class and all its methods

### Changed

- Refactored GCSReader to use SimpleDirectoryReader for file operations
- Improved authentication error handling in GCSReader
- Enhanced `get_resource_info` method to return more detailed GCS object information
- Updated `load_data` method to use SimpleDirectoryReader
- Refactored code for better readability and maintainability

## [0.1.0] - 2024-03-25

- Add maintainers and keywords from library.json (llamahub)
