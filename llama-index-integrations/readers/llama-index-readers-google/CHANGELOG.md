# CHANGELOG

## [0.3.1] - 2024-08-19

- Implemented ResourcesReaderMixin and FileSystemReaderMixin for extended functionality
- New methods: list_resources, get_resource_info, load_resource, read_file_content
- Comprehensive logging for better debugging and operational insights
- Detailed exception handling

## [0.2.4] - 2024-04-01

- Add support for additional params when initializing GoogleDriveReader

## [0.2.2] - 2024-03-26

- Add class name method for serialization

## [0.2.1] - 2024-03-26

- Allow passing credentials directly as a string
- Make the reader serializable
- Don't write credentials to disk in cloud mode

## [0.2.0] - 2024-03-26

- Use separate arg for service account key file, don't conflate client secrets with service account key
- Remove unused PyDrive dependency and code

## [0.1.5] - 2024-03-06

- Add missing README.md for all readers folder lost during the last migration from llamahub
- Add `query_string` in Google Drive Reader
- Some others minor fixes

## [0.1.2] - 2024-02-13

- Add maintainers and keywords from library.json (llamahub)
