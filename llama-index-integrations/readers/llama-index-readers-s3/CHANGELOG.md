# CHANGELOG

## [0.1.7] - 2024-04-12

- Fix wrong doc id when using default s3 endpoint

## [0.1.6] - 2024-04-11

- Use `None` as the default value for `s3_endpoint_url`

## [0.1.5] - 2024-03-27

- Update `README.md` to include installation instructions.

## [0.1.4] - 2024-03-18

- Refactor: Take advantage of `SimpleDirectoryReader` now supporting `fs` by using `s3fs` instead of downloading files to local disk.
- Make the reader serializable by inheriting from `BasePydanticReader` instead of `BaseReader`.

## [0.1.2] - 2024-02-13

- Add maintainers and keywords from library.json (llamahub)
