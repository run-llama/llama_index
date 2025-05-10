# CHANGELOG

## [0.4.0]

- Add support for PUT, DELETE
- Allow configuring timeout!
- Change url to url_template and add path_params so that the LLM doesn't have to put URLs together itself (it often guesses wrong)
- Make meaning of query parameters argument clearer
- Permit POST, PATCH, etc. to have path and query parameters
- Fix incorrect use of empty dict as a default param
- In general, this should work better especially when combined with OpenAPI tool

## [0.1.2] - 2024-02-13

- Add maintainers and keywords from library.json (llamahub)
