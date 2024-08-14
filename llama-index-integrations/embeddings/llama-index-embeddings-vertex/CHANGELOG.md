---

### CHANGELOG

**Version:** 0.1.1

#### Added
- **Credential Processing Enhancements:**
  - Introduced `_process_credentials` function to handle credentials provided as strings, dictionaries, or `service_account.Credentials` objects. This allows for flexible credential input formats.

- **Model Name Handling in Embedding Requests:**
  - Updated `_get_embedding_request` to accept `model_name` as a parameter, enabling the function to handle models that either support or do not support the `task_type` parameter, such as `textembedding-gecko@001`.

#### Changed
- **VertexTextEmbedding Initialization:**
  - Updated `VertexTextEmbedding` class constructor to use `_process_credentials` for handling credentials, streamlining initialization with different credential formats.

- **Removed Redundant Imports:**
  - Removed redundant imports and streamlined the import statements.

#### Fixed
- **Compatibility with Older Models:**
  - Fixed issues with models that do not support the `task_type` parameter by introducing `_UNSUPPORTED_TASK_TYPE_MODEL` set, ensuring compatibility with models like `textembedding-gecko@001`.

---
