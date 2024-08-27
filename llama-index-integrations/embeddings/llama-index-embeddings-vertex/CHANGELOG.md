---

### CHANGELOG

**Version:** 0.1.1

#### Added

- **Model Name Handling in Embedding Requests:**
  - Updated `_get_embedding_request` to accept `model_name` as a parameter, enabling the function to handle models that either support or do not support the `task_type` parameter, such as `textembedding-gecko@001`.

#### Changed

- **Removed Redundant Imports:**
  - Credential Management: Supports both direct credentials and service account info for secure API access.

#### Fixed
- **Compatibility with Older Models:**
  - Fixed issues with models that do not support the `task_type` parameter by introducing `_UNSUPPORTED_TASK_TYPE_MODEL` set, ensuring compatibility with models like `textembedding-gecko@001`.

---
