# LlamaIndex Embeddings Integration: Vertex

Implements Vertex AI Embeddings Models:

| Model                                | Release Date      |
| ------------------------------------ | ----------------- |
| textembedding-gecko@003              | December 12, 2023 |
| textembedding-gecko@002              | November 2, 2023  |
| textembedding-gecko-multilingual@001 | November 2, 2023  |
| textembedding-gecko@001              | June 7, 2023      |
| multimodalembedding                  |                   |

**Note**: Currently Vertex AI does not support async on `multimodalembedding`.
Otherwise, `VertexTextEmbedding` supports async interface.

---

### New Features

- **Flexible Credential Handling:**

  - The `_process_credentials` function now supports credentials provided as:
    - **JSON String**: `credentials = '{"type": "service_account", ...}'`
    - **Dictionary**: `credentials = {"type": "service_account", ...}`
    - **`service_account.Credentials` object**: Directly pass the credentials object.

  This ensures that credentials are correctly processed, regardless of the input format, and are used to initialize Vertex AI.

- **Model Name Handling in Embedding Requests:**
  - The `_get_embedding_request` function now accepts the `model_name` parameter, allowing it to manage models that do not support the `task_type` parameter, like `textembedding-gecko@001`.

### Example Usage

```python
from llama_index.embeddings.vertex import (
    VertexTextEmbedding,
    VertexEmbeddingMode,
)

# Example using a JSON string for credentials
json_credentials = (
    '{"type": "service_account", "project_id": "your-project", ...}'
)
text_embedder = VertexTextEmbedding(
    model_name="textembedding-gecko@003",
    project="your-gcp-project",
    location="your-gcp-location",
    credentials=json_credentials,
    embed_mode=VertexEmbeddingMode.RETRIEVAL_MODE,
)

embeddings = text_embedder.get_text_embedding("Hello World!")
```
