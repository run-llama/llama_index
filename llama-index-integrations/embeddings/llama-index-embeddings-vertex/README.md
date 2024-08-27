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

  - Credential Management: Supports both direct credentials and service account info for secure API access.

- **Model Name Handling in Embedding Requests:**
  - The `_get_embedding_request` function now accepts the `model_name` parameter, allowing it to manage models that do not support the `task_type` parameter, like `textembedding-gecko@001`.

### Example Usage

```python
from google.oauth2 import service_account
from llama_index.embeddings.vertex import VertexTextEmbedding

credentials = service_account.Credentials.from_service_account_file(
    "path/to/your/service-account.json"
)

embedding = VertexTextEmbedding(
    model_name="textembedding-gecko@003",
    project="your-project-id",
    location="your-region",
    credentials=credentials,
)
```

Alternatively, you can directly pass the required service account parameters:

```python
from llama_index.embeddings.vertex import VertexTextEmbedding

embedding = VertexTextEmbedding(
    model_name="textembedding-gecko@003",
    project="your-project-id",
    location="your-region",
    client_email="your-service-account-email",
    token_uri="your-token-uri",
    private_key_id="your-private-key-id",
    private_key="your-private-key",
)
```
