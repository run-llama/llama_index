# LlamaIndex Embeddings Integration: Nvidia Triton

This integration allows LlamaIndex to use embedding models hosted on a [Triton Inference Server Github](https://github.com/triton-inference-server/server).

## Usage:

```python
from llama_index.embeddings.nvidia_triton import NvidiaTritonEmbedding

embedding = NvidiaTritonEmbedding(
    model_name="text_embeddings",
    server_url="localhost:8000",
    client_kwargs={"ssl": False},
)

print(embedding.get_text_embedding("hello world"))
```

Parameters:

- `model_name`: the name of the embedding model.
- `server_url`: the URL to the Triton Inference Server, normally on the HTTP port.
- `client_kwargs`: additional arguments to be passed to the `tritonclient.http.InferenceServerClient` instance, such us timeouts, ssl, etc.
- `input_tensor_name`: the name of the tensor the embedding model expects the input to be. Default: `INPUT_TEXT`.
- `output_tensor_name`: the name of the tensor the embedding model will serve the output embedding to. Default: `OUTPUT_EMBEDDINGS`.
