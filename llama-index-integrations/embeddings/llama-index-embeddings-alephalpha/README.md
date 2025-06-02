# LlamaIndex Embeddings Integration: Aleph Alpha

This README provides an overview of integrating Aleph Alpha's semantic embeddings with LlamaIndex. Aleph Alpha's API enables the generation of semantic embeddings from text, which can be used for downstream tasks such as semantic similarity and models like classifiers.

## Features

- **Semantic Embeddings:** Generate embeddings for text prompts using Aleph Alpha models.
- **Model Selection:** Utilize the latest version of specified models for generating embeddings.
- **Representation Types:** Choose from `symmetric`, `document`, and `query` embeddings based on your use case.
- **Compression:** Option to compress embeddings to 128 dimensions for faster comparison.
- **Normalization:** Retrieve normalized embeddings to optimize cosine similarity calculations.

## Installation

```bash
pip install llama-index-embeddings-alephalpha
```

## Usage

```python
from llama_index.embeddings.alephalpha import AlephAlphaEmbedding
```

1. **Request Parameters:**

   - `model`: Model name (e.g., `luminous-base`). The latest model version is used.
   - `representation`: Type of embedding (`symmetric`, `document`, `query`).
   - `prompt`: Text or multimodal prompt to embed. Supports text strings or an array of multimodal items.
   - `compress_to_size`: Optional compression to 128 dimensions.
   - `normalize`: Set to `true` for normalized embeddings.

2. **Advanced Parameters:**
   - `hosting`: Datacenter processing option (`aleph-alpha` for maximal data privacy).
   - `contextual_control_threshold`, `control_log_additive`: Control attention parameters for advanced use cases.

## Response Structure

- `model_version`: Model name and version used for inference.
- `embedding`: List of floats representing the generated embedding.
- `num_tokens_prompt_total`: Total number of tokens in the input prompt.

## Example

See the [example notebook](../../../docs/examples/embeddings/alephalpha.ipynb) for a detailed walkthrough of using Aleph Alpha embeddings with LlamaIndex.

## API Documentation

For more detailed API documentation and available models, visit [Aleph Alpha's API Docs](https://docs.aleph-alpha.com/api/semantic-embed/).
