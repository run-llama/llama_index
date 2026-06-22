# LlamaIndex Embeddings Integration: TwelveLabs

```bash
pip install llama-index-embeddings-twelvelabs
```

`TwelveLabsEmbedding` generates multimodal embeddings with [TwelveLabs](https://twelvelabs.io) **Marengo**. Marengo embeds text, images, audio, and video into a single shared vector space, so a text query and an image (or video) land in the same space and can be compared directly with cosine similarity — enabling cross-modal retrieval such as text-to-image search.

This integration exposes Marengo's text and image embeddings through LlamaIndex's `MultiModalEmbedding` interface. Set the `TWELVELABS_API_KEY` environment variable (or pass `api_key=...`). Get a key at [playground.twelvelabs.io](https://playground.twelvelabs.io).

## Usage

```python
from llama_index.embeddings.twelvelabs import TwelveLabsEmbedding

embed_model = TwelveLabsEmbedding()  # reads TWELVELABS_API_KEY from the environment

# Text
text_vector = embed_model.get_text_embedding("a cat playing piano")

# Image (a public URL or a local file path) — lands in the same space as text
image_vector = embed_model.get_image_embedding("https://example.com/cat.jpg")
image_vector = embed_model.get_image_embedding("/path/to/cat.jpg")
```

Because text and images share a vector space, you can index images and retrieve them with a text query (and vice versa) using LlamaIndex's multi-modal indices/retrievers.

### Options

- `api_key` — TwelveLabs API key (defaults to `TWELVELABS_API_KEY`).
- `model_name` — Marengo model (default `marengo3.0`).

### Note on video

Marengo also embeds full videos, but a video yields multiple time-segment vectors via an async task — not the single vector LlamaIndex's embedding interface expects — so video embedding is intentionally out of scope here. To turn a video into text for indexing, use [`llama-index-readers-twelvelabs`](../../readers/llama-index-readers-twelvelabs) (TwelveLabs Pegasus).
