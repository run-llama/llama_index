# TextEmbed - Embedding Inference Server

Maintained by Keval Dekivadiya, TextEmbed is licensed under the [Apache-2.0 License](https://opensource.org/licenses/Apache-2.0).

TextEmbed is a high-throughput, low-latency REST API designed for serving vector embeddings. It supports a wide range of sentence-transformer models and frameworks, making it suitable for various applications in natural language processing.

## Features

- **High Throughput & Low Latency**: Designed to handle a large number of requests efficiently.
- **Flexible Model Support**: Works with various sentence-transformer models.
- **Scalable**: Easily integrates into larger systems and scales with demand.
- **Batch Processing**: Supports batch processing for better and faster inference.
- **OpenAI Compatible REST API Endpoint**: Provides an OpenAI compatible REST API endpoint.
- **Single Line Command Deployment**: Deploy multiple models via a single command for efficient deployment.
- **Support for Embedding Formats**: Supports binary, float16, and float32 embeddings formats for faster retrieval.

## Getting Started

### Prerequisites

Ensure you have Python 3.10 or higher installed. You will also need to install the required dependencies.

### Installation via PyPI

Install the required dependencies:

```bash
pip install -U textembed
```

### Start the TextEmbed Server

Start the TextEmbed server with your desired models:

```bash
python -m textembed.server --models sentence-transformers/all-MiniLM-L12-v2 --workers 4 --api-key TextEmbed
```

### Example Usage with llama-index

Here's a simple example to get you started with llama-index:

```python
from llama_index.embeddings.textembed import TextEmbedEmbedding

# Initialize the TextEmbedEmbedding class
embed = TextEmbedEmbedding(
    model_name="sentence-transformers/all-MiniLM-L12-v2",
    base_url="http://0.0.0.0:8000/v1",
    auth_token="TextEmbed",
)

# Get embeddings for a batch of texts
embeddings = embed.get_text_embedding_batch(
    [
        "It is raining cats and dogs here!",
        "India has a diverse cultural heritage.",
    ]
)

print(embeddings)
```

For more information, please read the [documentation](https://github.com/kevaldekivadiya2415/textembed/blob/main/docs/setup.md).
