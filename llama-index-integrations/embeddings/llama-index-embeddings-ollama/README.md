# LlamaIndex Embeddings Integration: Ollama

The `llama-index-embeddings-ollama` package contains LlamaIndex integrations for generating embeddings using [Ollama](https://ollama.ai/), a tool for running large language models locally.

Ollama allows you to run embedding models on your local machine, providing privacy, cost savings, and the ability to work offline. This integration enables you to use Ollama's embedding models seamlessly with LlamaIndex's vector store and retrieval systems.

## Installation

To install the `llama-index-embeddings-ollama` package, run the following command:

```bash
pip install llama-index-embeddings-ollama
```

You'll also need to have Ollama installed and running on your machine. Visit [ollama.ai](https://ollama.ai/) to download and install Ollama.

## Prerequisites

Before using this integration, ensure you have:

1. **Ollama installed**: Download from [ollama.ai](https://ollama.ai/)
2. **Ollama running**: Start the Ollama service (usually runs on `http://localhost:11434` by default)
3. **An embedding model pulled**: Pull an embedding model using Ollama CLI:
   ```bash
   ollama pull nomic-embed-text
   # or
   ollama pull embeddinggemma
   ```

## Basic Usage

### Simple Embedding Generation

```python
from llama_index.embeddings.ollama import OllamaEmbedding

# Initialize the embedding model
embed_model = OllamaEmbedding(
    model_name="nomic-embed-text",  # or "embeddinggemma"
    base_url="http://localhost:11434",  # default Ollama URL
)

# Generate an embedding for a single text
text_embedding = embed_model.get_text_embedding("Hello, world!")
print(f"Embedding dimension: {len(text_embedding)}")

# Generate an embedding for a query
query_embedding = embed_model.get_query_embedding("What is AI?")
```

### Batch Embedding Generation

```python
# Generate embeddings for multiple texts at once
texts = [
    "The capital of France is Paris.",
    "Python is a programming language.",
    "Machine learning is a subset of AI.",
]

embeddings = embed_model.get_text_embeddings(texts)
print(f"Generated {len(embeddings)} embeddings")
```

## Integration with LlamaIndex

### Using with VectorStoreIndex

The most common use case is to integrate Ollama embeddings with LlamaIndex's vector store:

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.ollama import OllamaEmbedding

# Set the embedding model globally
Settings.embed_model = OllamaEmbedding(
    model_name="nomic-embed-text",
    base_url="http://localhost:11434",
)

# Load documents
documents = SimpleDirectoryReader("data").load_data()

# Create index with Ollama embeddings
index = VectorStoreIndex.from_documents(documents)

# Query the index
query_engine = index.as_query_engine()
response = query_engine.query("What is the main topic?")
print(response)
```

### Using with Custom LLM

You can combine Ollama embeddings with other LLMs (including Ollama LLMs):

```python
from llama_index.core import VectorStoreIndex, Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

# Set both LLM and embedding model
Settings.llm = Ollama(model="llama3.1", base_url="http://localhost:11434")
Settings.embed_model = OllamaEmbedding(
    model_name="nomic-embed-text",
    base_url="http://localhost:11434",
)

# Your documents and indexing code here...
```

## Configuration Options

The `OllamaEmbedding` class supports several configuration options:

```python
embed_model = OllamaEmbedding(
    model_name="nomic-embed-text",  # Required: Ollama model name
    base_url="http://localhost:11434",  # Optional: Ollama server URL (default: http://localhost:11434)
    embed_batch_size=10,  # Optional: Batch size for embeddings (default: 10)
    keep_alive="5m",  # Optional: How long to keep model in memory (default: "5m")
    query_instruction=None,  # Optional: Instruction to prepend to queries
    text_instruction=None,  # Optional: Instruction to prepend to text
    ollama_additional_kwargs={},  # Optional: Additional kwargs for Ollama API
    client_kwargs={},  # Optional: Additional kwargs for Ollama client
)
```

### Parameter Details

- **`model_name`** (required): The name of the Ollama embedding model to use (e.g., `"nomic-embed-text"`, `"embeddinggemma"`)
- **`base_url`** (optional): The base URL of your Ollama server. Defaults to `"http://localhost:11434"`
- **`embed_batch_size`** (optional): Number of texts to process in each batch. Must be between 1 and 2048. Defaults to 10
- **`keep_alive`** (optional): Controls how long the model stays loaded in memory after a request. Can be a duration string (e.g., `"5m"`, `"10s"`) or a number of seconds. Defaults to `"5m"`
- **`query_instruction`** (optional): Instruction text to prepend to query strings before embedding
- **`text_instruction`** (optional): Instruction text to prepend to document text before embedding
- **`ollama_additional_kwargs`** (optional): Additional keyword arguments to pass to the Ollama API
- **`client_kwargs`** (optional): Additional keyword arguments for the Ollama client (e.g., authentication headers)

## Using Instructions for Better Retrieval

Some embedding models benefit from prepending instructions to queries and documents. This can improve retrieval quality:

```python
embed_model = OllamaEmbedding(
    model_name="nomic-embed-text",
    query_instruction="Represent the question for retrieving supporting documents:",
    text_instruction="Represent the document for retrieval:",
)

# The instructions will be automatically prepended
query_embedding = embed_model.get_query_embedding("What is machine learning?")
# Internally processes: "Represent the question for retrieving supporting documents: What is machine learning?"

text_embedding = embed_model.get_text_embedding(
    "Machine learning is a method of data analysis."
)
# Internally processes: "Represent the document for retrieval: Machine learning is a method of data analysis."
```

## Async Usage

The integration supports asynchronous operations for better performance:

```python
import asyncio
from llama_index.embeddings.ollama import OllamaEmbedding

embed_model = OllamaEmbedding(model_name="nomic-embed-text")


async def main():
    # Async single embedding
    embedding = await embed_model.aget_text_embedding("Hello, world!")

    # Async batch embeddings
    embeddings = await embed_model.aget_text_embeddings(
        [
            "Text 1",
            "Text 2",
            "Text 3",
        ]
    )

    # Async query embedding
    query_embedding = await embed_model.aget_query_embedding("What is AI?")


asyncio.run(main())
```

## Remote Ollama Server

If you're running Ollama on a remote server, specify the `base_url`:

```python
embed_model = OllamaEmbedding(
    model_name="nomic-embed-text",
    base_url="http://your-remote-server:11434",
)
```

## Available Models

Popular embedding models available in Ollama include:

- **`nomic-embed-text`**: General-purpose embedding model
- **`embeddinggemma`**: Google's Gemma-based embedding model
- **`mxbai-embed-large`**: Large embedding model for better quality

Pull a model using:

```bash
ollama pull nomic-embed-text
```

## Examples

For more detailed examples, see the [Ollama Embeddings notebook](https://github.com/run-llama/llama_index/blob/main/docs/examples/embeddings/ollama_embedding.ipynb) in the LlamaIndex documentation.

## Troubleshooting

### Connection Errors

If you encounter connection errors, ensure:

1. Ollama is running: `ollama serve` or check the service status
2. The `base_url` matches your Ollama server address
3. The model is pulled: `ollama pull <model-name>`

### Model Not Found

If you get a "model not found" error:

1. List available models: `ollama list`
2. Pull the required model: `ollama pull <model-name>`
3. Verify the model name matches exactly in your code

## License

This package is licensed under the MIT License. See the LICENSE file for details.
