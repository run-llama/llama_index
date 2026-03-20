# LlamaIndex Embeddings Integration: Snowflake Cortex

Generate text embeddings using [Snowflake Cortex](https://docs.snowflake.com/en/user-guide/snowflake-cortex/llm-functions) via the REST API.

Supports all Cortex embedding models including `snowflake-arctic-embed-m-v1.5` (768-dim) and `snowflake-arctic-embed-l-v2.0` (1024-dim).

## Installation

```bash
pip install llama-index-embeddings-cortex
```

## Usage

```python
from llama_index.embeddings.cortex import CortexEmbedding

embed_model = CortexEmbedding(
    model_name="snowflake-arctic-embed-m-v1.5",
    account="ORG_ID-ACCOUNT_ID",
    user="MY_USER",
    private_key_file="/path/to/rsa_key.p8",
)

# Single embedding
embedding = embed_model.get_text_embedding("Hello world")

# Batch embeddings
embeddings = embed_model.get_text_embedding_batch(["Hello", "World"])

# Use with LlamaIndex
from llama_index.core import Settings

Settings.embed_model = embed_model
```

## Authentication

The following authentication methods are supported (in order of precedence):

1. **Key-pair auth**: Pass `private_key_file`, `account`, and `user`
2. **JWT token**: Pass `jwt_token` (string or filepath)
3. **Snowpark Session**: Pass a `snowflake.snowpark.Session` object as `session`
4. **Environment variable**: Set `SNOWFLAKE_KEY_FILE`, `SNOWFLAKE_ACCOUNT`, and `SNOWFLAKE_USERNAME`
5. **SPCS auto-detect**: Automatically uses the default OAuth token in Snowpark Container Services

## Supported Models

| Model                              | Dimensions | Max Tokens |
| ---------------------------------- | ---------- | ---------- |
| `snowflake-arctic-embed-m-v1.5`    | 768        | 512        |
| `snowflake-arctic-embed-m`         | 768        | 512        |
| `e5-base-v2`                       | 768        | 512        |
| `snowflake-arctic-embed-l-v2.0`    | 1024       | 512        |
| `snowflake-arctic-embed-l-v2.0-8k` | 1024       | 8192       |
| `nv-embed-qa-4`                    | 1024       | 512        |
| `multilingual-e5-large`            | 1024       | 512        |
| `voyage-multilingual-2`            | 1024       | 32000      |
