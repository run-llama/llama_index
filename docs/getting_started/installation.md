# Installation and Setup

## Installation from Pip

Install from pip:

```
pip install llama-index
```

**NOTE:** LlamaIndex may download and store local files for various packages (NLTK, HuggingFace, ...). Use the environment variable "LLAMA_INDEX_CACHE_DIR" to control where these files are saved.

If you prefer to install from source, see below.

## Important: OpenAI Environment Setup

By default, we use the OpenAI `gpt-3.5-turbo` model for text generation and `text-embedding-ada-002` for retrieval and embeddings. In order to use this, you must have an OPENAI_API_KEY set up as an environment variable.
You can obtain an API key by logging into your OpenAI account and [and creating a new API key](https://platform.openai.com/account/api-keys).

```{tip}
You can also [use one of many other available LLMs](/module_guides/models/llms/usage_custom.md). You may
need additional environment keys + tokens setup depending on the LLM provider.
```

## Local Model Setup

If you don't wish to use OpenAI, consider setting up a local LLM and embedding model in the service context.

A full guide to using and configuring LLMs available [here](/module_guides/models/llms.md).

A full guide to using and configuring embedding models is available [here](/module_guides/models/embeddings.md).

## Installation from Source

Git clone this repository: `git clone https://github.com/jerryjliu/llama_index.git`. Then do the following:

- [Install poetry](https://python-poetry.org/docs/#installation) - this will help you manage package dependencies
- `poetry shell` - this command creates a virtual environment, which keeps installed packages contained to this project
- `poetry install` - this will install the core package requirements
- (Optional) `poetry install --with dev,docs` - this will install all dependencies needed for most local development

## Optional Dependencies

By default LlamaIndex installs a core set of dependencies; we also provide a convenient way to install commonly-required optional dependencies. These are currently in three sets:

- `pip install llama-index[local_models]` installs tools useful for private LLMs, local inference, and HuggingFace models
- `pip install llama-index[postgres]` is useful if you are working with Postgres, PGVector or Supabase
- `pip install llama-index[query_tools]` gives you tools for hybrid search, structured outputs, and node post-processing
