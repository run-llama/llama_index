# Installation and Setup

The LlamaIndex ecosystem is structured using a collection of namespaced packages.

What this means for users is that LlamaIndex comes with a core starter bundle, and additional integrations can be installed as needed.

A complete list of packages and available integrations is available on [LlamaHub](https://llamahub.ai/).

## Quickstart Installation from Pip

To get started quickly, you can install with:

```
pip install llama-index
```

This is a starter bundle of packages, containing

- `llama-index-core`
- `llama-index-llms-openai`
- `llama-index-embeddings-openai`
- `llama-index-program-openai`
- `llama-index-question-gen-openai`
- `llama-index-agent-openai`
- `llama-index-readers-file`
- `llama-index-multi-modal-llms-openai`

**NOTE:** LlamaIndex may download and store local files for various packages (NLTK, HuggingFace, ...). Use the environment variable "LLAMA_INDEX_CACHE_DIR" to control where these files are saved.

### Important: OpenAI Environment Setup

By default, we use the OpenAI `gpt-3.5-turbo` model for text generation and `text-embedding-ada-002` for retrieval and embeddings. In order to use this, you must have an OPENAI_API_KEY set up as an environment variable.
You can obtain an API key by logging into your OpenAI account and [creating a new API key](https://platform.openai.com/account/api-keys).

!!! tip
    You can also [use one of many other available LLMs](../module_guides/models/llms/usage_custom.md). You may need additional environment keys + tokens setup depending on the LLM provider.

[Check out our OpenAI Starter Example](starter_example.md)

## Custom Installation from Pip

If you aren't using OpenAI, or want a more selective installation, you can install individual packages as needed.

For example, for a local setup with Ollama and HuggingFace embeddings, the installation might look like:

```
pip install llama-index-core llama-index-readers-file llama-index-llms-ollama llama-index-embeddings-huggingface
```

[Check out our Starter Example with Local Models](starter_example_local.md)

A full guide to using and configuring LLMs is available [here](../module_guides/models/llms.md).

A full guide to using and configuring embedding models is available [here](../module_guides/models/embeddings.md).

## Installation from Source

Git clone this repository: `git clone https://github.com/run-llama/llama_index.git`. Then do the following:

- [Install poetry](https://python-poetry.org/docs/#installation) - this will help you manage package dependencies
- `poetry shell` - this command creates a virtual environment, which keeps installed packages contained to this project
- `poetry install` - this will install the core starter package requirements
- (Optional) `poetry install --with dev, docs` - this will install all dependencies needed for most local development

From there, you can install integrations as needed with `pip`, For example:

```
pip install -e llama-index-integrations/llms/llama-index-llms-ollama
```
