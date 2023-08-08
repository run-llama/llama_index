# Installation and Setup

### Installation from Pip

You can simply do:
```
pip install llama-index
```

### Installation from Source
Git clone this repository: `git clone https://github.com/jerryjliu/llama_index.git`. Then do:

- `pip install -e .` if you want to do an editable install (you can modify source files) of just the package itself.
- `pip install -r requirements.txt` if you want to install optional dependencies + dependencies used for development (e.g. unit testing).


### OpenAI Environment Setup

By default, we use the OpenAI `gpt-3.5-turbo` model for text generation and `text-embedding-ada-002` for retrieval and embeddings. In order to use this, you must have an OPENAI_API_KEY setup.
You can register an API key by logging into [OpenAI's page and creating a new API token](https://beta.openai.com/account/api-keys).

```{tip}
You can also [customize the underlying LLM](/core_modules/model_modules/llms/usage_custom.md). You may
need additional environment keys + tokens setup depending on the LLM provider.
```

### Local Environment Setup

If you don't wish to use OpenAI, the environment will automatically fallback to using `LlamaCPP` and `llama2-chat-13B` for text generation and `sentence-transformers/all-mpnet-base-v2` for retrieval and embeddings. This models will all run locally.

In order to use `LlamaCPP`, follow the installation guide [here](/examples/llm/llama_2_llama_cpp.ipynb). You'll need to install the `llama-cpp-python` package, preferably compiled to support your GPU.

In order to use the local embeddings, simply run `pip install sentence-transformers`.
