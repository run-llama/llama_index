# LlamaIndex Callbacks Integration: Opik

```shell
pip install llama-index-callbacks-opik
```

This integration allows you to get one-click observability of your LlamaIndex RAG pipelines on [Opik](https://comet.com/site/products/opik/?utm_medium=docs&utm_source=llamaindex&utm_campaign=opik).

The simplest way to get started and try out Opik is to signup on our [cloud instance](https://comet.com/signup?from=llm?utm_medium=docs&utm_source=llamaindex&utm_campaign=opik).
You can then get you API key from the quickstart page or the user menu and start logging !

You can initialize globally using

```python
from llama_index.core import set_global_handler

# You should provide your OPIK API key and Workspace using the following environment variables:
# OPIK_API_KEY, OPIK_WORKSPACE
set_global_handler("opik")
```

or:

```python
from llama_index.core import Settings
from llama_index.core.callbacks import CallbackManager
from opik.integrations.llama_index import LlamaIndexCallbackHandler

opik_callback_handler = LlamaIndexCallbackHandler()
Settings.callback_manager = CallbackManager([opik_callback_handler])
```
