# Cohere Citations Chat Engine Pack

Creates and runs a custom `VectorStoreIndexWithCitationsChat` -- which provides the chat engine with documents/citation mode.
See the documentation [here](https://docs.cohere.com/docs/retrieval-augmented-generation-rag) and [here](https://docs.cohere.com/docs/retrieval-augmented-generation-rag).

## CLI Usage

You can download llamapacks directly using `llamaindex-cli`, which comes installed with the `llama-index` python package:

```bash
llamaindex-cli download-llamapack CohereCitationChatEnginePack --download-dir ./cohere_citation_chat_pack
```

You can then inspect the files at `./cohere_citation_chat_pack` and use them as a template for your own project!

You can also directly install it if you don't want to look at/inspect the source code:

```bash
pip install llama-index-packs-cohere-citation-chat
```

## Code Usage

You can download the pack to the `./cohere_citation_chat_pack` directory:

```python
from llama_index.readers.web import SimpleWebPageReader
from llama_index.core.llama_pack import download_llama_pack

# download and install dependencies
CohereCitationChatEnginePack = download_llama_pack(
    "CohereCitationChatEnginePack", "./cohere_citation_chat_pack"
)
documents = SimpleWebPageReader().load_data(
    [
        "https://raw.githubusercontent.com/jerryjliu/llama_index/adb054429f642cc7bbfcb66d4c232e072325eeab/examples/paul_graham_essay/data/paul_graham_essay.txt"
    ]
)
cohere_citation_chat_pack = CohereCitationChatEnginePack(
    documents=documents, cohere_api_key="your-api-key"
)
chat_engine = cohere_citation_chat_pack.run()
response = chat_engine.chat("What can you tell me about LLMs?")

# print chat response
print(response)

# print documents
print(response.documents)

# print citations
print(response.citations)
```

See the [notebook on llama](https://github.com/run-llama/llama_index/blob/main/llama-index-packs/llama-index-packs-cohere-citation-chat/examples/cohere_citation_chat_example.ipynb) for a full example.
