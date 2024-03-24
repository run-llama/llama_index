# Fuzzy Citation Query Engine Pack

Creates and runs a `CustomQueryEngine` -- `FuzzCitationQueryEngine` -- which post-processes response objects to identify source sentences using fuzzy matching.

The identified sentences are available in the `response.metadata` dictionary, containing a mapping of `(response_sentence, source_chunk)` -> `{"start_char_idx": idx, "end_char_idx": idx, "node" node}`. The start/end idxs represent the character indexes in the node text that the source chunk comes from.

The fuzzy matching uses `fuzz.ratio()` to compare sentences. The default threshold score is 50.

## CLI Usage

You can download llamapacks directly using `llamaindex-cli`, which comes installed with the `llama-index` python package:

```bash
llamaindex-cli download-llamapack FuzzyCitationEnginePack --download-dir ./fuzzy_citation_pack
```

You can then inspect the files at `./fuzzy_citation_pack` and use them as a template for your own project!

## Code Usage

You can download the pack to a the `./fuzzy_citation_pack` directory:

```python
from llama_index.core import Document, VectorStoreIndex
from llama_index.core.llama_pack import download_llama_pack

# download and install dependencies
FuzzyCitationEnginePack = download_llama_pack(
    "FuzzyCitationEnginePack", "./fuzzy_citation_pack"
)

index = VectorStoreIndex.from_documents([Document.example()])
query_engine = index.as_query_engine()

fuzzy_engine = FuzzyCitationEnginePack(query_engine, threshold=50)
```

The `run()` function is a light wrapper around `query_engine.query()`. The response will have metadata attached to it indicating the fuzzy citations.

```python
response = fuzzy_engine.run("What can you tell me about LLMs?")

# print source sentences
print(response.metadata.keys())

# print full source sentence info
print(response.metadata)
```

See the [notebook on llama-hub](https://github.com/run-llama/llama-hub/blob/main/llama_hub/llama_packs/fuzzy_citation/fuzzy_citation_example.ipynb) for a full example.
