# Dense-X-Retrieval Pack

This LlamaPack creates a query engine that uses a `RecursiveRetriever` in llama-index to fetch nodes based on propoistions extracted from each node.

This follows the idea from the paper [Dense X Retrieval: What Retrieval Granularity Should We Use?](https://arxiv.org/abs/2312.06648).

From the paper, a proposition is described as:

```
Propositions are defined as atomic expressions within text, each encapsulating a distinct factoid and presented in a concise, self-contained natural language format.
```

We use the provided OpenAI prompt from their paper to generate propositions, which are then embedded and used to retrieve their parent node chunks.

**NOTE:** While the paper uses a fine-tuned model to extract propositions, it is unreleased at the time of writing. Currently, this pack uses the LLM to extract propositions, which can be expensive for large amounts of data.

## CLI Usage

You can download llamapacks directly using `llamaindex-cli`, which comes installed with the `llama-index` python package:

```bash
llamaindex-cli download-llamapack DenseXRetrievalPack --download-dir ./dense_pack
```

You can then inspect the files at `./dense_pack` and use them as a template for your own project!

## Code Usage

You can download the pack to a the `./dense_pack` directory:

```python
from llama_index.core import SimpleDirectoryReader
from llama_index.core.llama_pack import download_llama_pack

# download and install dependencies
DenseXRetrievalPack = download_llama_pack(
    "DenseXRetrievalPack", "./dense_pack"
)

documents = SimpleDirectoryReader("./data").load_data()

# uses the LLM to extract propositions from every document/node!
dense_pack = DenseXRetrievalPack(documents)

# for streaming
dense_pack = DenseXRetrievalPack(documents, streaming=True)
```

The `run()` function is a light wrapper around `query_engine.query()`.

```python
response = dense_pack.run("What can you tell me about LLMs?")

print(response)
```

for streaming:

The `run()` function is a light wrapper around `query_engine.query()`.

```python
stream_response = dense_pack.run("What can you tell me about LLMs?")

stream_response.print_response_stream()
```

See the [notebook on llama-hub](https://github.com/run-llama/llama-hub/blob/main/llama_hub/llama_packs/dense_x_retrieval/dense_x_retrieval.ipynb) for a full example.
