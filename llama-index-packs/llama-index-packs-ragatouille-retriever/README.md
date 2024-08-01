# RAGatouille Retriever Pack

RAGatouille is a [cool library](https://github.com/bclavie/RAGatouille) that lets you use e.g. ColBERT and other SOTA retrieval models in your RAG pipeline. You can use it to either run inference on ColBERT, or use it to train/fine-tune models.

This LlamaPack shows you an easy way to bundle RAGatouille into your RAG pipeline. We use RAGatouille to index a corpus of documents (by default using colbertv2.0), and then we combine it with LlamaIndex query modules to synthesize an answer with an LLM.

A full notebook guide can be found [here](https://github.com/run-llama/llama-hub/blob/main/llama_hub/llama_packs/ragatouille_retriever/ragatouille_retriever.ipynb).

## CLI Usage

You can download llamapacks directly using `llamaindex-cli`, which comes installed with the `llama-index` python package:

```bash
llamaindex-cli download-llamapack RAGatouilleRetrieverPack --download-dir ./ragatouille_pack
```

You can then inspect the files at `./` and use them as a template for your own project!

## Code Usage

You can download the pack to a `./ragatouille_pack` directory:

```python
from llama_index.core.llama_pack import download_llama_pack

# download and install dependencies
RAGatouilleRetrieverPack = download_llama_pack(
    "RAGatouilleRetrieverPack", "./ragatouille_pack"
)
```

From here, you can use the pack, or inspect and modify the pack in `./ragatouille_pack`.

Then, you can set up the pack like so:

```python
# create the pack
ragatouille_pack = RAGatouilleRetrieverPack(
    docs,  # List[Document]
    llm=OpenAI(model="gpt-3.5-turbo"),
    index_name="my_index",
    top_k=5,
)
```

The `run()` function is a light wrapper around `query_engine.query`.

```python
response = ragatouille_pack.run("How does ColBERTv2 compare to BERT")
```

You can also use modules individually.

```python
from llama_index.core.response.notebook_utils import display_source_node

retriever = ragatouille_pack.get_modules()["retriever"]
nodes = retriever.retrieve("How does ColBERTv2 compare with BERT?")

for node in nodes:
    display_source_node(node)

# try out the RAG module directly
RAG = ragatouille_pack.get_modules()["RAG"]
results = RAG.search(
    "How does ColBERTv2 compare with BERT?", index_name=index_name, k=4
)
results
```
