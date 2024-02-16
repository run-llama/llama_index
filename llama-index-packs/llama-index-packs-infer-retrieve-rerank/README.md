# Infer-Retrieve-Rerank LlamaPack

This is our implementation of the paper ["In-Context Learning for Extreme Multi-Label Classification](https://arxiv.org/pdf/2401.12178.pdf) by Oosterlinck et al.

The paper proposes "infer-retrieve-rerank", a simple paradigm using frozen LLM/retriever models that can do "extreme"-label classification (the label space is huge).

1. Given a user query, use an LLM to predict an initial set of labels.
2. For each prediction, retrieve the actual label from the corpus.
3. Given the final set of labels, rerank them using an LLM.

All of these can be implemented as LlamaIndex abstractions.

A full notebook guide can be found [here](https://github.com/run-llama/llama-hub/blob/main/llama_hub/llama_packs/research/infer_retrieve_rerank/infer_retrieve_rerank.ipynb).

## CLI Usage

You can download llamapacks directly using `llamaindex-cli`, which comes installed with the `llama-index` python package:

```bash
llamaindex-cli download-llamapack InferRetrieveRerankPack --download-dir ./infer_retrieve_rerank_pack
```

You can then inspect the files at `./infer_retrieve_rerank_pack` and use them as a template for your own project!

## Code Usage

You can download the pack to a `./infer_retrieve_rerank_pack` directory:

```python
from llama_index.core.llama_pack import download_llama_pack

# download and install dependencies
InferRetrieveRerankPack = download_llama_pack(
    "InferRetrieveRerankPack", "./infer_retrieve_rerank_pack"
)
```

From here, you can use the pack, or inspect and modify the pack in `./infer_retrieve_rerank_pack`.

Then, you can set up the pack like so:

```python
# create the pack
pack = InferRetrieveRerankPack(
    labels,  # list of all label strings
    llm=llm,
    pred_context="<pred_context>",
    reranker_top_n=3,
    verbose=True,
)
```

The `run()` function runs predictions.

```python
pred_reactions = pack.run(inputs=[s["text"] for s in samples])
```

You can also use modules individually.

```python
# call the llm.complete()
llm = pack.llm
label_retriever = pack.label_retriever
```
