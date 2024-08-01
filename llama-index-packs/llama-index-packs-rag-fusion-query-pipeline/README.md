# RAG Fusion Pipeline Llama Pack

This LlamaPack creates the RAG Fusion Query Pipeline, which runs multiple retrievers in parallel (with varying chunk sizes), and aggregates the results in the end with reciprocal rank fusion.

You can run it out of the box, but we also encourage you to inspect the code to take a look at how our `QueryPipeline` syntax works. More details on query pipelines can be found here: https://docs.llamaindex.ai/en/stable/module_guides/querying/pipeline/root.html.

Check out our [notebook guide](https://github.com/run-llama/llama-hub/blob/main/llama_hub/llama_packs/query/rag_fusion_pipeline/rag_fusion_pipeline.ipynb) as well.

## CLI Usage

You can download llamapacks directly using `llamaindex-cli`, which comes installed with the `llama-index` python package:

```bash
llamaindex-cli download-llamapack RAGFusionPipelinePack --download-dir ./rag_fusion_pipeline_pack
```

You can then inspect the files at `./rag_fusion_pipeline_pack` and use them as a template for your own project!

## Code Usage

You can download the pack to a `./rag_fusion_pipeline_pack` directory:

```python
from llama_index.core.llama_pack import download_llama_pack

# download and install dependencies
RAGFusionPipelinePack = download_llama_pack(
    "RAGFusionPipelinePack", "./rag_fusion_pipeline_pack"
)
```

From here, you can use the pack, or inspect and modify the pack in `./rag_fusion_pipeline_pack`.

Then, you can set up the pack like so:

```python
# create the pack
pack = RAGFusionPipelinePack(docs, llm=OpenAI(model="gpt-3.5-turbo"))
```

The `run()` function is a light wrapper around `query_pipeline.run(*args, **kwargs)`.

```python
response = pack.run(input="What did the author do during his time in YC?")
```

You can also use modules individually.

```python
# get query pipeline directly
pack.query_pipeline

# get retrievers for each chunk size
pack.retrievers

# get query engines for each chunk size
pack.query_engines
```
