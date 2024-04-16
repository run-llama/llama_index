# Retrieval-Augmented Generation (RAG) Evaluation Pack

Get benchmark scores on your own RAG pipeline (i.e. `QueryEngine`) on a RAG
dataset (i.e., `LabelledRagDataset`). Specifically this pack takes in as input a
query engine and a `LabelledRagDataset`, which can also be downloaded from
[llama-hub](https://llamahub.ai).

## CLI Usage

You can download llamapacks directly using `llamaindex-cli`, which comes installed with the `llama-index` python package:

```bash
llamaindex-cli download-llamapack RagEvaluatorPack --download-dir ./rag_evaluator_pack
```

You can then inspect the files at `./rag_evaluator_pack` and use them as a template for your own project!

## Code Usage

You can download the pack to the `./rag_evaluator_pack` directory through python
code as well. The sample script below demonstrates how to construct `RagEvaluatorPack`
using a `LabelledRagDataset` downloaded from `llama-hub` and a simple RAG pipeline
built off of its source documents.

```python
from llama_index.core.llama_dataset import download_llama_dataset
from llama_index.core.llama_pack import download_llama_pack
from llama_index.core import VectorStoreIndex

# download a LabelledRagDataset from llama-hub
rag_dataset, documents = download_llama_dataset(
    "PaulGrahamEssayDataset", "./paul_graham"
)

# build a basic RAG pipeline off of the source documents
index = VectorStoreIndex.from_documents(documents=documents)
query_engine = index.as_query_engine()

# Time to benchmark/evaluate this RAG pipeline
# Download and install dependencies
RagEvaluatorPack = download_llama_pack(
    "RagEvaluatorPack", "./rag_evaluator_pack"
)

# construction requires a query_engine, a rag_dataset, and optionally a judge_llm
rag_evaluator_pack = RagEvaluatorPack(
    query_engine=query_engine, rag_dataset=rag_dataset
)

# PERFORM EVALUATION
benchmark_df = rag_evaluator_pack.run()  # async arun() also supported
print(benchmark_df)
```

`Output:`

```text
rag                            base_rag
metrics
mean_correctness_score         4.511364
mean_relevancy_score           0.931818
mean_faithfulness_score        1.000000
mean_context_similarity_score  0.945952
```

Note that `rag_evaluator_pack.run()` will also save two files in the same directory
in which the pack was invoked:

```bash
.
├── benchmark.csv (CSV format of the benchmark scores)
└── _evaluations.json (raw evaluation results for all examples & predictions)
```
