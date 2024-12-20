# LlamaDataset Metadata Pack

As part of the `LlamaDataset` submission package into [llamahub](https://llamahub.ai),
two metadata files are required, namely: `card.json` and `README.md`. This pack
creates these two files and saves them to disk to help expedite the submission
process.

## CLI Usage

You can download llamapacks directly using `llamaindex-cli`, which comes installed with the `llama-index` python package:

```bash
llamaindex-cli download-llamapack LlamaDatasetMetadataPack --download-dir ./llama_dataset_metadata_pack
```

You can then inspect the files at `./llama_dataset_metadata_pack` and use them as a template for your own project!

## Code Usage

You can download the pack to the `./llama_dataset_metadata_pack` directory through python
code as well. The sample script below demonstrates how to construct `LlamaDatasetMetadataPack`
using a `LabelledRagDataset` downloaded from `llama-hub` and a simple RAG pipeline
built off of its source documents.

```python
from llama_index.core.llama_pack import download_llama_pack

# Download and install dependencies
LlamaDatasetMetadataPack = download_llama_pack(
    "LlamaDatasetMetadataPack", "./llama_dataset_metadata_pack"
)

# construction requires a query_engine, a rag_dataset, and optionally a judge_llm
llama_dataset_metadata_pack = LlamaDatasetMetadataPack()

# create and save `card.json` and `README.md` to disk
dataset_description = (
    "A labelled RAG dataset based off an essay by Paul Graham, consisting of "
    "queries, reference answers, and reference contexts."
)

llama_dataset_metadata_pack.run(
    name="Paul Graham Essay Dataset",
    description=dataset_description,
    rag_dataset=rag_dataset,  # defined earlier not shown here
    index=index,  # defined earlier not shown here
    benchmark_df=benchmark_df,  # defined earlier not shown here
    baseline_name="llamaindex",
)
```

NOTE: this pack should be used only after performing a RAG evaluation (i.e., by
using `RagEvaluatorPack` on a `LabelledRagDataset`). In the code snippet above,
`index`, `rag_dataset`, and `benchmark_df` are all objects that you'd expect to
have only after performing the RAG evaluation as mention in the previous sentence.
