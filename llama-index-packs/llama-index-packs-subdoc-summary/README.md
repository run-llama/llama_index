# LlamaIndex Packs Integration: Subdoc-Summary

This LlamaPack provides an advanced technique for injecting each chunk with "sub-document" metadata. This context augmentation technique is helpful for both retrieving relevant context and for synthesizing correct answers.

It is a step beyond simply adding a summary of the document as the metadata to each chunk. Within a long document, there can be multiple distinct themes, and we want each chunk to be grounded in global but relevant context.

This technique was inspired by our "Practical Tips and Tricks" video: https://www.youtube.com/watch?v=ZP1F9z-S7T0.

## Installation

```bash
pip install llama-index llama-index-packs-subdoc-summary
```

## CLI Usage

You can download llamapacks directly using `llamaindex-cli`, which comes installed with the `llama-index` python package:

```bash
llamaindex-cli download-llamapack SubDocSummaryPack --download-dir ./subdoc_summary_pack
```

You can then inspect the files at `./subdoc_summary_pack` and use them as a template for your own project.

## Code Usage

You can download the pack to a the `./subdoc_summary_pack` directory:

```python
from llama_index.core.llama_pack import download_llama_pack

# download and install dependencies
SubDocSummaryPack = download_llama_pack(
    "SubDocSummaryPack", "./subdoc_summary_pack"
)

# You can use any llama-hub loader to get documents!
subdoc_summary_pack = SubDocSummaryPack(
    documents,
    parent_chunk_size=8192,  # default,
    child_chunk_size=512,  # default
    llm=OpenAI(model="gpt-3.5-turbo"),
    embed_model=OpenAIEmbedding(),
)
```

Initializing the pack will split documents into parent chunks and child chunks. It will inject parent chunk summaries into child chunks, and index the child chunks.

Running the pack will run the query engine over the vectorized child chunks.

```python
response = subdoc_summary_pack.run("<query>", similarity_top_k=2)
```
