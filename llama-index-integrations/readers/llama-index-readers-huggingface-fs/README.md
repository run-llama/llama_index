# Hugging Face FS Loader

This loader uses Hugging Face Hub's Filesystem API (> 0.14) to
load datasets.

Besides the existing `load_data` function, you may also choose to use
`load_dicts` and `load_df`.

## Usage

To use this loader, you need to pass in a path to a Hugging Face dataset.

```python
from pathlib import Path
from llama_index import download_loader

HuggingFaceFSReader = download_loader("HuggingFaceFSReader")

# load documents
loader = HuggingFaceFSReader()
documents = loader.load_data("datasets/dair-ai/emotion/data/data.jsonl.gz")

# load dicts
dicts = loader.load_dicts("datasets/dair-ai/emotion/data/data.jsonl.gz")

# load df
df = loader.load_df("datasets/dair-ai/emotion/data/data.jsonl.gz")
```

This loader is designed to be used as a way to load data into [LlamaIndex](https://github.com/run-llama/llama_index/tree/main/llama_index) and/or subsequently used as a Tool in a [LangChain](https://github.com/hwchase17/langchain) Agent. See [here](https://github.com/emptycrown/llama-hub/tree/main) for examples.
