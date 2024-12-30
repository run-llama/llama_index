# Hugging Face FS Loader

```bash
pip install llama-index-readers-huggingface-fs
```

This loader uses Hugging Face Hub's Filesystem API (> 0.14) to
load datasets.

Besides the existing `load_data` function, you may also choose to use
`load_dicts` and `load_df`.

## Usage

To use this loader, you need to pass in a path to a Hugging Face dataset.

```python
from pathlib import Path

from llama_index.readers.huggingface_fs import HuggingFaceFSReader

# load documents
loader = HuggingFaceFSReader()
documents = loader.load_data("datasets/dair-ai/emotion/data/data.jsonl.gz")

# load dicts
dicts = loader.load_dicts("datasets/dair-ai/emotion/data/data.jsonl.gz")

# load df
df = loader.load_df("datasets/dair-ai/emotion/data/data.jsonl.gz")
```

This loader is designed to be used as a way to load data into [LlamaIndex](https://github.com/run-llama/llama_index/).
