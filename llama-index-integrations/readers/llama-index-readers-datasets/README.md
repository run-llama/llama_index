# LlamaIndex Readers Integration: HuggingFace Datasets

## Overview

HuggingFace Datasets Reader is a tool designed to load HuggingFace datasets as documents.

### Installation

You can install HuggingFace Datasets Reader via pip:

```bash
pip install llama-index-readers-datasets
```

## Usage

```python
from llama_index.readers.datasets import DatasetsReader
from datasets import load_dataset

reader = DatasetsReader()

# Load train split (default) as metadata
docs = reader.load_data("lhoestq/demo1")

# Load test split as metadata
docs = reader.load_data("lhoestq/demo1", split="test")

# Load specify the dictionary key to use as text value
docs = reader.load_data("lhoestq/demo1", text_key="review")

# Pass additional arguments to datasets.load_dataset
docs = reader.load_data("lhoestq/demo1", cache_dir="/tmp/huggingface")

# Load from a preloaded dataset (ignore all other arguments)
dataset = load_dataset("lhoestq/demo1", split="train")
docs = reader.load_data(dataset=dataset)

# Lazy loading (stream samples)
for it in reader.lazy_load_data(
    "lhoestq/demo1", split="test", text_key="review", doc_id_key="id"
):
    print(it)
```
