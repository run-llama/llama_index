# MangoppsGuides Loader

```bash
pip install llama-index-readers-mangoapps-guides
```

This loader fetches the text from Mangopps Guides.

## Usage

To use this loader, you need to pass base url of the MangoppsGuides installation (e.g. `https://guides.mangoapps.com/`) and the limit , i.e. max number of links it should crawl

```python
from llama_index.readers.mangoapps_guides import MangoppsGuidesReader

loader = MangoppsGuidesReader()
documents = loader.load_data(
    domain_url="https://guides.mangoapps.com", limit=1
)
```

This loader is designed to be used as a way to load data into [LlamaIndex](https://github.com/run-llama/llama_index/).
