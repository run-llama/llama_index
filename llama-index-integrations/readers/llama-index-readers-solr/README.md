# LlamaIndex Readers Integration: Solr

## Overview

Solr Reader retrieves documents through an existing Solr index. These documents can then be used in a downstream LlamaIndex data structure.

### Installation

You can install Solr Reader via pip:

```bash
pip install llama-index-readers-solr
```

## Usage

```python
from llama_index.readers.solr import SolrReader

# Initialize SolrReader with the Solr URL. The Solr URL should include the path
# to the core (if single node) or collection (if Solr Cloud).
reader = SolrReader(endpoint="<Endpoint with full solr path>")

# Load data from Solr index
documents = reader.load_data(
    query={"q": "*:*", "rows": 10},  # Solr query parameters
    field="content_t",  # Only results with populated values in this field will be returned
    metadata_fields=["title_t", "category_s"],
)
```

This loader is designed to be used as a way to load data into
[LlamaIndex](https://github.com/run-llama/llama_index/tree/main/llama_index) and/or subsequently
used as a Tool in a [LangChain](https://github.com/hwchase17/langchain) Agent.
