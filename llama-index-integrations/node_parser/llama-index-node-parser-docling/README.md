# Docling Node Parser

## Overview

Docling Node Parser parses [Docling](https://github.com/DS4SD/docling) JSON output into LlamaIndex nodes with rich metadata for usage in downstream pipelines for RAG / QA etc.

## Installation

```console
pip install llama-index-node-parser-docling
```

## Usage

Docling Node Parser parses LlamaIndex documents containing JSON-serialized Docling format, as created by a Docling Reader.

Basic usage looks like this:

```python
# docs = ...  # e.g. created using Docling Reader in JSON mode

from llama_index.node_parser.docling import DoclingNodeParser

node_parser = DoclingNodeParser()
nodes = node_parser.get_nodes_from_documents(documents=docs)
print(f"{nodes[12].text[:70]}...")
# > Docling provides an easy code interface to convert PDF documents from ...

print(nodes[12].metadata)
# > {'doc_items': [
# >    'self_ref': '#/main-text/21',
# >    'prov': [
# >      'page_no': 2,
# >      'bbox': {'l': 107.3, 't': 499.5, 'r': 504.0, 'b': 456.7, ...},
# >      ...
# >  ],
# >  'headings': ['2 Getting Started'],
# >  ...
# > }
```
