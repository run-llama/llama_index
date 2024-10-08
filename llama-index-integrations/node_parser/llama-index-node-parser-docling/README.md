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
print(f"{nodes[6].text[:70]}...")
# > Docling provides an easy code interface to convert PDF documents from ...

print(nodes[6].metadata)
# > {'dl_doc_hash': '556ad9e23b...',
# >  'path': '#/main-text/22',
# >  'heading': '2 Getting Started',
# >  'page': 2,
# >  'bbox': [107.40, 456.93, 504.20, 499.65]}
```
