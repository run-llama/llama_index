# LayoutIR Reader

## Overview

LayoutIR Reader uses [LayoutIR](https://pypi.org/project/layoutir/) - a production-grade document ingestion and canonicalization engine with compiler-like architecture. Unlike simple PDF-to-Markdown converters, LayoutIR processes documents through an Intermediate Representation (IR) layer, enabling precise preservation of complex layouts, tables, and multi-column structures.

## Why LayoutIR?

LayoutIR stands out for its:

- **Deterministic Processing**: Hash-based stable IDs ensure reproducible results
- **Layout Preservation**: Maintains complex multi-column layouts and table structures
- **Canonical IR Schema**: Typed intermediate representation for reliable downstream processing
- **Flexible Chunking**: Semantic section-based or fixed-size chunking strategies
- **GPU Acceleration**: Optional GPU support for faster document processing
- **Production-Ready**: Designed for enterprise-grade document pipelines

## Installation

### Basic Installation

```bash
pip install llama-index-readers-layoutir
```

### With GPU Support

For GPU acceleration, first install PyTorch with CUDA support:

```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install llama-index-readers-layoutir
```

## Usage

### Basic Usage

Load a PDF document with default settings:

```python
from llama_index.readers.layoutir import LayoutIRReader

reader = LayoutIRReader()
documents = reader.load_data(file_path="document.pdf")

# Each document preserves block structure and metadata
for doc in documents:
    print(f"Block Type: {doc.metadata['block_type']}")
    print(f"Page: {doc.metadata['page_number']}")
    print(f"Content: {doc.text[:100]}...")
```

### With GPU Acceleration

Enable GPU processing for faster performance:

```python
from llama_index.readers.layoutir import LayoutIRReader

reader = LayoutIRReader(use_gpu=True)
documents = reader.load_data(file_path="large_document.pdf")
```

### Custom Chunking Strategy

Use semantic section-based chunking:

```python
from llama_index.readers.layoutir import LayoutIRReader

reader = LayoutIRReader(
    chunk_strategy="semantic",
    max_heading_level=2,  # Split at h1 and h2 headings
)
documents = reader.load_data(file_path="structured_document.pdf")
```

### Processing Multiple Files

Process a batch of documents:

```python
from llama_index.readers.layoutir import LayoutIRReader
from pathlib import Path

reader = LayoutIRReader(use_gpu=True)

file_paths = ["report_2024.pdf", "technical_spec.pdf", "user_manual.pdf"]

documents = reader.load_data(file_path=file_paths)
print(f"Loaded {len(documents)} document blocks from {len(file_paths)} files")
```

### Integration with VectorStoreIndex

Build a searchable index from LayoutIR-processed documents:

```python
from llama_index.readers.layoutir import LayoutIRReader
from llama_index.core import VectorStoreIndex

# Load documents with preserved layout structure
reader = LayoutIRReader(
    use_gpu=True, chunk_strategy="semantic", max_heading_level=2
)
documents = reader.load_data(file_path="company_knowledge_base.pdf")

# Create index
index = VectorStoreIndex.from_documents(documents)

# Query with layout-aware context
query_engine = index.as_query_engine()
response = query_engine.query("What are the key financial metrics in Q4?")
print(response)
```

### With SimpleDirectoryReader

Integrate LayoutIR for PDF processing in directory operations:

```python
from llama_index.core import SimpleDirectoryReader
from llama_index.readers.layoutir import LayoutIRReader

reader = LayoutIRReader(use_gpu=True)

dir_reader = SimpleDirectoryReader(
    input_dir="/path/to/documents",
    file_extractor={".pdf": reader},
)

documents = dir_reader.load_data()
print(f"Processed {len(documents)} blocks")
```

### Advanced Configuration

Full configuration example:

```python
from llama_index.readers.layoutir import LayoutIRReader

reader = LayoutIRReader(
    use_gpu=True,  # Enable GPU acceleration
    chunk_strategy="semantic",  # Use semantic chunking
    max_heading_level=3,  # Split up to h3 level
    model_name="custom_model",  # Optional: specify model
    api_key="your_api_key",  # Optional: for remote processing
)

documents = reader.load_data(
    file_path="complex_layout.pdf",
    extra_info={"department": "research", "year": 2026},
)

# Access rich metadata
for doc in documents:
    print(f"ID: {doc.doc_id}")
    print(f"Type: {doc.metadata['block_type']}")
    print(f"Page: {doc.metadata['page_number']}")
    print(f"Department: {doc.metadata['department']}")
```

## Metadata Structure

Each Document includes the following metadata:

- `file_path`: Source file path
- `file_name`: Source file name
- `block_type`: Type of content block (table, paragraph, heading, etc.)
- `block_index`: Index of the block in the document
- `page_number`: Page number where the block appears
- `source`: Always "layoutir"
- Plus any `extra_info` passed to `load_data()`

## Requirements

- Python >= 3.10
- llama-index-core >= 0.13.0
- layoutir >= 1.0.3
- Optional: PyTorch with CUDA for GPU acceleration

## License

MIT
