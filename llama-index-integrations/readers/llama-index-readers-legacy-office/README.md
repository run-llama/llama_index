# LlamaIndex Legacy Office Reader

<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/data_connectors/lagecy_office_reader.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

## Overview

The Legacy Office Reader allows loading data from legacy Office documents (like Word 97 `.doc` files) using Apache Tika. It runs the Tika server locally to avoid remote server calls.

### Installation

You can install the Legacy Office Reader via pip:

```bash
pip install llama-index-readers-legacy-office
```

### Usage

#### Basic Usage

```python
from llama_index.readers.legacy_office import LegacyOfficeReader

# Initialize LegacyOfficeReader
reader = LegacyOfficeReader(
    tika_server_jar_path="path/to/tika-server.jar",  # Optional: Path to Tika server JAR
)

# Load data from a legacy Office document
documents = reader.load_data(
    file="path/to/document.doc",  # Path to the legacy Office document
)
```

#### Using with SimpleDirectoryReader

```python
from llama_index.core import SimpleDirectoryReader
from llama_index.readers.legacy_office import LegacyOfficeReader

reader = SimpleDirectoryReader(
    input_dir="path/to/directory/",
    file_extractor={".doc": LegacyOfficeReader()},
)
documents = reader.load_data()
```

### Features

- Parses legacy Office documents (`.doc`) using Apache Tika
- Optionally (default) runs Tika server locally to avoid remote server calls/dependencies
- Extracts both content and metadata from documents
- Supports batch processing of multiple documents
- Seamless integration with SimpleDirectoryReader

### Requirements

- Java Runtime Environment (JRE) 11 or higher (required for Apache Tika 3.x)
- Python 3.8 or higher

### Notes

- The first time you use the reader, it will download the Tika server JAR file if not provided
- The Tika server will run locally on port `9998`
- All document metadata is preserved in the Document objects
- Make sure you have Java 11+ installed and available in your system PATH
- The reader uses Apache Tika 3.x

### Credits

This reader is built on top of:

- [Apache Tika](https://tika.apache.org/) - Content analysis toolkit
- [tika-python](https://github.com/chrismattmann/tika-python/) - Python bindings for Apache Tika
