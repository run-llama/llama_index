# LlamaIndex Legacy Office Reader

## Overview

The Legacy Office Reader allows loading data from legacy Office documents (like Word 97 .doc files) using Apache Tika. It runs the Tika server locally to avoid remote server calls.

### Installation

You can install the Legacy Office Reader via pip:

```bash
pip install llama-index-readers-legacy-office
```

### Usage

```python
from llama_index.readers.legacy_office import LegacyOfficeReader

# Initialize LegacyOfficeReader
reader = LegacyOfficeReader(
    tika_server_jar_path="path/to/tika-server.jar",  # Optional: Path to Tika server JAR
    tika_server_port=9998,  # Optional: Port to run Tika server on (default: 9998)
)

# Load data from a legacy Office document
documents = reader.load_data(
    file_path="path/to/document.doc",  # Path to the legacy Office document
)

# Or load multiple documents
documents = reader.load_data(
    file_path=["path/to/doc1.doc", "path/to/doc2.doc"],
)
```

### Features

- Parses legacy Office documents (.doc, etc.) using Apache Tika
- Runs Tika server locally to avoid remote server calls
- Extracts both content and metadata from documents
- Supports batch processing of multiple documents

### Requirements

- Java Runtime Environment (JRE) 8 or higher
- Python 3.8 or higher

### Notes

- The first time you use the reader, it will download the Tika server JAR file if not provided
- The Tika server will run locally on the specified port
- All document metadata is preserved in the Document objects
