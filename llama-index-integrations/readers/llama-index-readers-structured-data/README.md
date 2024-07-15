# LlamaIndex Readers Integration: Structured-Data

The function 'StructuredDataReader' supports reading files in JSON, JSONL, CSV, and XLSX formats. It provides parameters 'col_index' and 'col_metadata' to differentiate between columns that should be written into the document's main text and additional metadata.

## Install package

```bash
pip install llama-index-readers-structured-data
```

Or install locally:

```bash
pip install -e llama-index-integrations/readers/llama-index-readers-structured-data
```

## Usage

1. for single document:

```python
from pathlib import Path
from llama_index.readers.structured_data.base import StructuredDataReader

parser = StructuredDataReader(col_index=["col1", "col2"], col_metadata=0)
documents = parser.load_data(Path("your/file/path.json"))
```

2. for dictory of documents:

```python
from pathlib import Path
from llama_index.core import SimpleDirectoryReader
from llama_index.readers.structured_data.base import StructuredDataReader

parser = StructuredDataReader(col_index=[1, -1], col_metadata="col3")
file_extractor = {
    ".xlsx": parser,
    ".csv": parser,
    ".json": parser,
    ".jsonl": parser,
}
documents = SimpleDirectoryReader(
    "your/dic/path", file_extractor=file_extractor
).load_data()
```
