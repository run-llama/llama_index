# XML Loader

```bash
pip install llama-index-readers-file
```

This loader extracts the text from a local XML file. A single local file is passed in each time you call `load_data`.

## Usage

To use this loader, you need to pass in a `Path` to a local file.

```python
from pathlib import Path

from llama_index.readers.file import XMLReader

loader = XMLReader()
documents = loader.load_data(file=Path("../example.xml"))
```

This loader is designed to be used as a way to load data into [LlamaIndex](https://github.com/run-llama/llama_index/).
