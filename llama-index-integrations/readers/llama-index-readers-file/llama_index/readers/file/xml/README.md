# XML Loader

```bash
pip install llama-index-readers-file
```

This loader extracts XML documents from a local file or an fsspec filesystem.
A single file is passed in each time you call `load_data`.

## Usage

To use this loader, you need to pass in a `Path` to a local file.

```python
from pathlib import Path

from llama_index.readers.file import XMLReader

loader = XMLReader()
documents = loader.load_data(file=Path("../example.xml"))
```

For another filesystem, pass its instance with `fs` and a path understood by that
filesystem. For example, an in-memory file can be loaded without a local copy:

```python
import fsspec

fs = fsspec.filesystem("memory")
fs.pipe("/example.xml", b"<items><item>Apple</item></items>")
documents = loader.load_data(file="/example.xml", fs=fs)
```

The reader opens filesystem files in binary mode so the XML parser can respect
the encoding declared in the document. `tree_level_split` and `extra_info` work
the same way for local and filesystem-backed files.

This loader is designed to be used as a way to load data into [LlamaIndex](https://github.com/run-llama/llama_index/).
