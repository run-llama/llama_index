# PyMuPDF Loader

```bash
pip install llama-index-readers-file pymupdf
```

This loader extracts text from a local PDF file using the `PyMuPDF` Python library. If `metadata` is passed as True while calling `load` function; extracted documents will include basic metadata such as page numbers, file path and total number of pages in pdf.

## Usage

To use this loader, you need to pass file path of the local file as string or `Path` when you call `load` function. By default, including metadata is set to True. You can also pass extra information in a `dict` format when you call `load` function.

```python
from pathlib import Path

from llama_index.readers.file import PyMuPDFReader

loader = PyMuPDFReader()
documents = loader.load_data(file_path=Path("./article.pdf"), metadata=True)
```

This loader is designed to be used as a way to load data into [LlamaIndex](https://github.com/run-llama/llama_index/).
