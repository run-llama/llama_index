# LlamaIndex MarkItDown Reader Integration

[MarkItDown](https://github.com/microsoft/markitdown) is a powerful tool that converts various file formats to Markdown.

`llama-index-readers-markitdown` is an integration that uses MarkItDown to extract text from various file formats, supporting:

- .txt files and text-based files without extension
- .csv, .xml and .json files
- HTML files (.html)
- Presentations (.pptx)
- Word documents (.docx)
- PDF documents (.pdf)
- ZIP files (.zip)

You can install it via:

```bash
pip install llama-index-readers-markitdown
```

And you can use it in your scripts as follows:

```python
from llama_index.readers.markitdown import MarkItDownReader

reader = MarkItDownReader()
documents = reader.load_data("presentation.pptx")
```
