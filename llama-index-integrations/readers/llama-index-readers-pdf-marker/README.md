# LlamaIndex Readers Integration: Pdf-Marker

Uses the [pdf-marker](https://github.com/VikParuchuri/marker/) library to extract the content of a PDF file.

From the original README:

Marker converts PDF to markdown quickly and accurately.

- Supports a wide range of documents (optimized for books and scientific papers)
- Supports all languages
- Removes headers/footers/other artifacts
- Formats tables and code blocks
- Extracts and saves images along with the markdown
- Converts most equations to latex
- Works on GPU, CPU, or MPS

## Usage

Here's an example usage of the PDFMarkerReader.

```python
from llama_index.readers.pdf_marker import PDFMarkerReader
from pathlib import Path

path = Path("/path/to/pdf")
reader = PDFMarkerReader()
reader.load_data(path)
```

## License

The marker-pdf library is licensed under the GPL-3.0 license (see https://github.com/VikParuchuri/marker), meaning that you may copy, distribute and modify the software as long as you track changes/dates in source files.
Any modifications to or software including (via compiler) GPL-licensed code must also be made available under the GPL along with build & install instructions.

There is also commercial usage limitations (see https://github.com/VikParuchuri/marker?tab=readme-ov-file#commercial-usage).
