# Nougat OCR loader

```bash
pip install llama-index-readers-nougat-ocr
```

This loader reads the equations, symbols, and tables included in the PDF.

Users can input the path of the academic PDF document `file` which they want to parse. This OCR understands LaTeX math and tables.

## Usage

Here's an example usage of the PDFNougatOCR.

```python
from llama_index.readers.nougat_ocr import PDFNougatOCR

reader = PDFNougatOCR()

pdf_path = Path("/path/to/pdf")

documents = reader.load_data(pdf_path)
```

## Miscellaneous

An `output` folder will be created with the same name as the pdf and `.mmd` extension.
