# Paddle OCR loader

```bash
pip install llama-index-readers-paddle-ocr
```

This loader reads the images and tables included in the PDF.

Users can input the path of the PDF document which they want to parse. This OCR understands images and tables.

## Usage

Here's an example usage of the PDFPaddleOCRReader.

```python
from llama_index.readers.paddle_ocr import PDFPaddleOCRReader

reader = PDFPaddleOCRReader()

documents = reader.load_data("/path/to/pdf")

for doc in documents:
    print(doc.text)
```

## Examples

This loader is designed to be used as a way to load data into [LlamaIndex](https://github.com/run-llama/llama_index/).
