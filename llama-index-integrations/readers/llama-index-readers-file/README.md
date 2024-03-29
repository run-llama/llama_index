# LlamaIndex Readers Integration: File

```bash
pip install llama-index-readers-file
```

This is the default integration for different loaders that are used within `SimpleDirectoryReader`.

Provides support for the following loaders:
- DocxReader
- HWPReader
- PDFReader
- EpubReader
- FlatReader
- HTMLTagReader
- ImageCaptionReader
- ImageReader
- ImageVisionLLMReader
- IPYNBReader
- MarkdownReader
- MboxReader
- PptxReader
- PandasCSVReader
- VideoAudioReader
- UnstructuredReader
- PyMuPDFReader
- ImageTabularChartReader
- XMLReader
- PagedCSVReader
- CSVReader
- RTFReader

## Installation

```bash
pip install llama-index-readers-file
```

## Usage

Once installed, You can import any of the loader. Here's an example usage of one of the loader.

```python
from llama_index.readers.file import PDFReader
from llama_index.core import SimpleDirectoryReader

parser = PDFReader()
file_extractor = {".pdf": parser}
documents = SimpleDirectoryReader("./data", file_extractor=file_extractor).load_data()

```

This loader is designed to be used as a way to load data into [LlamaIndex](https://github.com/run-llama/llama_index/tree/main/llama_index) and/or subsequently used as a Tool in a [LangChain](https://github.com/hwchase17/langchain) Agent.