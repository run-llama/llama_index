# LlamaIndex Readers Integration: File

```sh
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
- FunASRReader
- UnstructuredReader
- PyMuPDFReader
- ImageTabularChartReader
- XMLReader
- PagedCSVReader
- CSVReader
- RTFReader

## Installation

```sh
pip install llama-index-readers-file
```

```sh
## Optional dependency for FunASRReader

`FunASRReader` uses the local FunASR Python package for speech-to-text 
transcription.

To use `FunASRReader`, install the optional dependency:

```bash
pip install funasr
```

After installing `funasr`, you can use `FunASRReader` with `SimpleDirectoryReader`
as shown in the example below.
```

## Usage

Once installed, You can import any of the loader. Here's an example usage of one of the loader.

```python
from llama_index.core import SimpleDirectoryReader
from llama_index.readers.file import (
    DocxReader,
    HWPReader,
    PDFReader,
    EpubReader,
    FlatReader,
    HTMLTagReader,
    ImageCaptionReader,
    ImageReader,
    ImageVisionLLMReader,
    IPYNBReader,
    MarkdownReader,
    MboxReader,
    PptxReader,
    PandasCSVReader,
    VideoAudioReader,    
    FunASRReader,
    UnstructuredReader,
    PyMuPDFReader,
    ImageTabularChartReader,
    XMLReader,
    PagedCSVReader,
    CSVReader,
    RTFReader,
)

# PDF Reader with `SimpleDirectoryReader`
parser = PDFReader()
file_extractor = {".pdf": parser}
documents = SimpleDirectoryReader(
    "./data", file_extractor=file_extractor
).load_data()

# Docx Reader example
parser = DocxReader()
file_extractor = {".docx": parser}
documents = SimpleDirectoryReader(
    "./data", file_extractor=file_extractor
).load_data()

# HWP Reader example
parser = HWPReader()
file_extractor = {".hwp": parser}
documents = SimpleDirectoryReader(
    "./data", file_extractor=file_extractor
).load_data()

# Epub Reader example
parser = EpubReader()
file_extractor = {".epub": parser}
documents = SimpleDirectoryReader(
    "./data", file_extractor=file_extractor
).load_data()

# Flat Reader example
parser = FlatReader()
file_extractor = {".txt": parser}
documents = SimpleDirectoryReader(
    "./data", file_extractor=file_extractor
).load_data()

# HTML Tag Reader example
parser = HTMLTagReader()
file_extractor = {".html": parser}
documents = SimpleDirectoryReader(
    "./data", file_extractor=file_extractor
).load_data()

# Image Reader example
parser = ImageReader()
file_extractor = {
    ".jpg": parser,
    ".jpeg": parser,
    ".png": parser,
}  # Add other image formats as needed
documents = SimpleDirectoryReader(
    "./data", file_extractor=file_extractor
).load_data()

# IPYNB Reader example
parser = IPYNBReader()
file_extractor = {".ipynb": parser}
documents = SimpleDirectoryReader(
    "./data", file_extractor=file_extractor
).load_data()

# Markdown Reader example
parser = MarkdownReader()
file_extractor = {".md": parser}
documents = SimpleDirectoryReader(
    "./data", file_extractor=file_extractor
).load_data()

# Mbox Reader example
parser = MboxReader()
file_extractor = {".mbox": parser}
documents = SimpleDirectoryReader(
    "./data", file_extractor=file_extractor
).load_data()

# Pptx Reader example
# Basic usage - extracts text, tables, charts, and speaker notes
parser = PptxReader()

# Advanced usage - control parsing behavior
parser = PptxReader(
    extract_images=True,  # Enable image captioning
    context_consolidation_with_llm=True,  # Use LLM for content synthesis
    num_workers=4,  # Parallel processing
    batch_size=10,  # Slides processed per worker batch
    raise_on_error=True,  # Raise value error if file_parsing is not successful
)

file_extractor = {".pptx": parser}
documents = SimpleDirectoryReader(
    "./data", file_extractor=file_extractor
).load_data()


# Pandas CSV Reader example
parser = PandasCSVReader()
file_extractor = {".csv": parser}  # Add other CSV formats as needed
documents = SimpleDirectoryReader(
    "./data", file_extractor=file_extractor
).load_data()

# PyMuPDF Reader example
parser = PyMuPDFReader()
file_extractor = {".pdf": parser}
documents = SimpleDirectoryReader(
    "./data", file_extractor=file_extractor
).load_data()

# XML Reader example
parser = XMLReader()
file_extractor = {".xml": parser}
documents = SimpleDirectoryReader(
    "./data", file_extractor=file_extractor
).load_data()

# Paged CSV Reader example
parser = PagedCSVReader()
file_extractor = {".csv": parser}  # Add other CSV formats as needed
documents = SimpleDirectoryReader(
    "./data", file_extractor=file_extractor
).load_data()

# CSV Reader example
parser = CSVReader()
file_extractor = {".csv": parser}  # Add other CSV formats as needed
documents = SimpleDirectoryReader(
    "./data", file_extractor=file_extractor
).load_data()

# FunASR Reader example
#
# FunASR is an optional dependency.
# Install it before using FunASRReader:
#
#   pip install funasr
parser = FunASRReader(
    model="iic/SenseVoiceSmall",
    device="cpu",)
file_extractor = {
    ".wav": parser,
    ".mp3": parser,
    ".m4a": parser,
    ".flac": parser,}
documents = SimpleDirectoryReader(
    "./data", file_extractor=file_extractor
).load_data()


```

```text
Python was not found; run without arguments to install from the Microsoft Store, or disable this shortcut from Settings > Apps > Advanced app settings > App execution aliases.
```

This loader is designed to be used as a way to load data into [LlamaIndex](https://github.com/run-llama/llama_index/).