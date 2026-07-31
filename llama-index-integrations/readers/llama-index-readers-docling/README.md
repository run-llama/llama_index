# Docling Reader

## Overview

Docling Reader uses [Docling](https://github.com/DS4SD/docling) to enable fast and easy extraction of PDF, DOCX, HTML, and other document types, into Markdown or JSON-serialized Docling format, for usage in LlamaIndex pipelines for RAG / QA etc.

## Installation

```console
pip install llama-index-readers-docling
```

This installs Docling's lightweight service client without the local models or
PyTorch. To use the local `DocumentConverter`, install the full Docling package
as well:

```console
pip install "docling>=2.92.0,<3"
```

## Usage

`DoclingReader` accepts either the local `DocumentConverter` used by default
(when the full Docling package is installed) or any compatible converter
returning Docling's standard `ConversionResult`, such as
`DoclingServiceClient`.

### Markdown export

By default, Docling Reader exports to Markdown. Basic usage looks like this:

```python
from llama_index.readers.docling import DoclingReader

reader = DoclingReader()
docs = reader.load_data(file_path="https://arxiv.org/pdf/2408.09869")
print(f"{docs[0].text[389:442]}...")
# > ## Abstract
# >
# > This technical report introduces Docling...
```

### JSON export

Docling Reader can also export Docling's native format to JSON:

```python
from llama_index.readers.docling import DoclingReader

reader = DoclingReader(export_type=DoclingReader.ExportType.JSON)
docs = reader.load_data(file_path="https://arxiv.org/pdf/2408.09869")
print(f"{docs[0].text[:53]}...")
# > {"schema_name": "DoclingDocument", "version": "1.0.0"...
```

> [!IMPORTANT]
> To appropriately parse Docling's native format, when using JSON export make sure
> to use a Docling Node Parser in your pipeline.

### Managed Docling service

To process documents with
[Docling for IBM watsonx](https://www.ibm.com/products/docling), or with a
self-hosted Docling Serve endpoint, supply Docling's existing service client.
Conversion features are configured with Docling's native
`ConvertDocumentsOptions` model:

```python
import os

from docling.datamodel.service.options import ConvertDocumentsOptions
from docling.service_client import DoclingServiceClient
from llama_index.readers.docling import DoclingReader

options = ConvertDocumentsOptions(
    do_ocr=True,
    table_mode="accurate",
)

with DoclingServiceClient(
    url=os.environ["DOCLING_SERVICE_URL"],
    api_key=os.environ["DOCLING_API_KEY"],
    options=options,
) as client:
    reader = DoclingReader(doc_converter=client)
    docs = reader.load_data(
        file_path="https://arxiv.org/pdf/2408.09869",
    )
```

The service client and local converter return the same conversion result, so
the reader's Markdown and JSON export behavior is unchanged. See the
[Docling service API documentation](https://docling-project.github.io/docling/usage/api_server/rest_api/)
for the full set of conversion options.

### With Simple Directory Reader

The Docling Reader can also be used directly in combination with Simple Directory Reader, for example:

```python
from llama_index.core import SimpleDirectoryReader

dir_reader = SimpleDirectoryReader(
    input_dir="/path/to/docs",
    file_extractor={".pdf": reader},
)
docs = dir_reader.load_data()
print(docs[0].metadata)
# > {'file_path': '/path/to/docs/2408.09869v3.pdf',
# >  'file_name': '2408.09869v3.pdf',
# >  'file_type': 'application/pdf',
# >  'file_size': 5566574,
# >  'creation_date': '2024-10-06',
# >  'last_modified_date': '2024-10-03'}
```
