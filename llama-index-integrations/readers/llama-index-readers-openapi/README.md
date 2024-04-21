# LlamaIndex Readers Integration: Open API Specification

This module provides a reader for Open API Specification (OAS) JSON files. The reader is able to parse OAS files and split them at into atomic elements, such as paths, operations, parameters, etc.

It also provides some basic customizations to the reader such as changing the depth of the split, and the ability to exclude certain elements.

## Usage

```python
from llama_index.readers.openapi import OpenAPIReader

openapi_reader = OpenAPIReader(discard=["info", "servers"])
openapi_reader.load_data("path/to/openapi.json")
```
