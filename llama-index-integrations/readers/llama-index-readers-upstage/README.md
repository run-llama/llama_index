# UpstageDocumentParseReader

## UpstageDocumentParseReader

```bash
pip install llama-index-readers-upstage
```

This reader loads document files and detects elements such as text, tables, and figures using the Upstage Document Parse API. Users wishing to utilize this reader must obtain an API key from the [Upstage console](https://console.upstage.ai).

### Construction

The `UpstageDocumentParseReader` is equipped with the following three optional parameters during instantiation:

- `api_key`: This parameter is designed to accept a string that serves as an API access token. If the API key has already been registered in the environment variable `UPSTAGE_API_KEY`, there is no necessity to input it again during the reader's configuration.

- `ocr`: A string value indicating whether to perform OCR inference on the document before layout detection. The possible value is one of `auto` and `force`. The default is `auto` which means that OCR is performed for image input only. When this option is set to `auto` for PDF or non-image documents, the engine directly extracts text and coordinates from the document without converting it to images. Otherwise, the engine converts the input file to images and performs OCR inference before layout detection.

- `output_format`: A list of string value indicating in which each layout element output is formatted. Possible values are `text`, `html`, and `markdown`. The default value is `"html"`

- `coordinates`: A boolean value indicating whether to return coordinates of bounding boxes of each layout element. The default is `true`

- `base64_encoding`: A list of string value indicating which layout category should be provided as base64 encoded string. Categories are include `paragraph`, `table`, `figure`, `header`, `footer`, `caption`, `equation`, `heading1`, `list`, `index`, `footnote`, `chart`. This feature is useful when user wants to crop the layout element from the original document image and store and use it for their own purpose. For example, users can extract image base64 encoding of all tables of the input document with `["table"]`. All layout categories can be specified.

### `load_data`

The `load_data` function, encompassed within the `UpstageDocumentParseReader`, extends from the `BaseReader` class. The `lazy_load_data` function mirrors the functionalities of the `load_data` function but with an enhanced focus on efficiency and lazy loading, making it particularly suitable for handling large files. Utilizing this function effectively necessitates a thorough understanding of its parameters and their respective expected inputs to harness its full potential:

- **`file_path` (required):** This critical parameter accepts either a single string or `pathlib.Path` object, or a list comprising multiple of these elements, representing the path(s) to the file(s) intended for loading. Proper accessibility and precise specification of these path(s) are essential to ensure smooth operation.

### Usage

Here's an example usage of the UpstageDocumentParseReader.

```python
import os

os.environ["UPSTAGE_API_KEY"] = "YOUR_API_KEY"


from llama_index.readers.upstage import UpstageDocumentParseReader

file_path = "/PATH/TO/YOUR/FILE.pdf"

reader = UpstageDocumentParseReader()

# For improved memory efficiency, consider using the lazy_load_data method to load documents page by page.
docs = reader.load_data(file_path=file_path)

for doc in docs[:3]:
    print(doc)
```
