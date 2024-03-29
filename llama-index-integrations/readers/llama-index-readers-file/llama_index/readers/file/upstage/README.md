# Upstage Document Reader

```bash
pip install llama-index-readers-file
```

The [Upstage Layout Analyzer API](https://developers.upstage.ai/docs/capabilities/layout-analyzer) enables this reader to extract text from various unstructured files, including PDFs and images. Future updates aim to support more document types, such as MS Word and PowerPoint files.

## Usage

To utilize this reader, you first need to obtain an API key from the [Upstage console](https://console.upstage.ai/). Next, input the API key into the Upstage document reader constructor. This API key grants access to the Upstage Layout Analyzer API, allowing it to extract text from your target files. Utilize the load function to extract text from your file in Llama document format.

There are two types of control parameters. The `output_type` parameter offers two options: `text` and `html`. Choose the `text` option for **regular text** extraction, or select `html` to obtain text along with **layout information in HTML format**. The `split` option determines the unit of document division. Choosing the `page` option divides text **by pages**, while selecting the `element` option splits **every element** in the document. If you prefer not to split, simply input `none`.

```python
from llama_index.readers.file import UpstageDocumentReader

api_key = "YOUR_API_KEY"

reader = UpstageDocumentReader(api_key)
docs = reader.load("/PATH/TO/FILE", output_type="html", split="page")
```
