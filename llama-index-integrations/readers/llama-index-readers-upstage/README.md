# UpstageLayoutAnalysisReader

```bash
pip install llama-index-readers-upstage
```

This reader loads document files and detects elements such as text, tables, and figures using the Upstage Layout Analysis API. Users wishing to utilize this reader must obtain an API key from the [Upstage console](https://console.upstage.ai).

### Construction

The `UpstageLayoutAnalysisReader` is equipped with the following three optional parameters during instantiation:

- `api_key`: This parameter is designed to accept a string that serves as an API access token. If the API key has already been registered in the environment variable `UPSTAGE_API_KEY`, there is no necessity to input it again during the reader's configuration.

- `use_ocr`: By default, the Upstage Layout Analysis API accesses and utilizes text information directly from digitally-born PDF documents when `use_ocr = False`. If `use_ocr` is set to `True`, Optical Character Recognition (OCR) is activated, allowing the OCR model to detect and extract text from any type of document file, including those in image formats. However, it is important to note that utilizing the OCR feature may lead to increased processing times due to the inference demands of the OCR model.

- `exclude`: This parameter allows users to specifically filter out and exclude certain categories of document elements. The `exclude` parameter requires a list of categories to be provided. The default setting is `["header", "footer"]`, implying that the elements labeled as "header" and "footer" will not be included in the output. Available categories for exclusion include:
  - "paragraph"
  - "caption"
  - "table"
  - "figure"
  - "equation"
  - "footer"
  - "header"

Each of these parameters enhances the flexibility and customization of the document processing capabilities offered by the `UpstageLayoutAnalysisReader`.

### `load_data`

The `load_data` function, encompassed within the `UpstageLayoutAnalysisReader`, extends from the `BaseReader` class. The `lazy_load_data` function mirrors the functionalities of the `load_data` function but with an enhanced focus on efficiency and lazy loading, making it particularly suitable for handling large files. Utilizing this function effectively necessitates a thorough understanding of its parameters and their respective expected inputs to harness its full potential:

- **`file_path` (required):** This critical parameter accepts either a single string or `pathlib.Path` object, or a list comprising multiple of these elements, representing the path(s) to the file(s) intended for loading. Proper accessibility and precise specification of these path(s) are essential to ensure smooth operation.

- **`output_type` (optional):** This parameter tailors the format of the output data. By default, it is configured to "html," but it can be adjusted to "text". Moreover, for more refined control, it can be specified through a dictionary format, such as `{"category": "output_type", ...}`. The allowed values for `output_type` within this setup are "text" or "html". For categories not specified in the dictionary format of `output_type` and not listed in the `exclude` during construction, the default output type will be `html`.

- **`split` (optional):** This parameter manages the segmentation of the data during the loading process. The selection made here should align with the user's data handling strategy to ensure effective data management. The available splitting modes are:

  - **"none":** Applies no splitting to the data (default setting).
  - **"page":** Segments the document page by page.
  - **"element":** Splits the document by individual elements (paragraphs, tables, etc.).

Understanding and setting these parameters correctly allow users to optimize data loading and processing according to their specific requirements and workflows.

## Usage

Here's an example usage of the UpstageLayoutAnalysisReader.

```python
import os

os.environ["UPSTAGE_API_KEY"] = "YOUR_API_KEY"


from llama_index.readers.upstage import UpstageLayoutAnalysisReader

file_path = "/PATH/TO/YOUR/FILE.pdf"

reader = UpstageLayoutAnalysisReader(
    use_ocr=False, exclude=["header", "footer"]
)

# For improved memory efficiency, consider using the lazy_load_data method to load documents page by page.
docs = reader.load_data(
    file_path=file_path, split="element", output_type={"paragraph": "text"}
)

for doc in docs[:3]:
    print(doc)
```
