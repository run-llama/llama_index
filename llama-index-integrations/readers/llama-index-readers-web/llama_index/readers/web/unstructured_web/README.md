# Unstructured.io URL Loader

```bash
pip install llama-index-readers-web
```

This loader extracts the text from URLs using [Unstructured.io](https://github.com/Unstructured-IO/unstructured). The partition_html function partitions an HTML document and returns a list
of document Element objects.

## Usage

```python
from llama_index.readers.web import UnstructuredURLLoader

urls = [
    "https://www.understandingwar.org/backgrounder/russian-offensive-campaign-assessment-february-8-2023",
    "https://www.understandingwar.org/backgrounder/russian-offensive-campaign-assessment-february-9-2023",
]

loader = UnstructuredURLLoader(
    urls=urls, continue_on_failure=False, headers={"User-Agent": "value"}
)
loader.load_data()
```

> Note:
>
> If the version of unstructured is less than 0.5.7 and headers is not an empty dict, the user will see a warning (You are using old version of unstructured. The headers parameter is ignored).
>
> If the user will create the object of UnstructuredURLLoader without the headers parameter or with an empty dict, he will not see the warning.
