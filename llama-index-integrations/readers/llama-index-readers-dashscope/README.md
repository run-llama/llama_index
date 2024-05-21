# LlamaIndex Readers Integration: Dashscope

## Installation

```shelll
pip install llama-index-readers-dashscope
```

## Usage

```python
from llama_index.readers.dashscope.base import DashScopeParse
from llama_index.readers.dashscope.utils import ResultType

file_list = [
    # your files (accept doc, docx, pdf)
]

parse = DashScopeParse(result_type=ResultType.DASHCOPE_DOCMIND)
documents = parse.load_data(file_path=file_list)
```

## Reader Setting:

A full list of retriever settings/kwargs is below:

- api_key: Optional[str] -- Your dashscope API key, which can be passed in through environment variables or parameters.
  The parameter settings will override the results from the environment variables
- workspace_id: Optional[str] -- Your dashscope workspace_id, which can be passed in through environment variables or
  parameters. The parameter settings will override the results from the environment variables
- base_url: Optional[str] -- The base url for the Dashscope API. The default value is "https://dashscope.aliyuncs.com".
  The parameter settings will override the results from the environment variables.
- result_type: Optional[ResultType] -- The result type for the parser. The default value is ResultType.DASHCOPE_DOCMIND.
- num_workers: Optional[int] -- The number of workers to use sending API requests for parsing. The default value is 4,
  greater than 0, less than 10.
- check_interval: Optional[int] -- The interval in seconds to check if the parsing is done. The default value is 5.
- max_timeout: Optional[int] -- The maximum timeout in seconds to wait for the parsing to finish. The default value is 3600.
- verbose: Optional[bool] -- Whether to print the progress of the parsing. The default value is True.
- show_progress: Optional[bool] -- Show progress when parsing multiple files. The default value is True.
- ignore_errors: Optional[bool] -- Whether or not to ignore and skip errors raised during parsing. The default value is
  True.

## Reader Input:

- file_path: Union[str, List[str]] -- The file path or list of file paths to parse.

## Reader Output:

- List[llama_index.core.schema.Document] -- The list of documents parsed from the file.
  - text: str -- The text of the document from DASHCOPE_DOCMIND.
