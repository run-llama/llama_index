# Gmail Loader

`pip install llama-index-readers-google`

This loader searches your Gmail account and parses the resulting emails into `Document`s. The search query can include normal query params, like `from: email@example.com label:inbox`.

As a prerequisite, you will need to register with Google and generate a `credentials.json` file in the directory where you run this loader. See [here](https://developers.google.com/workspace/guides/create-credentials) for instructions.

## Usage

To use this loader, you simply need to pass in a search query string.

```python
from llama_index.readers.google import GmailReader

loader = GmailReader(query="from: me label:inbox")
documents = loader.load_data()
```

## Examples

This loader is designed to be used as a way to load data into [LlamaIndex](https://github.com/run-llama/llama_index/tree/main/llama_index).
