# Memos Loader

```bash
pip install llama-index-readers-memos
```

This loader fetches text from self-hosted [memos](https://github.com/usememos/memos).

## Usage

To use this loader, you need to specify the host where memos is deployed. If you need to filter, pass the [corresponding parameter](https://github.com/usememos/memos/blob/4fe8476169ecd2fc4b164a25611aae6861e36812/api/memo.go#L76) in `load_data`.

```python
from llama_index.readers.memos import MemosReader

loader = MemosReader("https://demo.usememos.com/")
documents = loader.load_data({"creatorId": 101})
```

This loader is designed to be used as a way to load data into [LlamaIndex](https://github.com/run-llama/llama_index/).
