# Paged CSV Loader

```bash
pip install llama-index-readers-file
```

This loader extracts the text from a local .csv file by formatting each row in an LLM-friendly way and inserting it into a separate Document. A single local file is passed in each time you call `load_data`. For example, a Document might look like:

```
First Name: Bruce
Last Name: Wayne
Age: 28
Occupation: Unknown
```

## Usage

To use this loader, you need to pass in a `Path` to a local file.

```python
from pathlib import Path

from llama_index.readers.file import PagedCSVReader

loader = PagedCSVReader(encoding="utf-8")
documents = loader.load_data(file=Path("./transactions.csv"))
```

This loader is designed to be used as a way to load data into [LlamaIndex](https://github.com/jerryjliu/llama_index) and/or subsequently used as a Tool in a [LangChain](https://github.com/hwchase17/langchain) Agent. See [here](https://github.com/emptycrown/llama-hub/tree/main) for examples.
