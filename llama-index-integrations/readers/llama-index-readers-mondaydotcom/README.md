# Monday Loader

```bash
pip install llama-index-readers-mondaydotcom
```

This loader loads data from monday.com. The user specifies an API token to initialize the MondayReader. They then specify a monday.com board id to load in the corresponding Document objects.

## Usage

Here's an example usage of the MondayReader.

```python
from llama_index.readers.mondaydotcom import MondayReader

reader = MondayReader("<monday_api_token>")
documents = reader.load_data("<board_id: int>")
```

Check out monday.com API docs - [here](https://developer.monday.com/apps/docs/mondayapi)

This loader is designed to be used as a way to load data into [LlamaIndex](https://github.com/run-llama/llama_index/tree/main/llama_index) and/or subsequently used as a Tool in a [LangChain](https://github.com/hwchase17/langchain) Agent. See [here](https://github.com/jerryjliu/llama_index) for examples.
