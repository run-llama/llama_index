# Guru Loader

This loader loads documents from [Guru](https://www.getguru.com/). The user specifies a username and api key to initialize the GuruReader.

Note this is not your password. You need to create a new api key in the admin tab of the portal.

## Usage

Here's an example usage of the GuruReader.

```python
from llama_index import download_loader

GuruReader = download_loader("GuruReader")

reader = GuruReader(username="<GURU_USERNAME>", api_key="<GURU_API_KEY>")

# Load all documents in a collection
documents = reader.load_data(
    collection_ids=["<COLLECTION_ID_1>", "<COLLECTION_ID_2>"]
)

# Load specific cards by card id
documents = reader.load_data(card_ids=["<CARD_ID_1>", "<CARD_ID_2>"])
```

This loader is designed to be used as a way to load data into [LlamaIndex](https://github.com/run-llama/llama_index/tree/main/llama_index) and/or subsequently used as a Tool in a [LangChain](https://github.com/hwchase17/langchain) Agent. See [here](https://github.com/emptycrown/llama-hub/tree/main) for examples.
