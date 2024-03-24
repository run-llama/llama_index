# Trello Loader

```bash
pip install llama-index-readers-trello
```

This loader loads documents from Trello. The user specifies an API key and API token to initialize the TrelloReader. They then specify a board_id to
load in the corresponding Document objects representing Trello cards.

## Usage

Here's an example usage of the TrelloReader.

```python
import os

from llama_index.readers.trello import TrelloReader

reader = TrelloReader("<Trello_API_KEY>", "<Trello_API_TOKEN>")
documents = reader.load_data(board_id="<BOARD_ID>")
```

This loader is designed to be used as a way to load data into LlamaIndex and/or subsequently used as a Tool in a LangChain Agent. See here for
examples.
