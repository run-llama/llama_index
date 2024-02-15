# Hubspot Loader

This loader loads documents from Hubspot. The user specifies an access token to initialize the HubspotReader.

At the moment, this loader only supports access token authentication. To obtain an access token, you will need to create a private app by following instructions [here](https://developers.hubspot.com/docs/api/private-apps).

## Usage

Here's an example usage of the HubspotReader.

```python
from llama_index import download_loader
import os

HubspotReader = download_loader("HubspotReader")

reader = HubspotReader("<HUBSPOT_ACCESS_TOKEN>")
documents = reader.load_data()
```

This loader is designed to be used as a way to load data into [LlamaIndex](https://github.com/run-llama/llama_index/tree/main/llama_index) and/or subsequently used as a Tool in a [LangChain](https://github.com/hwchase17/langchain) Agent. See [here](https://github.com/emptycrown/llama-hub/tree/main) for examples.
