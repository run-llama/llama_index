# Hubspot Loader

```bash
pip install llama-index-readers-hubspot
```

This loader loads documents from Hubspot. The user specifies an access token to initialize the HubspotReader.

At the moment, this loader only supports access token authentication. To obtain an access token, you will need to create a private app by following instructions [here](https://developers.hubspot.com/docs/api/private-apps).

## Usage

Here's an example usage of the HubspotReader.

```python
import os

from llama_index.readers.hubspot import HubspotReader

reader = HubspotReader("<HUBSPOT_ACCESS_TOKEN>")
documents = reader.load_data()
```

This loader is designed to be used as a way to load data into [LlamaIndex](https://github.com/run-llama/llama_index/).
