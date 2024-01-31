# Firebase Realtime Database Loader

This loader retrieves documents from Firebase Realtime Database. The user specifies the Firebase Realtime Database URL and, optionally, the path to a service account key file for authentication.

## Usage

Here's an example usage of the FirebaseRealtimeDatabaseReader.

```python
from llama_index import download_loader

FirebaseRealtimeDatabaseReader = download_loader(
    "FirebaseRealtimeDatabaseReader"
)

database_url = "<database_url>"
service_account_key_path = "<service_account_key_path>"
path = "<path>"
reader = FirebaseRealtimeDatabaseReader(database_url, service_account_key_path)
documents = reader.load_data(path)
```

This loader is designed to be used as a way to load data into [LlamaIndex](https://github.com/run-llama/llama_index/tree/main/llama_index) and/or subsequently used as a Tool in a [LangChain](https://github.com/hwchase17/langchain) Agent. See [here](https://github.com/emptycrown/llama-hub/tree/main) for examples.
