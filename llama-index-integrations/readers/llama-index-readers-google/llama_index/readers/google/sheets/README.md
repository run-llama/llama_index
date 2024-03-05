# Google Sheets Loader

This loader reads your upcoming Google Sheets and parses the relevant info into `Documents`. 

As a prerequisite, you will need to register with Google and generate a `credentials.json` file in the directory where you run this loader. See [here](https://developers.google.com/workspace/guides/create-credentials) for instructions.

## Usage

Here's an example usage of the GoogleSheetsReader.

```python
from llama_index import download_loader

GoogleSheetsReader = download_loader('GoogleSheetsReader')

loader = GoogleSheetsReader()
documents = loader.load_data()
```

## Example

This loader is designed to be used as a way to load data into [LlamaIndex](https://github.com/run-llama/llama_index/tree/main/llama_index) and/or subsequently used as a Tool in a [LangChain](https://github.com/hwchase17/langchain) Agent.

### LlamaIndex

```python
from llama_index import VectorStoreIndex, download_loader

GoogleSheetsReader = download_loader('GoogleSheetsReader')

loader = GoogleSheetsReader()
documents = loader.load_data()
index = VectorStoreIndex.from_documents(documents)
index.query('When am I meeting Gordon?')
```
