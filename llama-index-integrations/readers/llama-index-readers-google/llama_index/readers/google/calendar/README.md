# Google Calendar Loader

This loader reads your upcoming Google Calendar events and parses the relevant info into `Documents`. 

As a prerequisite, you will need to register with Google and generate a `credentials.json` file in the directory where you run this loader. See [here](https://developers.google.com/workspace/guides/create-credentials) for instructions.

## Usage

Here's an example usage of the GoogleCalendar. It will retrieve up to 100 future events, unless an optional `number_of_results` argument is passed. It will also retrieve only future events, unless an optional `start_date` argument is passed.

```python
from llama_index import download_loader

GoogleCalendarReader = download_loader('GoogleCalendarReader')

loader = GoogleCalendarReader()
documents = loader.load_data()
```

## Example

This loader is designed to be used as a way to load data into [LlamaIndex](https://github.com/run-llama/llama_index/tree/main/llama_index) and/or subsequently used as a Tool in a [LangChain](https://github.com/hwchase17/langchain) Agent.

### LlamaIndex

```python
from llama_index import VectorStoreIndex, download_loader

GoogleCalendarReader = download_loader('GoogleCalendarReader')

loader = GoogleCalendarReader()
documents = loader.load_data()
index = VectorStoreIndex.from_documents(documents)
index.query('When am I meeting Gordon?')
```
