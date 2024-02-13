# Zendesk Loader

This loader fetches the text from Zendesk help articles using the Zendesk API. It also uses the BeautifulSoup library to parse the HTML and extract the text from the articles.

## Usage

To use this loader, you need to pass in the subdomain of a Zendesk account. No authentication is required. You can also set the locale of articles as needed.

```python
from llama_index import download_loader

ZendeskReader = download_loader("ZendeskReader")

loader = ZendeskReader(zendesk_subdomain="my_subdomain", locale="en-us")
documents = loader.load_data()
```

This loader is designed to be used as a way to load data into [LlamaIndex](https://github.com/run-llama/llama_index/tree/main/llama_index) and/or subsequently used as a Tool in a [LangChain](https://github.com/hwchase17/langchain) Agent. See [here](https://github.com/emptycrown/llama-hub/tree/main) for examples.
