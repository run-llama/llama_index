# Zendesk Loader

```bash
pip install llama-index-readers-zendesk
```

This loader fetches the text from Zendesk help articles using the Zendesk API. It also uses the BeautifulSoup library to parse the HTML and extract the text from the articles.

## Usage

To use this loader, you need to pass in the subdomain of a Zendesk account. No authentication is required. You can also set the locale of articles as needed.

```python
from llama_index.readers.zendesk import ZendeskReader

loader = ZendeskReader(zendesk_subdomain="my_subdomain", locale="en-us")
documents = loader.load_data()
```

This loader is designed to be used as a way to load data into [LlamaIndex](https://github.com/run-llama/llama_index/).
