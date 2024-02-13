# Intercom Loader

This loader fetches the text from Intercom help articles using the Intercom API. It also uses the BeautifulSoup library to parse the HTML and extract the text from the articles.

## Usage

To use this loader, you need to pass in an Intercom account access token.

```python
from llama_index import download_loader

IntercomReader = download_loader("IntercomReader")

loader = IntercomReader(intercom_access_token="my_access_token")
documents = loader.load_data()
```

This loader is designed to be used as a way to load data into [LlamaIndex](https://github.com/run-llama/llama_index/tree/main/llama_index) and/or subsequently used as a Tool in a [LangChain](https://github.com/hwchase17/langchain) Agent. See [here](https://github.com/emptycrown/llama-hub/tree/main) for examples.
