# Snscrape twitter Loader

```bash
pip install llama-index-readers-snscrape-twitter
```

This loader loads documents from Twitter using the Snscrape Python package.

## Usage

Here's an example usage of the SnscrapeReader.

```python
import os

from llama_index.readers.snscrape_twitter import SnscrapeTwitterReader

loader = SnscrapeReader()
documents = loader.load_data(username="elonmusk", num_tweets=10)
```

This loader is designed to be used as a way to load data into [LlamaIndex](https://github.com/run-llama/llama_index/tree/main/llama_index) and/or subsequently used as a Tool in a [LangChain](https://github.com/hwchase17/langchain) Agent. See [here](https://github.com/emptycrown/llama-hub/tree/main) for examples.
