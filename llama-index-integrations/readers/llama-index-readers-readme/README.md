# Readme.com Loader

```bash
pip install llama-index-readers-readme
```

This loader fetches the text from [Readme](https://readme.com/) docs guides using the Readme API. It also uses the BeautifulSoup library to parse the HTML and extract the text from the docs.

## Usage

To use this loader, you need to pass in the API Key of a Readme account.

```python
from llama_index.readers.readme import ReadmeReader

loader = ReadmeReader(api_key="YOUR_API_KEY")
documents = loader.load_data()
```

This loader is designed to be used as a way to load data into [LlamaIndex](https://github.com/run-llama/llama_index/tree/main/llama_index) and/or subsequently used as a Tool in a [LangChain](https://github.com/hwchase17/langchain) Agent. See [here](https://github.com/emptycrown/llama-hub/tree/main) for examples.
