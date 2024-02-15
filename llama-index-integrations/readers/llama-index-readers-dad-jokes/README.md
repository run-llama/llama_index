# DadJoke Loader

This loader fetches a joke from icanhazdadjoke.

## Usage

To use this loader, load it.

```python
from llama_index import download_loader

DadJokesReader = download_loader("DadJokesReader")

loader = DadJokesReader()
documents = loader.load_data()
```

This loader is designed to be used as a way to load data into [LlamaIndex](https://github.com/run-llama/llama_index/tree/main/llama_index) and/or subsequently used as a Tool in a [LangChain](https://github.com/hwchase17/langchain) Agent. See [here](https://github.com/emptycrown/llama-hub/tree/main) for examples.
