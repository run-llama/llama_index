# GPT Repository Loader

This loader is an adaptation of https://github.com/mpoon/gpt-repository-loader
to LlamaHub. Full credit goes to mpoon for coming up with this!

## Usage

To use this loader, you need to pass in a path to a local Git repository

```python
from llama_index import download_loader

GPTRepoReader = download_loader("GPTRepoReader")

loader = GPTRepoReader()
documents = loader.load_data(
    repo_path="/path/to/git/repo",
    preamble_str="<text to put at beginning of Document>",
)
```

This loader is designed to be used as a way to load data into [LlamaIndex](https://github.com/run-llama/llama_index/tree/main/llama_index) and/or subsequently used as a Tool in a [LangChain](https://github.com/hwchase17/langchain) Agent. See [here](https://github.com/emptycrown/llama-hub/tree/main) for examples.
