# Remote Page/File Loader

This loader makes it easy to extract the text from any remote page or file using just its url. If there's a file at the url, this loader will download it temporarily and parse it using `SimpleDirectoryReader`. It is an all-in-one tool for (almost) any url.

As a result, any page or type of file is supported. For instance, if a `.txt` url such as a [Project Gutenberg book](https://www.gutenberg.org/cache/epub/69994/pg69994.txt) is passed in, the text will be parsed as is. On the other hand, if a hosted .mp3 url is passed in, it will be downloaded and parsed using `AudioTranscriber`.

## Usage

To use this loader, you need to pass in a `Path` to a local file. Optionally, you may specify a `file_extractor` for the `SimpleDirectoryReader` to use, other than the default one.

```python
from llama_index import download_loader

RemoteReader = download_loader("RemoteReader")

loader = RemoteReader()
documents = loader.load_data(
    url="https://en.wikipedia.org/wiki/File:Example.jpg"
)
```

This loader is designed to be used as a way to load data into [LlamaIndex](https://github.com/run-llama/llama_index/tree/main/llama_index) and/or subsequently used as a Tool in a [LangChain](https://github.com/hwchase17/langchain) Agent. See [here](https://github.com/emptycrown/llama-hub/tree/main) for examples.
