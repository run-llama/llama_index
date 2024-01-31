# Remote Page/File Loader

This loader makes it easy to extract the text from the links available in a webpage URL, and extract the links presents in the page. It's based on `RemoteReader` (reading single page), that is based on `SimpleDirectoryReader` (parsing the document if file is a pdf, etc). It is an all-in-one tool for (almost) any group of urls.

You can try with this MIT lecture link, it will be able to extract the syllabus, the PDFs, etc:
`https://ocw.mit.edu/courses/5-05-principles-of-inorganic-chemistry-iii-spring-2005/pages/syllabus/`

## Usage

You need to specify the parameter `depth` to specify how many levels of links you want to extract. For example, if you want to extract the links in the page, and the links in the links in the page, you need to specify `depth=2`.

```python
from llama_index import download_loader

RemoteDepthReader = download_loader("RemoteDepthReader")

loader = RemoteDepthReader()
documents = loader.load_data(
    url="https://ocw.mit.edu/courses/5-05-principles-of-inorganic-chemistry-iii-spring-2005/pages/syllabus/"
)
```

This loader is designed to be used as a way to load data into [LlamaIndex](https://github.com/run-llama/llama_index/tree/main/llama_index) and/or subsequently used as a Tool in a [LangChain](https://github.com/hwchase17/langchain) Agent. See [here](https://github.com/emptycrown/llama-hub/tree/main) for examples.
