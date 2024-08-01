# Preprocess Loader

```bash
pip install llama-index-readers-preprocess
```

[Preprocess](https://preprocess.co) is an API service that splits any kind of document into optimal chunks of text for use in language model tasks.
Given documents in input `Preprocess` splits them into chunks of text that respect the layout and semantics of the original document.
We split the content by taking into account sections, paragraphs, lists, images, data tables, text tables, and slides, and following the content semantics for long texts.
We support PDFs, Microsoft Office documents (Word, PowerPoint, Excel), OpenOffice documents (ods, odt, odp), HTML content (web pages, articles, emails), and plain text.

This loader integrates with the `Preprocess` API library to provide document conversion and chunking or to load already chunked files inside LlamaIndex.

## Requirements

Install the Python `Preprocess` library if it is not already present:

```
pip install pypreprocess
```

## Usage

To use this loader, you need to pass the `Preprocess API Key`.
When initializing `PreprocessReader`, you should pass your `API Key`, if you don't have it yet, please ask for one at [support@preprocess.co](mailto:support@preprocess.co). Without an `API Key`, the loader will raise an error.

To chunk a file pass a valid filepath and the reader will start converting and chunking it.
`Preprocess` will chunk your files by applying an internal `Splitter`. For this reason, you should not parse the document into nodes using a `Splitter` or applying a `Splitter` while transforming documents in your `IngestionPipeline`.

If you want to handle the nodes directly:

```python
from llama_index.core import VectorStoreIndex

from llama_index.readers.preprocess import PreprocessReader

# pass a filepath and get the chunks as nodes
loader = PreprocessReader(
    api_key="your-api-key", filepath="valid/path/to/file"
)
nodes = loader.get_nodes()

# import the nodes in a Vector Store with your configuration
index = VectorStoreIndex(nodes)
query_engine = index.as_query_engine()
```

By default load_data() returns a document for each chunk, remember to not apply any splitting to these documents

```python
from llama_index.core import VectorStoreIndex

from llama_index.readers.preprocess import PreprocessReader

# pass a filepath and get the chunks as nodes
loader = PreprocessReader(
    api_key="your-api-key", filepath="valid/path/to/file"
)
documents = loader.load_data()

# don't apply any Splitter parser to documents
# if you have an ingestion pipeline you should not apply a Splitter in the transformations
# import the documents in a Vector Store, if you set the service_context parameter remember to avoid including a splitter
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()
```

If you want to return only the extracted text and handle it with custom pipelines set `return_whole_document = True`

```python
# pass a filepath and get the chunks as nodes
loader = PreprocessReader(
    api_key="your-api-key", filepath="valid/path/to/file"
)
document = loader.load_data(return_whole_document=True)
```

If you want to load already chunked files you can do it via `process_id` passing it to the reader.

```python
# pass a process_id obtained from a previous instance and get the chunks as one string inside a Document
loader = PreprocessReader(api_key="your-api-key", process_id="your-process-id")
```

This loader is designed to be used as a way to load data into [LlamaIndex](https://github.com/run-llama/llama_index/).

## Other info

`PreprocessReader` is based on `pypreprocess` from [Preprocess](https://github.com/preprocess-co/pypreprocess) library.
For more information or other integration needs please check the [documentation](https://github.com/preprocess-co/pypreprocess).
