# Ingestion

Before your chosen LLM can act on your data, you first need to build an ingestion pipeline over that data to process it and load it into a storage system. This has parallels to data cleaning/feature engineering pipelines in the ML world, or ETL pipelines in the traditional data setting.

This ingestion pipeline consists of three main stages:

1. Load the data
2. Transform the data
3. Index and store the data

We cover indexing/storage in [future](/understanding/indexing/indexing.md) [sections](/understanding/storing/storing.md). In this guide we'll mostly talk about loaders and transformations.

## Loaders

Before your chosen LLM can act on your data you need to load it. The way LlamaIndex does this is via data connectors, also called `Reader`. Data connectors ingest data from different data sources and format the data into `Document` objects. A `Document` is a collection of data (currently text, and in future, images and audio) and metadata about that data.

### Loading using SimpleDirectoryReader

The easiest reader to use is our SimpleDirectoryReader, which creates documents out of every file in a given directory. It is built in to LlamaIndex and can read a variety of formats including Markdown, PDFs, Word documents, PowerPoint decks, images, audio and video.

```python
from llama_index import SimpleDirectoryReader

documents = SimpleDirectoryReader("./data").load_data()
```

### Using Readers from LlamaHub

Because there are so many possible places to get data, they are not all built-in. Instead, you download them from our registry of data connectors, [LlamaHub](/understanding/loading/llamahub.md).

In this example LlamaIndex downloads and installs the connector called [DatabaseReader](https://llamahub.ai/l/database), which runs a query against a SQL database and returns every row of the results as a `Document`:

```python
from llama_index import download_loader

DatabaseReader = download_loader("DatabaseReader")

reader = DatabaseReader(
    scheme=os.getenv("DB_SCHEME"),
    host=os.getenv("DB_HOST"),
    port=os.getenv("DB_PORT"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASS"),
    dbname=os.getenv("DB_NAME"),
)

query = "SELECT * FROM users"
documents = reader.load_data(query=query)
```

There are hundreds of connectors to use on [LlamaHub](https://llamahub.ai)!

## Transformations

After the data is loaded, you then need to process and transform your data before putting it into a storage system. These transformations include chunking, extracting metadata, and embedding each chunk. This is necessary to make sure that the data can be retrieved, and used optimally by the LLM.

Transformation input/outputs are `Node` objects (a `Document` is a subclass of a `Node`). Transformations can also be stacked and reordered.

We have both a high-level and lower-level API for transforming documents.

### High-Level Transformation API

Indexes have a `.from_documents()` method which accepts an array of Document objects and will correctly parse and chunk them up. However, sometimes you will want greater control over how your documents are split up.

```python
from llama_index import VectorStoreIndex

vector_index = VectorStoreIndex.from_documents(documents)
vector_index.as_query_engine()
```

Under the hood, this splits your Document into Node objects, which are similar to Documents (they contain text and metadata) but have a relationship to their parent Document.

### Lower-Level Transformation API

You can also define these steps explicitly.

You can do this by either using our transformation modules (text splitters, metadata extractors, etc.) as standalone components, or compose them in our declarative Transformation Pipeline interface.

<TODO: LEAVING THIS FOR NOW, FINISH THIS PART>

The way in which your text is split up can have a large effect on the performance of your application in terms of accuracy and relevance of results returned. The defaults work well for simple text documents, so depending on what your data looks like you will sometimes want to modify the default ways in which your documents are split up.

In this example, you load your documents, then create a SimpleNodeParser configured with a custom `chunk_size` and `chunk_overlap` (1024 and 20 are the defaults). You then assign the node parser to a `ServiceContext` and then pass it to your indexer:

```python
from llama_index import SimpleDirectoryReader, VectorStoreIndex, ServiceContext
from llama_index.text_splitter import SentenceSplitter

documents = SimpleDirectoryReader("./data").load_data()

text_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=10)
service_context = ServiceContext.from_defaults(text_splitter=text_splitter)

index = VectorStoreIndex.from_documents(
    documents, service_context=service_context
)
```

```{tip}
Remember, a ServiceContext is a simple bundle of configuration data passed to many parts of LlamaIndex.
```

You can learn more about [customizing your node parsing](/module_guides/loading/node_parsers/root.md)

## Creating and passing Nodes directly

If you want to, you can create nodes directly and pass a list of Nodes directly to an indexer:

```python
from llama_index.schema import TextNode

node1 = TextNode(text="<text_chunk>", id_="<node_id>")
node2 = TextNode(text="<text_chunk>", id_="<node_id>")

index = VectorStoreIndex([node1, node2])
```

## Creating Nodes from Documents directly

Using an `IngestionPipeline`, you can have more control over how nodes are created.

```python
from llama_index import Document
from llama_index.text_splitter import SentenceSplitter
from llama_index.ingestion import IngestionPipeline

# create the pipeline with transformations
pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(chunk_size=25, chunk_overlap=0),
    ]
)

# run the pipeline
nodes = pipeline.run(documents=[Document.example()])
```

You can learn more about the [`IngestionPipeline` here.](/module_guides/loading/ingestion_pipeline/root.md)

## Customizing Documents

When creating documents, you can also attach useful metadata that can be used at the querying stage. Any metadata added to a Document will be copied to the Nodes that get created from that document.

```python
document = Document(
    text="text",
    metadata={"filename": "<doc_file_name>", "category": "<category>"},
)
```

More about this can be found in [customizing Documents](/module_guides/loading/documents_and_nodes/usage_documents.md).

```{toctree}
---
maxdepth: 1
hidden: true
---
/understanding/loading/llamahub.md
```
