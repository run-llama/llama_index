# Using Vector Stores

GPT Index offers multiple integration points with vector stores / vector databases: 

1) GPT Index can load data from vector stores, similar to any other data connector. This data can then be used within GPT Index data structures.
2) GPT Index can use a vector store itself as an index. Like any other index, this index can store documents and be used to answer queries.


## Loading Data from Vector Stores using Data Connector
GPT Index supports loading data from the following sources. See [Data Connectors](data_connectors.md) for more details and API documentation.

- Qdrant (`QdrantReader`) [Installation](https://qdrant.tech/documentation/install/) [Python Client](https://qdrant.tech/documentation/install/#python-client)
- Weaviate (`WeaviateReader`). [Installation](https://weaviate.io/developers/weaviate/current/getting-started/installation.html). [Python Client](https://weaviate.io/developers/weaviate/current/client-libraries/python.html).
- Pinecone (`PineconeReader`). [Installation/Quickstart](https://docs.pinecone.io/docs/quickstart).
- Faiss (`FaissReader`). [Installation](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md).

NOTE: Both Pinecone and Faiss data loaders assume that the respective data sources only store vectors; text content is stored elsewhere. Therefore, both data loaders require that the user specifies an `id_to_text_map` in the load_data call.

For instance, this is an example usage of the Pinecone data loader `PineconeReader`:

![](/_static/vector_stores/pinecone_reader.png)

Qdrant stores both documents and vectors, so there is no need to provide `id_to_text_map`. This is an example of how to use it:

![](/_static/vector_stores/qdrant_reader.png)

NOTE: Since Weaviate can store a hybrid of document and vector objects, the user may either choose to explicitly specify `class_name` and `properties` in order to query documents, or they may choose to specify a raw GraphQL query. See below for usage.

![](/_static/vector_stores/weaviate_reader_0.png)
![](/_static/vector_stores/weaviate_reader_1.png)

[Example notebooks can be found here](https://github.com/jerryjliu/gpt_index/tree/main/examples/data_connectors).


(vector-store-index)=
## Using a Vector Store as an Index

GPT Index also supports using a vector store itself as an index. 
These are found in the following classes:
- `GPTSimpleVectorIndex`
- `GPTFaissIndex`
- `GPTWeaviateIndex`
- `GPTPineconeIndex`
- `GPTQdrantIndex`

Similar to any other index within GPT Index (tree, keyword table, list), this index can be constructed upon any collection
of documents. We use the vector store within the index to store embeddings for the input text chunks.

Once constructed, the index can be used for querying.

**Faiss Index Construction/Querying**
![](/_static/vector_stores/faiss_index_0.png)
![](/_static/vector_stores/faiss_index_1.png)

**Simple Index Construction/Querying**
![](/_static/vector_stores/simple_index_0.png)

**Weaviate Index Construction/Querying**
![](/_static/vector_stores/weaviate_index_0.png)

**Pinecone Index Construction/Querying**
![](/_static/vector_stores/pinecone_index_0.png)

**Qdrant Index Construction/Querying**
![](/_static/vector_stores/qdrant_index_0.png)

[Example notebooks can be found here](https://github.com/jerryjliu/gpt_index/tree/main/examples/vector_indices).
