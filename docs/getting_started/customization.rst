Customization Tutorial
======================

    If you haven't, `install <installation.html>`_, complete `starter tutorial <starter_example.html>`_, and learn the `high-level concepts <concepts.html>`_ before you read this. It will make a lot more sense!

In this tutorial, we will show you how to customize the `starter example <starter_example.md>`_:

.. code-block:: python

    from llama_index import VectorStoreIndex, SimpleDirectoryReader

    documents = SimpleDirectoryReader('data').load_data()
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine()
    response = query_engine.query("What did the author do growing up?")
    print(response)

**"I want to parse my documents into smaller chunks"**

.. code-block:: python

    from llama_index import ServiceContext
    service_context = ServiceContext.from_defaults(chunk_size=1000)

.. code-block:: python
    :emphasize-lines: 4

    from llama_index import VectorStoreIndex, SimpleDirectoryReader

    documents = SimpleDirectoryReader('data').load_data()
    index = VectorStoreIndex.from_documents(documents, service_context=service_context)
    query_engine = index.as_query_engine()
    response = query_engine.query("What did the author do growing up?")
    print(response)

**"I want to use a different vector store"**

.. code-block:: python

    import chromadb
    from llama_index.vector_stores import ChromaVectorStore
    from llama_index import StorageContext

    chroma_client = chromadb.Client()
    chroma_collection = chroma_client.create_collection("quickstart")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

.. code-block:: python
    :emphasize-lines: 4

    from llama_index import VectorStoreIndex, SimpleDirectoryReader

    documents = SimpleDirectoryReader('data').load_data()
    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
    query_engine = index.as_query_engine()
    response = query_engine.query("What did the author do growing up?")
    print(response)

**"I want to retrieve more context when I query"**

.. code-block:: python
    :emphasize-lines: 5

    from llama_index import VectorStoreIndex, SimpleDirectoryReader

    documents = SimpleDirectoryReader('data').load_data()
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine(similarity_top_k=5)
    response = query_engine.query("What did the author do growing up?")
    print(response)

**"I want to use a different response mode"**


.. code-block:: python
    :emphasize-lines: 5

    from llama_index import VectorStoreIndex, SimpleDirectoryReader

    documents = SimpleDirectoryReader('data').load_data()
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine(response_mode='tree_summarize')
    response = query_engine.query("What did the author do growing up?")
    print(response)

**"I want to stream the response back"**


.. code-block:: python
    :emphasize-lines: 5, 7

    from llama_index import VectorStoreIndex, SimpleDirectoryReader

    documents = SimpleDirectoryReader('data').load_data()
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine(streaming=True)
    response = query_engine.query("What did the author do growing up?")
    response.print_response_stream()