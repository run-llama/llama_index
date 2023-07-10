Customization Tutorial
======================
.. tip::
    If you haven't, `install <installation.html>`_, complete `starter tutorial <starter_example.html>`_, and learn the `high-level concepts <concepts.html>`_ before you read this. It will make a lot more sense!

In this tutorial, we show the most common customizations with the `starter example <starter_example.md>`_:

.. code-block:: python

    from llama_index import VectorStoreIndex, SimpleDirectoryReader

    documents = SimpleDirectoryReader('data').load_data()
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine()
    response = query_engine.query("What did the author do growing up?")
    print(response)

-----------------

**"I want to parse my documents into smaller chunks"**

.. code-block:: python

    from llama_index import ServiceContext
    service_context = ServiceContext.from_defaults(chunk_size=1000)

.. tip::
    `ServiceContext` is a bundle of services and configurations used across a LlamaIndex pipeline,
    Learn more `here <../how_to/customization/service_context.html>`_.

.. code-block:: python
    :emphasize-lines: 4

    from llama_index import VectorStoreIndex, SimpleDirectoryReader

    documents = SimpleDirectoryReader('data').load_data()
    index = VectorStoreIndex.from_documents(documents, service_context=service_context)
    query_engine = index.as_query_engine()
    response = query_engine.query("What did the author do growing up?")
    print(response)

-----------------

**"I want to use a different vector store"**

.. code-block:: python

    import chromadb
    from llama_index.vector_stores import ChromaVectorStore
    from llama_index import StorageContext

    chroma_client = chromadb.Client()
    chroma_collection = chroma_client.create_collection("quickstart")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

.. tip::
    `StorageContext` defines the storage backend for where the documents, embeddings, and indexes are stored.
    Learn more `here <../core_modules/data_modules/storage/customization.html>`_.

.. code-block:: python
    :emphasize-lines: 4

    from llama_index import VectorStoreIndex, SimpleDirectoryReader

    documents = SimpleDirectoryReader('data').load_data()
    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
    query_engine = index.as_query_engine()
    response = query_engine.query("What did the author do growing up?")
    print(response)

-----------------

**"I want to retrieve more context when I query"**

.. code-block:: python
    :emphasize-lines: 5

    from llama_index import VectorStoreIndex, SimpleDirectoryReader

    documents = SimpleDirectoryReader('data').load_data()
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine(similarity_top_k=5)
    response = query_engine.query("What did the author do growing up?")
    print(response)

.. tip::
    `as_query_engine` builds a default retriever and query engine on top of the index.
    You can configure the retriever and query engine by passing in keyword arguments.
    Here, we configure the retriever to return the top 5 most similar documents (instead of the default of 2).
    Learn more about vector index `here <../core_modules/data_modules/index/vector_store_guide.html>`_.

-----------------

**"I want to use a different LLM"**

.. code-block:: python

    from llama_index import ServiceContext
    from llama_index.llms import PaLM
    service_context = ServiceContext.from_defaults(llm=PaLM())

.. tip::
    Learn more about customizing LLMs `here <../core_modules/service_modules/llms/usage_custom.html>`_.

.. code-block:: python
    :emphasize-lines: 5

    from llama_index import VectorStoreIndex, SimpleDirectoryReader

    documents = SimpleDirectoryReader('data').load_data()
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine(service_context=service_context)
    response = query_engine.query("What did the author do growing up?")
    print(response)

-----------------

**"I want to use a different response mode"**


.. code-block:: python
    :emphasize-lines: 5

    from llama_index import VectorStoreIndex, SimpleDirectoryReader

    documents = SimpleDirectoryReader('data').load_data()
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine(response_mode='tree_summarize')
    response = query_engine.query("What did the author do growing up?")
    print(response)

.. tip::
    Learn more about query engine usage pattern `here <../core_modules/query_modules/query_engine/usage_pattern.html>`_ and available response modes `here <../core_modules/query_modules/query_engine/response_modes.html>`_.

-----------------

**"I want to stream the response back"**


.. code-block:: python
    :emphasize-lines: 5, 7

    from llama_index import VectorStoreIndex, SimpleDirectoryReader

    documents = SimpleDirectoryReader('data').load_data()
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine(streaming=True)
    response = query_engine.query("What did the author do growing up?")
    response.print_response_stream()

.. tip::
    Learn more about streaming `here <../how_to/customization/streaming.html>`_.

-----------------

**"I want a chatbot instead of Q&A"**

.. code-block:: python
    :emphasize-lines: 5, 6, 9

    from llama_index import VectorStoreIndex, SimpleDirectoryReader

    documents = SimpleDirectoryReader('data').load_data()
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_chat_engine()
    response = query_engine.chat("What did the author do growing up?")
    print(response)

    response = query_engine.chat("Oh interesting, tell me more.")
    print(response)

.. tip::
    Learn more about chat engine usage pattern `here <../core_modules/query_modules/chat_engines/usage_pattern.html>`_.

-----------------

.. admonition:: Next Steps

    * want a thorough walkthrough of (almost) everything you can configure? Try the `end-to-end tutorial on basic usage pattern <../end_to_end_tutorials/usage_pattern.html>`_.
    * want more in-depth understanding of specific modules? Check out the module guides 👈