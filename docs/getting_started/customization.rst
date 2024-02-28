Customization Tutorial
======================
.. tip::
    If you haven't already, `install LlamaIndex <installation.html>`_ and complete the `starter tutorial <starter_example.html>`_. If you run into terms you don't recognize, check out the `high-level concepts <concepts.html>`_.

In this tutorial, we start with the code you wrote for the `starter example <starter_example.html>`_ and show you the most common ways you might want to customize it for your use case:

.. code-block:: python

    from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

    documents = SimpleDirectoryReader("data").load_data()
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine()
    response = query_engine.query("What did the author do growing up?")
    print(response)

-----------------

**"I want to parse my documents into smaller chunks"**

.. code-block:: python

    # Global settings
    from llama_index.core import Settings

    Settings.chunk_size = 512

    # Local settings
    from llama_index.core.node_parser import SentenceSplitter

    index = VectorStoreIndex.from_documents(
        documents, transformations=[SentenceSplitter(chunk_size=512)]
    )

-----------------

**"I want to use a different vector store"**

First, you can install the vector store you want to use. For example, to use Chroma as the vector store, you can install it using pip:

.. code-block:: shell

    pip install llama-index-vector-stores-chroma


To learn more about all integrations available, check out `LlamaHub <https://llamahub.ai>`_.

Then, you can use it in your code:

.. code-block:: python

    import chromadb
    from llama_index.vector_stores.chroma import ChromaVectorStore
    from llama_index.core import StorageContext

    chroma_client = chromadb.PersistentClient()
    chroma_collection = chroma_client.create_collection("quickstart")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

`StorageContext` defines the storage backend for where the documents, embeddings, and indexes are stored. You can learn more about `storage <../module_guides/storing/storing.html>`_ and `how to customize it <../module_guides/storing/customization.html>`_.

.. code-block:: python
    :emphasize-lines: 4

    from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

    documents = SimpleDirectoryReader("data").load_data()
    index = VectorStoreIndex.from_documents(
        documents, storage_context=storage_context
    )
    query_engine = index.as_query_engine()
    response = query_engine.query("What did the author do growing up?")
    print(response)

-----------------

**"I want to retrieve more context when I query"**

.. code-block:: python
    :emphasize-lines: 5

    from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

    documents = SimpleDirectoryReader("data").load_data()
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine(similarity_top_k=5)
    response = query_engine.query("What did the author do growing up?")
    print(response)

`as_query_engine` builds a default `retriever` and `query engine` on top of the index. You can configure the retriever and query engine by passing in keyword arguments. Here, we configure the retriever to return the top 5 most similar documents (instead of the default of 2). You can learn more about `retrievers <../module_guides/querying/retriever/retrievers.html>`_ and `query engines <../module_guides/querying/retriever/root.html>`_

-----------------

**"I want to use a different LLM"**

.. code-block:: python

    # Global settings
    from llama_index.core import Settings
    from llama_index.llms.ollama import Ollama

    Settings.llm = Ollama(model="mistral", request_timeout=60.0)

    # Local settings
    index.as_query_engine(llm=Ollama(model="mistral", request_timeout=60.0))

You can learn more about `customizing LLMs <../module_guides/models/llms.html>`_.

-----------------

**"I want to use a different response mode"**


.. code-block:: python
    :emphasize-lines: 5

    from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

    documents = SimpleDirectoryReader("data").load_data()
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine(response_mode="tree_summarize")
    response = query_engine.query("What did the author do growing up?")
    print(response)

You can learn more about `query engines <../module_guides/querying/querying.html>`_ and `response modes <../module_guides/deploying/query_engine/response_modes.html>`_.

-----------------

**"I want to stream the response back"**


.. code-block:: python
    :emphasize-lines: 5, 7

    from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

    documents = SimpleDirectoryReader("data").load_data()
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine(streaming=True)
    response = query_engine.query("What did the author do growing up?")
    response.print_response_stream()

You can learn more about `streaming responses <../module_guides/deploying/query_engine/streaming.html>`_.

-----------------

**"I want a chatbot instead of Q&A"**

.. code-block:: python
    :emphasize-lines: 5, 6, 9

    from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

    documents = SimpleDirectoryReader("data").load_data()
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_chat_engine()
    response = query_engine.chat("What did the author do growing up?")
    print(response)

    response = query_engine.chat("Oh interesting, tell me more.")
    print(response)

Learn more about the `chat engine <../module_guides/deploying/chat_engines/usage_pattern.html>`_.

-----------------

.. admonition:: Next Steps

    * want a thorough walkthrough of (almost) everything you can configure? Get started with `Understanding LlamaIndex <../understanding/understanding.html>`_.
    * want more in-depth understanding of specific modules? Check out the module guides in the left nav 👈
