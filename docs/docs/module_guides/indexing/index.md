# Indexing

## Concept

An `Index` is a data structure that allows us to quickly retrieve relevant context for a user query.
For LlamaIndex, it's the core foundation for retrieval-augmented generation (RAG) use-cases.

At a high-level, `Indexes` are built from [Documents](../loading/documents_and_nodes/index.md).
They are used to build [Query Engines](../deploying/query_engine/index.md) and [Chat Engines](../deploying/chat_engines/index.md)
which enables question & answer and chat over your data.

Under the hood, `Indexes` store data in `Node` objects (which represent chunks of the original documents), and expose a [Retriever](../querying/retriever/index.md) interface that supports additional configuration and automation.

The most common index by far is the `VectorStoreIndex`; the best place to start is the [VectorStoreIndex usage guide](vector_store_index.md).

For other indexes, check out our guide to [how each index works](index_guide.md) to help you decide which one matches your use-case.

## Other Index resources

See the [modules guide](./modules.md).
