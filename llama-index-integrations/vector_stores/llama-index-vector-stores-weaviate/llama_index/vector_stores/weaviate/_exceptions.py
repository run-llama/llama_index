class SyncClientNotProvidedError(Exception):
    """Exception raised when no synchronous weaviate client was provided via  the `weaviate_client` parameter."""

    def __init__(
        self,
        message="Sync method called without a synchronous WeaviateClient provided. Either switch to using async methods together with a provided WeaviateAsyncClient or provide a synchronous WeaviateClient via `weaviate_client` to the constructor of WeaviateVectorStore.",
    ) -> None:
        self.message = message
        super().__init__(self.message)


class AsyncClientNotProvidedError(Exception):
    """Exception raised when the async weaviate client was not provided via  the `weaviate_client` parameter."""

    def __init__(
        self,
        message="Async method called without WeaviateAsyncClient provided. Pass the async weaviate client to be used via `weaviate_client` to the constructor of WeaviateVectorStore.",
    ) -> None:
        self.message = message
        super().__init__(self.message)
