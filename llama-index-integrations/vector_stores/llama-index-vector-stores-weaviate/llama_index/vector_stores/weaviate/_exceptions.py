class AsyncClientNotProvidedError(Exception):
    """Exception raised when the async weaviate client was not provided via  the `weaviate_aclient` parameter."""

    def __init__(
        self,
        message="Async method called without WeaviateAsyncClient provided. Pass the async weaviate client to be used via `weaviate_aclient` to the constructor of WeaviateVectorStore.",
    ) -> None:
        self.message = message
        super().__init__(self.message)
