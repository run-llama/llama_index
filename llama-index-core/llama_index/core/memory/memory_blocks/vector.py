from typing import Any, Dict, List, Optional

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.base.llms.types import ChatMessage, TextBlock
from llama_index.core.bridge.pydantic import Field, field_validator
from llama_index.core.memory.memory import BaseMemoryBlock
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.prompts import (
    BasePromptTemplate,
    RichPromptTemplate,
    PromptTemplate,
)
from llama_index.core.schema import TextNode, NodeWithScore
from llama_index.core.settings import Settings
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    MetadataFilter,
    MetadataFilters,
    VectorStoreQuery,
)

DEFAULT_RETRIEVED_TEXT_TEMPLATE = RichPromptTemplate("{{ text }}")


def get_default_embed_model() -> BaseEmbedding:
    return Settings.embed_model


class VectorMemoryBlock(BaseMemoryBlock[str]):
    """
    A memory block that retrieves relevant information from a vector store.

    This block stores conversation history in a vector store and retrieves
    relevant information based on the most recent messages.
    """

    name: str = Field(
        default="RetrievedMessages", description="The name of the memory block."
    )
    vector_store: BasePydanticVectorStore = Field(
        description="The vector store to use for retrieval."
    )
    embed_model: BaseEmbedding = Field(
        default_factory=get_default_embed_model,
        description="The embedding model to use for encoding queries and documents.",
    )
    similarity_top_k: int = Field(
        default=2, description="Number of top results to return."
    )
    retrieval_context_window: int = Field(
        default=5,
        description="Maximum number of messages to include for context when retrieving.",
    )
    format_template: BasePromptTemplate = Field(
        default=DEFAULT_RETRIEVED_TEXT_TEMPLATE,
        description="Template for formatting the retrieved information.",
    )
    node_postprocessors: List[BaseNodePostprocessor] = Field(
        default_factory=list,
        description="List of node postprocessors to apply to the retrieved nodes containing messages.",
    )
    query_kwargs: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional keyword arguments for the vector store query.",
    )

    @field_validator("vector_store", mode="before")
    def validate_vector_store(cls, v: Any) -> "BasePydanticVectorStore":
        if not isinstance(v, BasePydanticVectorStore):
            raise ValueError("vector_store must be a BasePydanticVectorStore")
        if not v.stores_text:
            raise ValueError(
                "vector_store must store text to be used as a retrieval memory block"
            )

        return v

    @field_validator("format_template", mode="before")
    @classmethod
    def validate_format_template(cls, v: Any) -> "BasePromptTemplate":
        if isinstance(v, str):
            if "{{" in v and "}}" in v:
                v = RichPromptTemplate(v)
            else:
                v = PromptTemplate(v)

        return v

    def _get_text_from_messages(self, messages: List[ChatMessage]) -> str:
        """Get the text from the messages."""
        text = ""
        for message in messages:
            for block in message.blocks:
                if isinstance(block, TextBlock):
                    text += block.text

        return text

    async def _aget(
        self,
        messages: Optional[List[ChatMessage]] = None,
        session_id: Optional[str] = None,
        **block_kwargs: Any,
    ) -> str:
        """Retrieve relevant information based on recent messages."""
        if not messages or len(messages) == 0:
            return ""

        # Use the last message or a context window of messages for the query
        if (
            self.retrieval_context_window > 1
            and len(messages) >= self.retrieval_context_window
        ):
            context = messages[-self.retrieval_context_window :]
        else:
            context = messages

        query_text = self._get_text_from_messages(context)
        if not query_text:
            return ""

        # Handle filtering by session_id
        if session_id is not None:
            filter = MetadataFilter(key="session_id", value=session_id)
            if "filters" in self.query_kwargs and isinstance(
                self.query_kwargs["filters"], MetadataFilters
            ):
                self.query_kwargs["filters"].filters.append(filter)
            else:
                self.query_kwargs["filters"] = MetadataFilters(filters=[filter])

        # Create and execute the query
        query_embedding = await self.embed_model.aget_query_embedding(query_text)
        query = VectorStoreQuery(
            query_str=query_text,
            query_embedding=query_embedding,
            similarity_top_k=self.similarity_top_k,
            **self.query_kwargs,
        )

        results = await self.vector_store.aquery(query)
        nodes_with_scores = [
            NodeWithScore(node=node, score=score)
            for node, score in zip(results.nodes or [], results.similarities or [])
        ]
        if not nodes_with_scores:
            return ""

        # Apply postprocessors
        for postprocessor in self.node_postprocessors:
            nodes_with_scores = await postprocessor.apostprocess_nodes(
                nodes_with_scores, query_str=query_text
            )

        # Format the results
        retrieved_text = "\n\n".join([node.get_content() for node in nodes_with_scores])
        return self.format_template.format(text=retrieved_text)

    async def _aput(self, messages: List[ChatMessage]) -> None:
        """Store messages in the vector store for future retrieval."""
        if not messages:
            return

        # Format messages with role, text content, and additional info
        texts = []
        session_id = None
        for message in messages:
            text = self._get_text_from_messages([message])
            if not text:
                continue

            # special case for session_id
            if "session_id" in message.additional_kwargs:
                session_id = message.additional_kwargs.pop("session_id")

            if message.additional_kwargs:
                text += f"\nAdditional Info: ({message.additional_kwargs!s})"

            text = f"<message role='{message.role.value}'>{text}</message>"
            texts.append(text)

        if not texts:
            return

        # Get embeddings
        text_node = TextNode(text="\n".join(texts), metadata={"session_id": session_id})
        text_node.embedding = await self.embed_model.aget_text_embedding(text_node.text)

        # Add to vector store, one node per entire message batch
        await self.vector_store.async_add([text_node])
