import asyncio
import logging
import random
from functools import lru_cache
from typing import (
    Any,
    ClassVar,
    Coroutine,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Set,
    Union,
)
from typing_extensions import TypeGuard

from llama_index.core import Settings
from llama_index.core.async_utils import DEFAULT_NUM_WORKERS, run_jobs
from llama_index.core.extractors import BaseExtractor
from llama_index.core.llms import (
    AudioBlock,
    ChatMessage,
    ChatResponse,
    ImageBlock,
    LLM,
    TextBlock,
)
from llama_index.core.schema import BaseNode, Node, TextNode
from llama_index.core.storage.docstore.simple_docstore import DocumentStore


def is_text_node(node: BaseNode) -> TypeGuard[Union[Node, TextNode]]:
    return isinstance(node, (Node, TextNode))


OversizeStrategy = Literal["warn", "error", "ignore"]


# original context prompt from the Anthropic cookbook/blogpost, works well
ORIGINAL_CONTEXT_PROMPT: str = """
Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk.
Answer only with the succinct context and nothing else.
"""

# miniaturized context prompt, generates better results, produces more keyword-laden results for better matches
SUCCINCT_CONTEXT_PROMPT: str = """
Generate keywords and brief phrases describing the main topics, entities, and actions in this text. Replace pronouns with their specific referents. Disambiguate pronouns and ambiguous terms in the chunk. Format as comma-separated phrases. Exclude meta-commentary about the text itself.
"""


class DocumentContextExtractor(BaseExtractor):
    """
    An LLM-based context extractor for enhancing RAG accuracy through document analysis.

    ! Nodes that already have the 'key' in node.metadata will NOT be processed - will be skipped !

    This extractor processes documents and their nodes to generate contextual metadata,
    implementing the approach described in the Anthropic "Contextual Retrieval" blog post.
    It handles rate limits, document size constraints, and parallel processing of nodes.

    Attributes:
        llm (LLM): Language model instance for generating context
        docstore (DocumentStore): Storage for parent documents
        key (str): Metadata key for storing extracted context
        prompt (str): Prompt template for context generation
        doc_ids (Set[str]): Set of processed document IDs
        max_context_length (int): Maximum allowed document context length
        max_output_tokens (int): Maximum tokens in generated context
        oversized_document_strategy (OversizeStrategy): Strategy for handling large documents

    Example:
        ```python
        extractor = DocumentContextExtractor(
            docstore=my_docstore,
            llm=my_llm,
            max_context_length=64000,
            max_output_tokens=256
        )
        metadata_list = await extractor.aextract(nodes)
        ```
    """

    # Pydantic fields
    llm: LLM
    docstore: DocumentStore
    key: str
    prompt: str
    doc_ids: Set[str]
    max_context_length: int
    max_output_tokens: int
    oversized_document_strategy: OversizeStrategy
    num_workers: int = DEFAULT_NUM_WORKERS

    ORIGINAL_CONTEXT_PROMPT: ClassVar[str] = ORIGINAL_CONTEXT_PROMPT
    SUCCINCT_CONTEXT_PROMPT: ClassVar[str] = SUCCINCT_CONTEXT_PROMPT

    DEFAULT_KEY: str = "context"

    def __init__(
        self,
        docstore: DocumentStore,
        llm: Optional[LLM] = None,
        max_context_length: int = 1000,
        key: str = DEFAULT_KEY,
        prompt: str = ORIGINAL_CONTEXT_PROMPT,
        num_workers: int = DEFAULT_NUM_WORKERS,
        max_output_tokens: int = 512,
        oversized_document_strategy: OversizeStrategy = "warn",
        **kwargs: Any,
    ) -> None:
        """Init params."""
        assert hasattr(
            llm, "achat"
        )  # not all LLMs have this, particularly the huggingfaceapi ones.

        super().__init__(
            llm=llm or Settings.llm,
            docstore=docstore,
            key=key,
            prompt=prompt,
            doc_ids=set(),
            max_context_length=max_context_length,
            max_output_tokens=max_output_tokens,
            oversized_document_strategy=oversized_document_strategy,
            num_workers=num_workers,
            **kwargs,
        )

    # this can take a surprisingly long time on longer docs so we cache it. For oversized docs, we end up counting twice, the 2nd time without the cache.
    # but if you're repeateddly running way oversize docs, the time that takes won't be what matters anyways.
    @staticmethod
    @lru_cache(maxsize=1000)
    def _count_tokens(text: str) -> int:
        """
        This can take a surprisingly long time on longer docs so we cache it, and we need to call it on every doc, regardless of size.
        """
        encoder = Settings.tokenizer
        tokens = encoder(text)
        return len(tokens)

    async def _agenerate_node_context(
        self,
        node: Union[Node, TextNode],
        metadata: Dict,
        document: Union[Node, TextNode],
        prompt: str,
        key: str,
    ) -> Dict:
        """
        Generate context for a node using LLM with retry logic.

        Implements exponential backoff for rate limit handling and uses prompt
        caching when available. The function retries on rate limits.

        Args:
            node: Node to generate context for
            metadata: Metadata dictionary to update
            document: Parent document containing the node
            prompt: Prompt template for context generation
            key: Metadata key for storing generated context

        Returns:
            Updated metadata dictionary with generated context

        Note:
            Uses exponential backoff starting at 60 seconds with up to 5 retries
            for rate limit handling.
        """
        cached_text = f"<document>{document.get_content()}</document>"
        messages = [
            ChatMessage(
                role="user",
                content=[
                    TextBlock(
                        text=cached_text,
                        type="text",
                    )
                ],
                additional_kwargs={"cache_control": {"type": "ephemeral"}},
            ),
            ChatMessage(
                role="user",
                content=[
                    TextBlock(
                        text=f"Here is the chunk we want to situate within the whole document:\n<chunk>{node.get_content()}</chunk>\n{prompt}",
                        type="text",
                    )
                ],
            ),
        ]

        max_retries = 5
        base_delay = 60

        for attempt in range(max_retries):
            try:
                # Extra headers typically dont cause issues
                headers = {"anthropic-beta": "prompt-caching-2024-07-31"}

                response: ChatResponse = await self.llm.achat(
                    messages, max_tokens=self.max_output_tokens, extra_headers=headers
                )

                first_block: Union[
                    TextBlock, ImageBlock, AudioBlock
                ] = response.message.blocks[0]
                if isinstance(first_block, TextBlock):
                    metadata[key] = first_block.text
                else:
                    logging.warning(
                        f"Received non-text block type: {type(first_block)}"
                    )
                return metadata

            except Exception as e:
                is_rate_limit = any(
                    message in str(e).lower()
                    for message in ["rate limit", "too many requests", "429"]
                )

                if is_rate_limit and attempt < max_retries - 1:
                    delay = (base_delay * (2**attempt)) + (random.random() * 0.5)
                    logging.warning(
                        f"Rate limit hit, retrying in {delay:.1f} seconds "
                        f"(attempt {attempt + 1}/{max_retries})"
                    )
                    await asyncio.sleep(delay)
                    continue

                if is_rate_limit:
                    logging.error(
                        f"Failed after {max_retries} retries due to rate limiting"
                    )
                else:
                    logging.warning(
                        f"Error generating context for node {node.node_id}: {e}",
                        exc_info=True,
                    )
                return metadata

        return metadata

    async def _get_document(self, doc_id: str) -> Optional[Union[Node, TextNode]]:
        """Counting tokens can be slow, as can awaiting the docstore (potentially), so we keep a small lru_cache."""
        # first we need to get the document
        try:
            doc = await self.docstore.aget_document(doc_id)
        except ValueError as e:
            if "not found" in str(e):
                logging.warning(f"Document {doc_id} not found in docstore")
                return None
        if not doc:
            logging.warning(f"Document {doc_id} not found in docstore")
            return None
        if not is_text_node(doc):
            logging.warning(f"Document {doc_id} is not an instance of (TextNode, Node)")
            return None

        # then truncate if necessary.
        if self.max_context_length is not None:
            strategy = self.oversized_document_strategy
            token_count = self._count_tokens(doc.get_content())
            if token_count > self.max_context_length:
                message = (
                    f"Document {doc.node_id} is too large ({token_count} tokens) "
                    f"to be processed. Doc metadata: {doc.metadata}"
                )

                if strategy == "warn":
                    logging.warning(message)
                elif strategy == "error":
                    raise ValueError(message)
                elif strategy == "ignore":
                    pass
                else:
                    raise ValueError(f"Unknown oversized document strategy: {strategy}")

        return doc

    async def aextract(self, nodes: Sequence[BaseNode]) -> List[Dict]:
        """
        Extract context for multiple nodes asynchronously, optimized for loosely ordered nodes.
        Processes each node independently without guaranteeing sequential document handling.
        Nodes will be *mostly* processed in document-order assuming nodes get passed in document-order.

        Args:
            nodes: List of nodes to process, ideally grouped by source document

        Returns:
            List of metadata dictionaries with generated context
        """
        metadata_list: List[Dict] = []
        for _ in nodes:
            metadata_list.append({})
        metadata_map = {
            node.node_id: metadata_dict
            for metadata_dict, node in zip(metadata_list, nodes)
        }

        # sorting takes a tiny amount of time - 0.4s for 1_000_000 nodes. but 1_000_000 nodes takes potentially hours to process
        # considering sorting CAN save the users hundreds of dollars in API costs, we just sort and leave no option to do otherwise.
        # The math always works out in the user's favor and we can't guarantee things are sorted in the first place.
        sorted_nodes = sorted(
            nodes, key=lambda n: n.source_node.node_id if n.source_node else ""
        )

        # iterate over all the nodes and generate the jobs
        node_tasks: List[Coroutine[Any, Any, Any]] = []
        for node in sorted_nodes:
            if not (node.source_node and is_text_node(node)):
                continue

            # Skip already processed nodes
            if self.key in node.metadata:
                continue

            doc: Optional[Union[Node, TextNode]] = await self._get_document(
                node.source_node.node_id
            )
            if not doc:
                continue

            metadata = metadata_map[node.node_id]
            # this modifies metadata in-place, adding a new key to the dictionary - we needed do anytyhing with the return value
            task = self._agenerate_node_context(
                node, metadata, doc, self.prompt, self.key
            )
            node_tasks.append(task)

        # then run the jobs - this does return the metadata list, but we already have it
        await run_jobs(
            node_tasks,
            show_progress=self.show_progress,
            workers=self.num_workers,
        )

        return metadata_list


if __name__ == "__main__":
    print(DocumentContextExtractor.ORIGINAL_CONTEXT_PROMPT)
