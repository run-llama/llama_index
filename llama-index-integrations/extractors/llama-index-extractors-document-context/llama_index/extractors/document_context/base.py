from llama_index.core.llms import ChatMessage, LLM
from llama_index.core.async_utils import DEFAULT_NUM_WORKERS, run_jobs
from llama_index.core.extractors import BaseExtractor
from llama_index.core.schema import BaseNode, TextNode, Node
from llama_index.core import Settings
from llama_index.core.storage.docstore.simple_docstore import DocumentStore
from typing import Optional, Dict, List, Set, Literal, Any, Sequence, Coroutine, Union
from llama_index.core.llms import TextBlock, ImageBlock, ChatResponse
import logging
import asyncio
import random
from functools import lru_cache
import tiktoken
from typing import TypeGuard

def is_text_node(node: BaseNode) -> TypeGuard[Union[Node, TextNode]]:
    return isinstance(node, (Node, TextNode))

OversizeStrategy = Literal["truncate_first", "truncate_last", "warn", "error", "ignore"]

# original context prompt from the Anthropic cookbook/blogpost, works well
ORIGINAL_CONTEXT_PROMPT: str = """
Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk.
Answer only with the succinct context and nothing else.
"""

# miniaturized context prompt, generates better results, produces more keyword-laden results for better matches
SUCCINCT_CONTEXT_PROMPT: str = """
Generate keywords and brief phrases describing the main topics, entities, and actions in this text. Replace pronouns with their specific referents. Disambiguate pronouns and ambiguous terms in the chunk. Format as comma-separated phrases. Exclude meta-commentary about the text itself.
"""

DEFAULT_CONTEXT_PROMPT:str = SUCCINCT_CONTEXT_PROMPT

DEFAULT_KEY: str = "context"

class DocumentContextExtractor(BaseExtractor):
    """
    An LLM-based context extractor for enhancing RAG accuracy through document analysis.
    
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
        max_contextual_tokens (int): Maximum tokens in generated context
        oversized_document_strategy (OversizeStrategy): Strategy for handling large documents
        warn_on_oversize (bool): Whether to log warnings for oversized documents
        tiktoken_encoder (str): Name of the tiktoken encoding to use
        
    Example:
        ```python
        extractor = DocumentContextExtractor(
            docstore=my_docstore,
            llm=my_llm,
            max_context_length=64000,
            max_contextual_tokens=256
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
    max_contextual_tokens: int
    oversized_document_strategy: OversizeStrategy
    warn_on_oversize: bool = True
    tiktoken_encoder: str

    def __init__(
        self,
        docstore: DocumentStore,
        llm: LLM,
        max_context_length: int,
        key: str = DEFAULT_KEY,
        prompt: str = DEFAULT_CONTEXT_PROMPT,
        num_workers: int = DEFAULT_NUM_WORKERS,
        max_contextual_tokens: int = 512,
        oversized_document_strategy: OversizeStrategy = "truncate_first",
        warn_on_oversize: bool = True,
        tiktoken_encoder: str = "cl100k_base",
        **kwargs
    ) -> None:
        super().__init__(num_workers=num_workers, **kwargs)

        self.llm = llm or Settings.llm
        self.docstore = docstore
        self.key = key
        self.prompt = prompt
        self.doc_ids = set()
        self.max_context_length = max_context_length
        self.max_contextual_tokens = max_contextual_tokens
        self.oversized_document_strategy = oversized_document_strategy
        self.warn_on_oversize = warn_on_oversize
        self.tiktoken_encoder = tiktoken_encoder

    # this can take a surprisingly long time on longer docs so we cache it. For oversized docs, we end up counting twice, the 2nd time withotu the cache.
    # but if you're repeateddly running way oversize docs, the time that takes wont be what matters anyways.
    @staticmethod
    @lru_cache(maxsize=1000)
    def _count_tokens(text: str, encoder:str="cl100k_base") -> int:
        """
        This can take a surprisingly long time on longer docs so we cache it, and we need to call it on every doc, regardless of size.
        """
        encoding = tiktoken.get_encoding(encoder)
        return len(encoding.encode(text))

    @staticmethod
    @lru_cache(maxsize=10)
    def _truncate_text(text: str, max_token_count: int, how: Literal['first', 'last'] = 'first', encoder="cl100k_base") -> str:
        """
        This can take a couple seconds. A small cache is nice here because the async calls will mostly happen in-order. If you DO hit an oversized document,
        you would otherwise be re-truncating 1000s of times as you procses through each chunk in your 200+ document.
        """
        encoding = tiktoken.get_encoding(encoder)
        tokens = encoding.encode(text)
        
        if how == 'first':
            truncated_tokens = tokens[:max_token_count]
        else:  # 'last'
            truncated_tokens = tokens[-max_token_count:]
            
        return encoding.decode(truncated_tokens)

    async def _agenerate_node_context(
    self,
    node: Union[Node, TextNode],
    metadata: Dict,
    document: Union[Node, TextNode],
    prompt: str,
    key: str
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
                    {
                        "text": cached_text,
                        "block_type": "text",
                        "cache_control": {"type": "ephemeral"},
                    },
                    {
                        "text": f"Here is the chunk we want to situate within the whole document:\n<chunk>{node.get_content()}</chunk>\n{prompt}",
                        "block_type": "text",
                    },
                ],
            ),
        ]

        max_retries = 5
        base_delay = 60

        for attempt in range(max_retries):
            try:
                response:ChatResponse = await self.llm.achat(
                    messages,
                    max_tokens=self.max_contextual_tokens,
                    extra_headers={"anthropic-beta": "prompt-caching-2024-07-31"},
                )
                first_block:Union[TextBlock,ImageBlock] = response.message.blocks[0]
                if isinstance(first_block, TextBlock):
                    metadata[key] = first_block.text
                else:
                    logging.warning(f"Received non-text block type: {type(first_block)}")
                return metadata

            except Exception as e:
                is_rate_limit = any(
                    message in str(e).lower()
                    for message in ["rate limit", "too many requests", "429"]
                )

                if is_rate_limit and attempt < max_retries - 1:
                    delay = (base_delay * (2 ** attempt)) + (random.random() * 0.5)
                    logging.warning(
                        f"Rate limit hit, retrying in {delay:.1f} seconds "
                        f"(attempt {attempt + 1}/{max_retries})"
                    )
                    await asyncio.sleep(delay)
                    continue
                
                if is_rate_limit:
                    logging.error(f"Failed after {max_retries} retries due to rate limiting")
                else:
                    logging.warning(f"Error generating context for node {node.node_id}: {str(e)}")
                
                return metadata
    
        return metadata
    
    async def _get_document(self, doc_id: str) -> Optional[Union[Node, TextNode]]:
        """counting tokens can be slow, as can awaiting the docstore (potentially), so we keep a small lru_cache"""

        # first we need to get the document
        doc = await self.docstore.aget_document(doc_id)
        if not doc:
            logging.warning(f"Document {doc_id} not found in docstore")
            return None
        if not is_text_node(doc):
            logging.warning(f"Document {doc_id} is an instance of BaseNode 'text' attribute (TextNode, Document)")
            return None
        
        # then truncate if necessary. 
        if self.max_context_length is not None:
            strategy = self.oversized_document_strategy
            token_count = self._count_tokens(doc.get_content(), self.tiktoken_encoder)
            if token_count > self.max_context_length:
                message = (
                    f"Document {doc.node_id} is too large ({token_count} tokens) "
                    f"to be processed. Doc metadata: {doc.metadata}"
                )
                
                if self.warn_on_oversize:
                    logging.warning(message)

                if strategy == "truncate_first":
                    text = self._truncate_text(doc.get_content(), self.max_context_length, 'first', self.tiktoken_encoder)
                    doc.set_content(text)
                elif strategy == "truncate_last":
                    text = self._truncate_text(doc.get_content(), self.max_context_length, 'last', self.tiktoken_encoder)    
                    doc.set_content(text)                
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
        metadata_list:List[Dict] = []
        for _ in nodes:
            metadata_list.append({})
        metadata_map = {node.node_id: metadata_dict for metadata_dict, node in zip(metadata_list, nodes)}        

        # iterate over all the nodes and generate the jobs
        node_tasks:List[Coroutine[Any, Any, Any]] = []
        for node in nodes:
            if not node.source_node:
                continue
            if not is_text_node(node):
                continue

            doc:Optional[Union[Node,TextNode]] = await self._get_document(node.source_node.node_id)
            if not doc:
                continue

            metadata = metadata_map[node.node_id]
            task = self._agenerate_node_context(node, metadata, doc, self.prompt, self.key)
            node_tasks.append(task)
        
        # then run the jobs
        await run_jobs(
            node_tasks,
            show_progress=self.show_progress,
            workers=self.num_workers,
        )
        
        return metadata_list