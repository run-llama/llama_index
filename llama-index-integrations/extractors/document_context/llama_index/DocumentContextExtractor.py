
from llama_index.core.llms import ChatMessage
from typing import Optional, Dict, List, Tuple, Set
from llama_index.core.llms.llm import LLM
from llama_index.core.async_utils import DEFAULT_NUM_WORKERS, run_jobs
from llama_index.core.extractors import BaseExtractor
from llama_index.core.schema import Document, Node
from llama_index.core import Settings
from llama_index.core.storage.docstore.simple_docstore import DocumentStore
from textwrap import dedent
import importlib
import logging

DEFAULT_CONTEXT_PROMPT: str = dedent("""
    Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Also disambiguate pronouns and key terms in the chunk. Answer only with the succinct context and nothing else.
    """).strip()

DEFAULT_KEY: str = "context"
from llama_index.core.node_parser import TokenTextSplitter

class DocumentContextExtractor(BaseExtractor):
    keys: List[str]
    prompts: List[str]
    llm: LLM
    docstore:DocumentStore
    doc_ids: Set
    max_context_length:int
    max_contextual_tokens:int
    oversized_document_strategy:str

    @staticmethod
    def _truncate_text(text:str, max_token_count:int, how='first')-> str:
        """
        Truncate the document to the specified token coutn
        :param document: The document to get the text of.
        :param max_token_count: The maximum number of tokens to return.
        :param how: How to truncate the document. Can be 'first' or 'last'.
        :return: The text of the documen
        """

        text_splitter = TokenTextSplitter(chunk_size=max_token_count, chunk_overlap=0)
        chunks = text_splitter.split_text(text)
        if how == 'first':
            text = chunks[0]
        elif how == 'last':
            text = chunks[-1]
        else:
            raise ValueError("Invalid truncation method. Must be 'first' or 'last'.")
        
        return text if text else ""
    
    def _count_tokens(text:str)->int:
        """
        Get the number of tokens in the document.
        :param document: The document to get the number of tokens of.
        :return: The number of tokens in the document
        """
        text_splitter = TokenTextSplitter(chunk_size=1, chunk_overlap=0)
        tokens = text_splitter.split_text(text)
        token_count = len(tokens)
        return token_count
            
    def __init__(self, docstore:DocumentStore, keys=None, prompts=None, llm: LLM = None,
                 num_workers: int = DEFAULT_NUM_WORKERS, max_context_length:int = 128000,
                 max_contextual_tokens:int = 512,
                 oversized_document_strategy = "truncate_first",
                 **kwargs):
        """
        Args:
            docstore (DocumentStore): DocumentStore to extract from
            keys (List[str]): List of keys to extract context for
            prompts (List[str]): List of prompts to use for context extraction
            llm (LLM): LLM to use for context extraction
            num_workers (int): Number of workers to use for context extraction
            max_context_length (int): Maximum context length to use for context extraction
            max_contextual_tokens (int): Maximum contextual tokens to use for context extraction
            oversized_document_strategy (str): Strategy to use for documents < max_context_length:
                "truncate_first" - Truncate the document from top down
                "truncate_last" - Truncate the document from bottom up
                "warn" - Warn about the oversized document
                "error" - Raise an error for the oversized document
                "ignore" - Ignore the oversized document
            **kwargs: Additional keyword arguments based to BaseExtractor
        """

        # check if 'tiktoken' is installed and if not warn that token counts will be less  accurate
        if not importlib.util.find_spec("tiktoken"):
            raise ValueError("TikToken is required for DocumentContextExtractor. Please install tiktoken.")

        # Process defaults and values first
        keys = keys or [DEFAULT_KEY]
        prompts = prompts or [DEFAULT_CONTEXT_PROMPT]

        if isinstance(keys, str):
            keys = [keys]
        if isinstance(prompts, str):
            prompts = [prompts]
    
        llm = llm or Settings.llm
        doc_ids = set()

        # Call super().__init__ at the end with all processed values
        super().__init__(
            keys=keys,
            prompts=prompts,
            llm=llm,
            docstore=docstore,
            num_workers=num_workers,
            doc_ids=doc_ids,
            max_context_length=max_context_length,
            oversized_document_strategy=oversized_document_strategy,
            max_contextual_tokens=max_contextual_tokens,
            **kwargs
        )


    async def _agenerate_node_context(self, node, metadata, document, prompt, key)->Dict:
        """
        Generate node context using the provided LLM.

        Args:
            node (Node): The node to generate context for.
            metadata (Dict): The metadata dictionary to update.
            document (Document): The document containing the node.
            prompt (str): The prompt to use for generating context.
            key (str): The key to use for storing the generated context.

        Returns:
            Dict: The updated metadata dictionary.
        """
        
        cached_text = f"<document>{document.text}</document>"

        messages = [
            # ChatMessage(role="system", content=self.system_prompt),
            ChatMessage(
                role="user",
                content=[
                    {
                        "text": cached_text,
                        "block_type": "text",
                        "cache_control": {"type": "ephemeral"},
                    },
                    {
                        "text":  f"Here is the chunk we want to situate within the whole document:\n<chunk>{node.text}</chunk>\n{prompt}",
                        "block_type": "text",
                    },
                ],
            ),
        ]
        response = await self.llm.achat(
            messages,
            max_tokens=self.max_contextual_tokens,
            extra_headers={"anthropic-beta": "prompt-caching-2024-07-31"},
        )
        response_text = response.message.blocks[0].text

        metadata[key] = response_text
        return metadata
    
    async def aextract(self, nodes) -> List[Dict]:
        # Extract node-level summary metadata
        metadata_list: List[Dict] = [{} for _ in nodes]
        # we need to preserve the order of the nodes, but process the nodes uot-of-order
        metadata_map = {node.node_id: metadata_dict for metadata_dict, node in zip(metadata_list, nodes)}

        source_doc_ids = set([node.source_node.node_id for node in nodes])

        # make a mapping of doc id: node
        doc_id_to_nodes = {}
        for node in nodes:
            if not (node.source_node and (node.source_node.node_id in source_doc_ids)):
                continue
            parent_id = node.source_node.node_id
            
            if parent_id not in doc_id_to_nodes:
                doc_id_to_nodes[parent_id] = []
            doc_id_to_nodes[parent_id].append(node)

        i = 0
        for doc_id in source_doc_ids:
            doc = self.docstore.get_document(doc_id)

            # warn trim or raise an error per the document strategy
            if self.max_context_length is not None:
                token_count = DocumentContextExtractor._count_tokens(doc.text)
                if token_count > self.max_context_length:
                    message = f"Document {doc.id} is too large ({token_count} tokens) to be processed. Doc metadata: {doc.metadata}"
                
                    if self.oversized_document_strategy == "truncate_first":
                        doc.text = DocumentContextExtractor._truncate_text(doc.text, self.max_context_length, how='first')
                    if self.oversized_document_strategy == "truncate_last":
                        doc.text = DocumentContextExtractor._truncate_text(doc.text, self.max_context_length, how='last')

                    elif self.oversized_document_strategy == "warn":
                        logging.warning(message)
                    elif self.oversized_document_strategy == "error":
                        raise ValueError(message)
                    elif self.oversized_document_strategy == "ignore":
                        continue
                    else:
                        raise ValueError(f"Unknown oversized document strategy: {self.oversized_document_strategy}")

            node_summaries_jobs = []
            for prompt, key in list(zip(self.prompts, self.keys)):
                for node in doc_id_to_nodes.get(doc_id,[]):
                    i += 1
                    metadata_dict = metadata_map[node.node_id]
                    node_summaries_jobs.append(self._agenerate_node_context(node, metadata_dict, doc, prompt, key))

            new_metadata = await run_jobs(
                node_summaries_jobs,
                show_progress=self.show_progress,
                workers=self.num_workers,
            )
            print(f"Jobs built. requesting {len(node_summaries_jobs)} nodes with {self.num_workers} workers.")

        print(metadata_list)
        return metadata_list