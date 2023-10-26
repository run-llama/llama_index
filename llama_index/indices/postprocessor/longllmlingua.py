"""Optimization related classes and functions."""
import logging
from typing import Callable, List, Optional, Any, Dict

from llama_index.bridge.pydantic import Field, PrivateAttr
from llama_index.embeddings.base import BaseEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.indices.postprocessor.types import BaseNodePostprocessor
from llama_index.indices.query.embedding_utils import get_top_k_embeddings
from llama_index.indices.query.schema import QueryBundle
from llama_index.schema import MetadataMode, NodeWithScore, TextNode

logger = logging.getLogger(__name__)


DEFAULT_INSTRUCTION_STR = "Given the context, please answer the final question"


class LongLLMLinguaPostprocessor(BaseNodePostprocessor):
    """Optimization of nodes.

    Compress using LongLLMLingua paper.
    
    """

    def __init__(
        self,
        model_name: str = "NousResearch/Llama-2-7b-hf",
        device_map: str = "cuda",
        use_auth_token: bool = False,
        open_api_config: Optional[dict] = None,
        metadata_mode: MetadataMode = MetadataMode.ALL,
        instruction_str: Optional[str] = None,
        target_token: int = 300,
        rank_method: str = "longllmlingua",
        additional_compress_kwargs: Dict[str, Any] = None,
    ):
        """LongLLMLingua Compressor for Node Context."""
        from llmlingua import PromptCompressor
        
        open_api_config = open_api_config or {}
        self._llm_lingua = PromptCompressor(
            model_name=model_name,
            device_map=device_map,
            use_auth_token=use_auth_token,
            open_api_config=open_api_config,
        )
        self._metadata_mode = metadata_mode
        self._instruction_str = instruction_str or DEFAULT_INSTRUCTION_STR
        self._target_token = target_token
        self._rank_method = rank_method
        self._additional_compress_kwargs = additional_compress_kwargs or {}

    @classmethod
    def class_name(cls) -> str:
        return "LongLLMLinguaPostprocessor"

    def postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        """Optimize a node text given the query by shortening the node text."""

        context_texts = [n.get_content(metadata_mode=self._metadata_mode) for n in nodes]
        # split by "\n\n" (recommended by LongLLMLingua authors)
        new_context_texts = [c for context in context_texts for c in context.split("\n\n")]

        # You can use it this way, although the question-aware fine-grained compression hasn't been enabled.
        compressed_prompt = self._llm_lingua.compress_prompt(
            new_context_texts, # ! Replace the previous context_list
            instruction=self._instruction_str,
            question=query_bundle.query_str,
            # target_token=2000,
            target_token=self._target_token,
            rank_method=self._rank_method,
            **self._additional_compress_kwargs,
        )

        compressed_prompt_txt = compressed_prompt["compressed_prompt"]

        # separate out the question and instruction (appended to top and bottom)
        compressed_prompt_txt_list = compressed_prompt_txt.split("\n\n")
        compressed_prompt_txt_list = compressed_prompt_txt_list[1:-1]

        # return nodes for each list
        new_nodes = [
            NodeWithScore(node=TextNode(text=t)) for t in compressed_prompt_txt_list
        ]

        return new_nodes
