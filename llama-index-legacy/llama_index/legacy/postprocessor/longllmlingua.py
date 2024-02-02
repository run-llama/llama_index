"""Optimization related classes and functions."""

import logging
from typing import Any, Dict, List, Optional

from llama_index.legacy.bridge.pydantic import Field, PrivateAttr
from llama_index.legacy.postprocessor.types import BaseNodePostprocessor
from llama_index.legacy.schema import MetadataMode, NodeWithScore, QueryBundle, TextNode

logger = logging.getLogger(__name__)


DEFAULT_INSTRUCTION_STR = "Given the context, please answer the final question"


class LongLLMLinguaPostprocessor(BaseNodePostprocessor):
    """Optimization of nodes.

    Compress using LongLLMLingua paper.

    """

    metadata_mode: MetadataMode = Field(
        default=MetadataMode.ALL, description="Metadata mode."
    )
    instruction_str: str = Field(
        default=DEFAULT_INSTRUCTION_STR, description="Instruction string."
    )
    target_token: int = Field(
        default=300, description="Target number of compressed tokens."
    )
    rank_method: str = Field(default="longllmlingua", description="Ranking method.")
    additional_compress_kwargs: Dict[str, Any] = Field(
        default_factory=dict, description="Additional compress kwargs."
    )

    _llm_lingua: Any = PrivateAttr()

    def __init__(
        self,
        model_name: str = "NousResearch/Llama-2-7b-hf",
        device_map: str = "cuda",
        model_config: Optional[dict] = {},
        open_api_config: Optional[dict] = {},
        metadata_mode: MetadataMode = MetadataMode.ALL,
        instruction_str: str = DEFAULT_INSTRUCTION_STR,
        target_token: int = 300,
        rank_method: str = "longllmlingua",
        additional_compress_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """LongLLMLingua Compressor for Node Context."""
        from llmlingua import PromptCompressor

        open_api_config = open_api_config or {}
        additional_compress_kwargs = additional_compress_kwargs or {}

        self._llm_lingua = PromptCompressor(
            model_name=model_name,
            device_map=device_map,
            model_config=model_config,
            open_api_config=open_api_config,
        )
        super().__init__(
            metadata_mode=metadata_mode,
            instruction_str=instruction_str,
            target_token=target_token,
            rank_method=rank_method,
            additional_compress_kwargs=additional_compress_kwargs,
        )

    @classmethod
    def class_name(cls) -> str:
        return "LongLLMLinguaPostprocessor"

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        """Optimize a node text given the query by shortening the node text."""
        if query_bundle is None:
            raise ValueError("Query bundle is required.")
        context_texts = [n.get_content(metadata_mode=self.metadata_mode) for n in nodes]
        # split by "\n\n" (recommended by LongLLMLingua authors)
        new_context_texts = [
            c for context in context_texts for c in context.split("\n\n")
        ]

        # You can use it this way, although the question-aware fine-grained compression hasn't been enabled.
        compressed_prompt = self._llm_lingua.compress_prompt(
            new_context_texts,  # ! Replace the previous context_list
            instruction=self.instruction_str,
            question=query_bundle.query_str,
            # target_token=2000,
            target_token=self.target_token,
            rank_method=self.rank_method,
            **self.additional_compress_kwargs,
        )

        compressed_prompt_txt = compressed_prompt["compressed_prompt"]

        # separate out the question and instruction (appended to top and bottom)
        compressed_prompt_txt_list = compressed_prompt_txt.split("\n\n")
        compressed_prompt_txt_list = compressed_prompt_txt_list[1:-1]

        # return nodes for each list
        return [
            NodeWithScore(node=TextNode(text=t)) for t in compressed_prompt_txt_list
        ]
