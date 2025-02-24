"""Optimization related classes and functions."""

import logging
from typing import Any, Dict, List, Optional, Literal

from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

logger = logging.getLogger(__name__)


DEFAULT_INSTRUCTION_STR = "Given the context, please answer the final question"


def format_metadata(nodes=List[NodeWithScore]):
    return {node.node.id_: node.metadata for node in nodes}


class LongLLMLinguaPostprocessor(BaseNodePostprocessor):
    """Optimization of nodes.

    Compress using LongLLMLingua paper.

    """

    instruction_str: str = Field(
        default=DEFAULT_INSTRUCTION_STR, description="Instruction string."
    )
    target_token: int = Field(
        default=-1, description="Target number of compressed tokens."
    )
    use_llmlingua2: bool = Field(
        default=False, description="Whether to use the llmlingua2 approach"
    )
    rank_method: str = Field(default="longllmlingua", description="Ranking method.")
    additional_compress_kwargs: Dict[str, Any] = Field(
        default_factory=dict, description="Additional compress kwargs."
    )

    _llm_lingua: Any = PrivateAttr()

    def __init__(
        self,
        model_name: str = "NousResearch/Llama-2-7b-hf",
        device_map: Literal["cuda", "cpu", "mps"] = "cuda",
        model_config: Optional[dict] = {},
        open_api_config: Optional[dict] = {},
        instruction_str: str = DEFAULT_INSTRUCTION_STR,
        target_token: float = -1,
        rank_method: str = "longllmlingua",
        additional_compress_kwargs: Optional[Dict[str, Any]] = {},
        use_llmlingua2: bool = False,
    ):
        """LongLLMLingua Compressor for Node Context."""
        from llmlingua import PromptCompressor

        super().__init__(
            instruction_str=instruction_str,
            target_token=target_token,
            rank_method=rank_method,
            additional_compress_kwargs=additional_compress_kwargs,
            use_llmlingua2=use_llmlingua2,
        )

        open_api_config = open_api_config or {}
        additional_compress_kwargs = additional_compress_kwargs or {}

        if self.use_llmlingua2 is True:
            assert (
                model_name == "microsoft/llmlingua-2-xlm-roberta-large-meetingbank"
            ), 'Must use "microsoft/llmlingua-2-xlm-roberta-large-meetingbank" as the model name for llmlingua2'

        self._llm_lingua = PromptCompressor(
            model_name=model_name,
            device_map=device_map,
            model_config=model_config,
            open_api_config=open_api_config,
            use_llmlingua2=self.use_llmlingua2,
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

        # The prompt compression for llmlingua2 works on raw texts, that's why it's better to just extract metadata texts.
        context_texts = [n.text for n in nodes]

        # Preserve metadata for prompt compressed nodes
        metadata = format_metadata(nodes)
        new_context_texts = "".join(context_texts)

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

        if self.use_llmlingua2 is False:
            compressed_prompt_txt_list = compressed_prompt_txt_list[1:-1]

        # return nodes for each list
        keys_to_exclude = list(metadata.keys())
        return [
            NodeWithScore(
                node=TextNode(
                    text=t,
                    metadata=metadata,
                    excluded_llm_metadata_keys=keys_to_exclude,
                    excluded_embed_metadata_keys=keys_to_exclude,
                )
            )
            for t in compressed_prompt_txt_list
        ]
