"""Node parser interface."""

from abc import abstractmethod
from copy import deepcopy
from typing import Any, Dict, List, Optional, Sequence, cast

from llama_index.core.async_utils import asyncio_run
from llama_index.core.bridge.pydantic import Field
from llama_index.core.schema import (
    BaseNode,
    MetadataMode,
    TextNode,
    TransformComponent,
)
from typing_extensions import Self

DEFAULT_NODE_TEXT_TEMPLATE = """\
[Excerpt from document]\n{metadata_str}\n\
Excerpt:\n-----\n{content}\n-----\n"""


class BaseExtractor(TransformComponent):
    """Metadata extractor."""

    is_text_node_only: bool = True

    show_progress: bool = Field(default=True, description="Whether to show progress.")

    metadata_mode: MetadataMode = Field(
        default=MetadataMode.ALL, description="Metadata mode to use when reading nodes."
    )

    node_text_template: str = Field(
        default=DEFAULT_NODE_TEXT_TEMPLATE,
        description="Template to represent how node text is mixed with metadata text.",
    )
    disable_template_rewrite: bool = Field(
        default=False, description="Disable the node template rewrite."
    )

    in_place: bool = Field(
        default=True, description="Whether to process nodes in place."
    )

    num_workers: int = Field(
        default=4,
        description="Number of workers to use for concurrent async processing.",
    )

    @classmethod
    def from_dict(cls, data: Dict[str, Any], **kwargs: Any) -> Self:  # type: ignore
        if isinstance(kwargs, dict):
            data.update(kwargs)

        data.pop("class_name", None)

        llm_predictor = data.get("llm_predictor", None)
        if llm_predictor:
            from llama_index.core.llm_predictor.loading import load_predictor

            llm_predictor = load_predictor(llm_predictor)
            data["llm_predictor"] = llm_predictor

        llm = data.get("llm", None)
        if llm:
            from llama_index.core.llms.loading import load_llm

            llm = load_llm(llm)
            data["llm"] = llm

        return cls(**data)

    @classmethod
    def class_name(cls) -> str:
        """Get class name."""
        return "MetadataExtractor"

    @abstractmethod
    async def aextract(self, nodes: Sequence[BaseNode]) -> List[Dict]:
        """Extracts metadata for a sequence of nodes, returning a list of
        metadata dictionaries corresponding to each node.

        Args:
            nodes (Sequence[Document]): nodes to extract metadata from

        """

    def extract(self, nodes: Sequence[BaseNode]) -> List[Dict]:
        """Extracts metadata for a sequence of nodes, returning a list of
        metadata dictionaries corresponding to each node.

        Args:
            nodes (Sequence[Document]): nodes to extract metadata from

        """
        return asyncio_run(self.aextract(nodes))

    async def aprocess_nodes(
        self,
        nodes: Sequence[BaseNode],
        excluded_embed_metadata_keys: Optional[List[str]] = None,
        excluded_llm_metadata_keys: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[BaseNode]:
        """Post process nodes parsed from documents.

        Allows extractors to be chained.

        Args:
            nodes (List[BaseNode]): nodes to post-process
            excluded_embed_metadata_keys (Optional[List[str]]):
                keys to exclude from embed metadata
            excluded_llm_metadata_keys (Optional[List[str]]):
                keys to exclude from llm metadata
        """
        if self.in_place:
            new_nodes = nodes
        else:
            new_nodes = [deepcopy(node) for node in nodes]

        cur_metadata_list = await self.aextract(new_nodes)
        for idx, node in enumerate(new_nodes):
            node.metadata.update(cur_metadata_list[idx])

        for idx, node in enumerate(new_nodes):
            if excluded_embed_metadata_keys is not None:
                node.excluded_embed_metadata_keys.extend(excluded_embed_metadata_keys)
            if excluded_llm_metadata_keys is not None:
                node.excluded_llm_metadata_keys.extend(excluded_llm_metadata_keys)
            if not self.disable_template_rewrite:
                if isinstance(node, TextNode):
                    cast(TextNode, node).text_template = self.node_text_template

        return new_nodes  # type: ignore

    def process_nodes(
        self,
        nodes: Sequence[BaseNode],
        excluded_embed_metadata_keys: Optional[List[str]] = None,
        excluded_llm_metadata_keys: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[BaseNode]:
        return asyncio_run(
            self.aprocess_nodes(
                nodes,
                excluded_embed_metadata_keys=excluded_embed_metadata_keys,
                excluded_llm_metadata_keys=excluded_llm_metadata_keys,
                **kwargs,
            )
        )

    def __call__(self, nodes: Sequence[BaseNode], **kwargs: Any) -> List[BaseNode]:
        """Post process nodes parsed from documents.

        Allows extractors to be chained.

        Args:
            nodes (List[BaseNode]): nodes to post-process
        """
        return self.process_nodes(nodes, **kwargs)

    async def acall(self, nodes: Sequence[BaseNode], **kwargs: Any) -> List[BaseNode]:
        """Post process nodes parsed from documents.

        Allows extractors to be chained.

        Args:
            nodes (List[BaseNode]): nodes to post-process
        """
        return await self.aprocess_nodes(nodes, **kwargs)
