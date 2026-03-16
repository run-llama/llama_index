"""Node parser interface."""

import asyncio
import logging
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

logger = logging.getLogger(__name__)

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

    max_retries: int = Field(
        default=0,
        description=(
            "Maximum number of retry attempts when aextract() raises an exception. "
            "0 means no retries (fail immediately, preserving current behaviour)."
        ),
    )

    retry_backoff: float = Field(
        default=1.0,
        description=(
            "Base delay in seconds for exponential backoff between retries. "
            "Actual delay is retry_backoff * 2^attempt."
        ),
    )

    raise_on_error: bool = Field(
        default=True,
        description=(
            "Whether to raise exceptions when extraction fails after all retries. "
            "If True, the exception propagates (current behaviour). "
            "If False, logs a warning and returns empty metadata dicts."
        ),
    )

    @classmethod
    def from_dict(cls, data: Dict[str, Any], **kwargs: Any) -> Self:  # type: ignore
        if isinstance(kwargs, dict):
            data.update(kwargs)

        data.pop("class_name", None)

        llm_predictor = data.get("llm_predictor")
        if llm_predictor:
            from llama_index.core.llm_predictor.loading import load_predictor

            llm_predictor = load_predictor(llm_predictor)
            data["llm_predictor"] = llm_predictor

        llm = data.get("llm")
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
        """
        Extracts metadata for a sequence of nodes, returning a list of
        metadata dictionaries corresponding to each node.

        Args:
            nodes (Sequence[Document]): nodes to extract metadata from

        """

    def extract(self, nodes: Sequence[BaseNode]) -> List[Dict]:
        """
        Extracts metadata for a sequence of nodes, returning a list of
        metadata dictionaries corresponding to each node.

        Args:
            nodes (Sequence[Document]): nodes to extract metadata from

        """
        return asyncio_run(self.aextract(nodes))

    async def _aextract_with_retry(self, nodes: Sequence[BaseNode]) -> List[Dict]:
        """Call aextract() with optional retry and error-policy logic."""
        last_exception: Optional[Exception] = None
        for attempt in range(1 + self.max_retries):
            try:
                return await self.aextract(nodes)
            except Exception as e:
                last_exception = e
                if attempt < self.max_retries:
                    delay = self.retry_backoff * (2**attempt)
                    logger.warning(
                        "Extraction attempt %d/%d failed (%s), retrying in %.1fs ...",
                        attempt + 1,
                        self.max_retries + 1,
                        e,
                        delay,
                    )
                    await asyncio.sleep(delay)

        # All retries exhausted
        if not self.raise_on_error:
            logger.warning(
                "Extraction failed after %d attempt(s) (%s). "
                "Returning empty metadata for %d node(s).",
                self.max_retries + 1,
                last_exception,
                len(nodes),
            )
            return [{} for _ in nodes]

        raise last_exception  # type: ignore[misc]

    async def aprocess_nodes(
        self,
        nodes: Sequence[BaseNode],
        excluded_embed_metadata_keys: Optional[List[str]] = None,
        excluded_llm_metadata_keys: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[BaseNode]:
        """
        Post process nodes parsed from documents.

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

        cur_metadata_list = await self._aextract_with_retry(new_nodes)
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
        """
        Post process nodes parsed from documents.

        Allows extractors to be chained.

        Args:
            nodes (List[BaseNode]): nodes to post-process

        """
        return self.process_nodes(nodes, **kwargs)

    async def acall(self, nodes: Sequence[BaseNode], **kwargs: Any) -> List[BaseNode]:
        """
        Post process nodes parsed from documents.

        Allows extractors to be chained.

        Args:
            nodes (List[BaseNode]): nodes to post-process

        """
        return await self.aprocess_nodes(nodes, **kwargs)
