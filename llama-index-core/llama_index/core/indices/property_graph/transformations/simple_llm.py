import logging
from typing import Any, Callable, Optional, Sequence, Union

from llama_index.core.async_utils import asyncio_run, run_jobs
from llama_index.core.indices.property_graph.utils import (
    default_parse_triplets_fn,
)
from llama_index.core.graph_stores.types import (
    EntityNode,
    Relation,
    KG_NODES_KEY,
    KG_RELATIONS_KEY,
)
from llama_index.core.llms.llm import LLM
from llama_index.core.prompts import PromptTemplate
from llama_index.core.prompts.default_prompts import (
    DEFAULT_KG_TRIPLET_EXTRACT_PROMPT,
)
from llama_index.core.schema import TransformComponent, BaseNode, MetadataMode

logger = logging.getLogger(__name__)


class SimpleLLMPathExtractor(TransformComponent):
    """
    Extract triples from a graph.

    Uses an LLM and a simple prompt + output parsing to extract paths (i.e. triples) from text.

    Args:
        llm (LLM):
            The language model to use.
        extract_prompt (Union[str, PromptTemplate]):
            The prompt to use for extracting triples.
        parse_fn (callable):
            A function to parse the output of the language model.
        num_workers (int):
            The number of workers to use for parallel processing.
        max_paths_per_chunk (int):
            The maximum number of paths to extract per chunk.

    """

    llm: LLM
    extract_prompt: PromptTemplate
    parse_fn: Callable
    num_workers: int
    max_paths_per_chunk: int
    raise_on_error: bool = False

    def __init__(
        self,
        llm: Optional[LLM] = None,
        extract_prompt: Optional[Union[str, PromptTemplate]] = None,
        parse_fn: Callable = default_parse_triplets_fn,
        max_paths_per_chunk: int = 10,
        num_workers: int = 4,
        raise_on_error: bool = False,
    ) -> None:
        """
        Init params.

        Args:
            llm (Optional[LLM]): The language model to use.
            extract_prompt (Optional[Union[str, PromptTemplate]]): The prompt to use for extracting triples.
            parse_fn (callable): A function to parse the output of the language model.
            num_workers (int): The number of workers to use for parallel processing.
            max_paths_per_chunk (int): The maximum number of paths to extract per chunk.
            raise_on_error (bool): Whether to raise exceptions if extraction fails. Defaults to False.

        """
        from llama_index.core import Settings

        if isinstance(extract_prompt, str):
            extract_prompt = PromptTemplate(extract_prompt)

        super().__init__(
            llm=llm or Settings.llm,
            extract_prompt=extract_prompt or DEFAULT_KG_TRIPLET_EXTRACT_PROMPT,
            parse_fn=parse_fn,
            num_workers=num_workers,
            max_paths_per_chunk=max_paths_per_chunk,
            raise_on_error=raise_on_error,
        )

    @classmethod
    def class_name(cls) -> str:
        return "SimpleLLMPathExtractor"

    def __call__(
        self, nodes: Sequence[BaseNode], show_progress: bool = False, **kwargs: Any
    ) -> Sequence[BaseNode]:
        """Extract triples from nodes."""
        return asyncio_run(self.acall(nodes, show_progress=show_progress, **kwargs))

    async def _aextract(self, node: BaseNode) -> BaseNode:
        """Extract triples from a node."""
        assert hasattr(node, "text")

        text = node.get_content(metadata_mode=MetadataMode.LLM)
        try:
            llm_response = await self.llm.apredict(
                self.extract_prompt,
                text=text,
                max_knowledge_triplets=self.max_paths_per_chunk,
            )
            triples = self.parse_fn(llm_response)
        except ValueError as e:
            logger.error(f"Error during extraction: {e!s}", exc_info=True)
            if self.raise_on_error:
                raise
            triples = []

        existing_nodes = node.metadata.pop(KG_NODES_KEY, [])
        existing_relations = node.metadata.pop(KG_RELATIONS_KEY, [])

        metadata = node.metadata.copy()
        for subj, rel, obj in triples:
            subj_node = EntityNode(name=subj, properties=metadata)
            obj_node = EntityNode(name=obj, properties=metadata)
            rel_node = Relation(
                label=rel,
                source_id=subj_node.id,
                target_id=obj_node.id,
                properties=metadata,
            )

            existing_nodes.extend([subj_node, obj_node])
            existing_relations.append(rel_node)

        node.metadata[KG_NODES_KEY] = existing_nodes
        node.metadata[KG_RELATIONS_KEY] = existing_relations

        return node

    async def acall(
        self, nodes: Sequence[BaseNode], show_progress: bool = False, **kwargs: Any
    ) -> Sequence[BaseNode]:
        """Extract triples from nodes async."""
        jobs = []
        for node in nodes:
            jobs.append(self._aextract(node))

        return await run_jobs(
            jobs,
            workers=self.num_workers,
            show_progress=show_progress,
            desc="Extracting paths from text",
        )
