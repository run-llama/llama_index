import asyncio
from typing import Any, List

from llama_index.core.async_utils import run_jobs
from llama_index.core.graph_stores.types import (
    EntityNode,
    Relation,
    KG_NODES_KEY,
    KG_RELATIONS_KEY,
)
from llama_index.core.schema import TransformComponent, BaseNode

DEFAULT_NODE_TYPE = "Entity"


class RelikPathExtractor(TransformComponent):
    """
    A transformer class for converting documents into graph structures
    using the Relik library and models.
    This class leverages relik models for extracting relationships
    and nodes from text documents and converting them into a graph format.
    The relationships are filtered based on a specified confidence threshold.
    For more details on the Relik library, visit their GitHub repository:
      https://github.com/SapienzaNLP/relik
    Args:
        model (str): The name of the pretrained Relik model to use.
          Default is "relik-ie/relik-relation-extraction-small-wikipedia".
        relationship_confidence_threshold (float): The confidence threshold for
          filtering relationships. Default is 0.0.
    """

    relik_model: Any
    relationship_confidence_threshold: float
    num_workers: int

    def __init__(
        self,
        model: str = "relik-ie/relik-relation-extraction-small-wikipedia",
        relationship_confidence_threshold: float = 0.0,
        num_workers: int = 4,
    ) -> None:
        """Init params."""
        try:
            import relik  # type: ignore
        except ImportError:
            raise ImportError(
                "Could not import relik python package. "
                "Please install it with `pip install relik`."
            )
        relik_model = relik.Relik.from_pretrained(model)
        super().__init__(
            relik_model=relik_model,
            relationship_confidence_threshold=relationship_confidence_threshold,
            num_workers=num_workers,
        )

    @classmethod
    def class_name(cls) -> str:
        return "RelikPathExtractor"

    def __call__(
        self, nodes: List[BaseNode], show_progress: bool = False, **kwargs: Any
    ) -> List[BaseNode]:
        """Extract triples from nodes."""
        return asyncio.run(self.acall(nodes, show_progress=show_progress, **kwargs))

    async def _aextract(self, node: BaseNode) -> BaseNode:
        """Extract triples from a node."""
        assert hasattr(node, "text")

        text = node.get_content(metadata_mode="llm")
        try:
            relik_out = self.relik_model(text)
        except ValueError:
            triples = []

        existing_nodes = node.metadata.pop(KG_NODES_KEY, [])
        existing_relations = node.metadata.pop(KG_RELATIONS_KEY, [])

        metadata = node.metadata.copy()
        # Extract nodes
        for n in relik_out.spans:
            existing_nodes.append(
                EntityNode(
                    name=n.text,
                    label=DEFAULT_NODE_TYPE
                    if n.label.strip() == "--NME--"
                    else n.label.strip(),
                    properties=metadata,
                )
            )
        # Extract relationships
        for triple in relik_out.triplets:
            # Ignore relationship if below confidence threshold
            if triple.confidence < self.relationship_confidence_threshold:
                continue
            rel_node = Relation(
                label=triple.label.replace(" ", "_").upper(),
                source_id=triple.subject.text,
                target_id=triple.object.text,
                properties=metadata,
            )

            existing_relations.append(rel_node)

        node.metadata[KG_NODES_KEY] = existing_nodes
        node.metadata[KG_RELATIONS_KEY] = existing_relations

        return node

    async def acall(
        self, nodes: List[BaseNode], show_progress: bool = False, **kwargs: Any
    ) -> List[BaseNode]:
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
