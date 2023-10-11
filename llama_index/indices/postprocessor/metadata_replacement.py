from typing import List, Optional

from llama_index.bridge.pydantic import Field
from llama_index.indices.postprocessor.types import BaseNodePostprocessor
from llama_index.indices.query.schema import QueryBundle
from llama_index.schema import MetadataMode, NodeWithScore


class MetadataReplacementPostProcessor(BaseNodePostprocessor):
    target_metadata_key: str = Field(
        description="Target metadata key to replace node content with."
    )

    def __init__(self, target_metadata_key: str) -> None:
        super().__init__(target_metadata_key=target_metadata_key)

    @classmethod
    def class_name(cls) -> str:
        return "MetadataReplacementPostProcessor"

    def postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        for n in nodes:
            n.node.set_content(
                n.node.metadata.get(
                    self.target_metadata_key,
                    n.node.get_content(metadata_mode=MetadataMode.NONE),
                )
            )

        return nodes
