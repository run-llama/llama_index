from typing import List, Optional

from llama_index.core.bridge.pydantic import Field
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle


class MetadataReplacementPostProcessor(BaseNodePostprocessor):
    target_metadata_key: str = Field(
        description="Target metadata key to replace node content with."
    )

    def __init__(self, target_metadata_key: str) -> None:
        super().__init__(target_metadata_key=target_metadata_key)

    @classmethod
    def class_name(cls) -> str:
        return "MetadataReplacementPostProcessor"

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        result = []
        for node_with_score in nodes:
            replacement = node_with_score.node.metadata.get(self.target_metadata_key)
            if replacement is None:
                result.append(node_with_score)
                continue

            node = node_with_score.node.model_copy()
            node.set_content(replacement)
            result.append(node_with_score.model_copy(update={"node": node}))

        return result
