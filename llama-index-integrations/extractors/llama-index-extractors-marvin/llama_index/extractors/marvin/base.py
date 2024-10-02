from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Sequence,
    Type,
)

from pydantic import BaseModel, Field
from llama_index.core.extractors.interface import BaseExtractor
from llama_index.core.schema import BaseNode, TextNode
from llama_index.core.utils import get_tqdm_iterable


class MarvinMetadataExtractor(BaseExtractor):
    # Forward reference to handle circular imports
    marvin_model: Type[BaseModel] = Field(
        description="The target pydantic model to cast the metadata into."
    )

    """Metadata extractor for custom metadata using Marvin.
    Node-level extractor. Extracts
    `marvin_metadata` metadata field.
    Args:
        marvin_model: The target pydantic model to cast the metadata into.
    Usage:
        #create extractor list
        extractors = [
            TitleExtractor(nodes=1, llm=llm),
            MarvinMetadataExtractor(marvin_model=YourMetadataModel),
        ]

        #create node parser to parse nodes from document
        node_parser = SentenceSplitter(
            text_splitter=text_splitter
        )

        #use node_parser to get nodes from documents
        from llama_index.ingestion import run_transformations
        nodes = run_transformations(documents, [node_parser] + extractors)
        print(nodes)
    """

    def __init__(
        self,
        marvin_model: Type[BaseModel],
        **kwargs: Any,
    ) -> None:
        """Init params."""
        super().__init__(marvin_model=marvin_model, **kwargs)

    @classmethod
    def class_name(cls) -> str:
        return "MarvinEntityExtractor"

    async def aextract(self, nodes: Sequence[BaseNode]) -> List[Dict]:
        from marvin import cast_async

        metadata_list: List[Dict] = []

        nodes_queue: Iterable[BaseNode] = get_tqdm_iterable(
            nodes, self.show_progress, "Extracting marvin metadata"
        )
        for node in nodes_queue:
            if self.is_text_node_only and not isinstance(node, TextNode):
                metadata_list.append({})
                continue

            metadata = await cast_async(node.get_content(), target=self.marvin_model)

            metadata_list.append({"marvin_metadata": metadata.model_dump()})
        return metadata_list
