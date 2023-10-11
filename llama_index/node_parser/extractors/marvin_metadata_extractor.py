from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Type, cast

if TYPE_CHECKING:
    from marvin import AIModel

from llama_index.bridge.pydantic import BaseModel, Field
from llama_index.node_parser.extractors.metadata_extractors import (
    MetadataFeatureExtractor,
)
from llama_index.schema import BaseNode, TextNode


class MarvinMetadataExtractor(MetadataFeatureExtractor):
    # Forward reference to handle circular imports
    marvin_model: Type["AIModel"] = Field(
        description="The Marvin model to use for extracting custom metadata"
    )
    llm_model_string: Optional[str] = Field(
        description="The LLM model string to use for extracting custom metadata"
    )

    """Metadata extractor for custom metadata using Marvin.
    Node-level extractor. Extracts
    `marvin_metadata` metadata field.
    Args:
        marvin_model: Marvin model to use for extracting metadata
        llm_model_string: (optional) LLM model string to use for extracting metadata
    Usage:
        #create metadata extractor
        metadata_extractor = MetadataExtractor(
            extractors=[
                TitleExtractor(nodes=1, llm=llm),
                MarvinMetadataExtractor(marvin_model=YourMarvinMetadataModel),
            ],
        )

        #create node parser to parse nodes from document
        node_parser = SimpleNodeParser(
            text_splitter=text_splitter,
            metadata_extractor=metadata_extractor,
        )

        #use node_parser to get nodes from documents
        nodes = node_parser.get_nodes_from_documents([Document(text=text)])
        print(nodes)
    """

    def __init__(
        self,
        marvin_model: Type[BaseModel],
        llm_model_string: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Init params."""
        import marvin
        from marvin import AIModel

        if not issubclass(marvin_model, AIModel):
            raise ValueError("marvin_model must be a subclass of AIModel")

        if llm_model_string:
            marvin.settings.llm_model = llm_model_string

        super().__init__(
            marvin_model=marvin_model, llm_model_string=llm_model_string, **kwargs
        )

    @classmethod
    def class_name(cls) -> str:
        return "MarvinEntityExtractor"

    def extract(self, nodes: Sequence[BaseNode]) -> List[Dict]:
        from marvin import AIModel

        ai_model = cast(AIModel, self.marvin_model)
        metadata_list: List[Dict] = []
        for node in nodes:
            if self.is_text_node_only and not isinstance(node, TextNode):
                metadata_list.append({})
                continue

            metadata = ai_model(node.get_content())

            metadata_list.append({"marvin_metadata": metadata.dict()})
        return metadata_list
