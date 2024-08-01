from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Type,
    cast,
)

if TYPE_CHECKING:
    from marvin import ai_model

from llama_index.legacy.bridge.pydantic import BaseModel, Field
from llama_index.legacy.extractors.interface import BaseExtractor
from llama_index.legacy.schema import BaseNode, TextNode
from llama_index.legacy.utils import get_tqdm_iterable


class MarvinMetadataExtractor(BaseExtractor):
    # Forward reference to handle circular imports
    marvin_model: Type["ai_model"] = Field(
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
        #create extractor list
        extractors = [
            TitleExtractor(nodes=1, llm=llm),
            MarvinMetadataExtractor(marvin_model=YourMarvinMetadataModel),
        ]

        #create node parser to parse nodes from document
        node_parser = SentenceSplitter(
            text_splitter=text_splitter
        )

        #use node_parser to get nodes from documents
        from llama_index.legacy.ingestion import run_transformations
        nodes = run_transformations(documents, [node_parser] + extractors)
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
        from marvin import ai_model

        if not issubclass(marvin_model, ai_model):
            raise ValueError("marvin_model must be a subclass of ai_model")

        if llm_model_string:
            marvin.settings.llm_model = llm_model_string

        super().__init__(
            marvin_model=marvin_model, llm_model_string=llm_model_string, **kwargs
        )

    @classmethod
    def class_name(cls) -> str:
        return "MarvinEntityExtractor"

    async def aextract(self, nodes: Sequence[BaseNode]) -> List[Dict]:
        from marvin import ai_model

        ai_model = cast(ai_model, self.marvin_model)
        metadata_list: List[Dict] = []

        nodes_queue: Iterable[BaseNode] = get_tqdm_iterable(
            nodes, self.show_progress, "Extracting marvin metadata"
        )
        for node in nodes_queue:
            if self.is_text_node_only and not isinstance(node, TextNode):
                metadata_list.append({})
                continue

            # TODO: Does marvin support async?
            metadata = ai_model(node.get_content())

            metadata_list.append({"marvin_metadata": metadata.dict()})
        return metadata_list
