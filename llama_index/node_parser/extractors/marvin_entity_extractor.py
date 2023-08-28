from typing import List, Optional, Sequence, Type, Dict

from llama_index.schema import BaseNode, TextNode

from llama_index.node_parser.extractors.metadata_extractors import (
    MetadataFeatureExtractor,
)


class MarvinEntityExtractor(MetadataFeatureExtractor):
    """Entity extractor for cusstom entities using Marvin. Node-level extractor. Extracts
    `marvin_entities` metadata field.
    Args:
        marvin_model: : Type[AIModel] Marvin model to use for extracting entities
        llm_model_string: (optional) LLM model string to use for extracting entities
    Usage:
        #create metadata extractor
        metadata_extractor = MetadataExtractor(
            extractors=[
                TitleExtractor(nodes=1, llm=llm),
                MarvinEntityExtractor(marvin_model=BusinessDocExcerpt), #let's extract custom entities for each node.
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

    def __init__(self, marvin_model, llm_model_string: Optional[str] = None) -> None:
        """Init params."""
        from marvin import AIModel, settings

        self._marvin_model: Type[AIModel] = marvin_model
        self.llm_model_string = llm_model_string

        if self.llm_model_string:
            settings.llm_model = llm_model_string

    def extract(self, nodes: Sequence[BaseNode]) -> List[Dict]:
        metadata_list: List[Dict] = []
        for node in nodes:
            if self.is_text_node_only and not isinstance(node, TextNode):
                metadata_list.append({})
                continue

            entities = self._marvin_model(node.get_content())

            metadata_list.append({"marvin_entities": entities.dict()})
        return metadata_list
