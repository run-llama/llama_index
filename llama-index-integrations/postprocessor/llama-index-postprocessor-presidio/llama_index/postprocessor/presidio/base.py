from typing import Any, Dict, List, Optional, Tuple
from copy import deepcopy

from presidio_anonymizer.operators import Operator, OperatorType
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import MetadataMode, NodeWithScore, QueryBundle

from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig


class EntityTypeCountAnonymizer(Operator):
    """
    Anonymizer which replaces the entity value
    with an type counter per entity.
    """

    REPLACING_FORMAT = "<{entity_type}_{index}>"

    def operate(self, text: str, params: Dict[str, Any]) -> str:
        """Anonymize the input text."""
        entity_type: str = params["entity_type"]
        entity_mapping: Dict[str, Dict] = params["entity_mapping"]
        deanonymize_mapping: Dict[str, str] = params["deanonymize_mapping"]

        entity_mapping_for_type = entity_mapping.get(entity_type)
        if not entity_mapping_for_type:
            entity_mapping_for_type = entity_mapping[entity_type] = {}

        if text in entity_mapping_for_type:
            return entity_mapping_for_type[text]

        new_text = self.REPLACING_FORMAT.format(
            entity_type=entity_type, index=len(entity_mapping_for_type) + 1
        )
        entity_mapping[entity_type][text] = new_text
        deanonymize_mapping[new_text] = text
        return new_text

    def validate(self, params: Dict[str, Any]) -> None:
        """Validate operator parameters."""
        if "entity_mapping" not in params:
            raise ValueError("An input Dict called `entity_mapping` is required.")
        if "entity_type" not in params:
            raise ValueError("An entity_type param is required.")
        if "deanonymize_mapping" not in params:
            raise ValueError("A deanonymize_mapping param is required.")

    def operator_name(self) -> str:
        return self.__class__.__name__

    def operator_type(self) -> OperatorType:
        return OperatorType.Anonymize


class PresidioPIINodePostprocessor(BaseNodePostprocessor):
    """
    presidio PII Node processor.
    Uses a presidio to analyse PIIs.
    """

    pii_node_info_key: str = "__pii_node_info__"
    entity_mapping: Dict[str, Dict] = {}
    mapping: Dict[str, str] = {}

    @classmethod
    def class_name(cls) -> str:
        return "PresidioPIINodePostprocessor"

    def mask_pii(self, text: str) -> Tuple[str, Dict]:
        analyzer = AnalyzerEngine()
        results = analyzer.analyze(text=text, language="en")
        engine = AnonymizerEngine()
        engine.add_anonymizer(EntityTypeCountAnonymizer)

        new_text = engine.anonymize(
            text=text,
            analyzer_results=results,
            operators={
                "DEFAULT": OperatorConfig(
                    "EntityTypeCountAnonymizer",
                    {
                        "entity_mapping": self.entity_mapping,
                        "deanonymize_mapping": self.mapping,
                    },
                )
            },
        )

        return new_text.text

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        """Postprocess nodes."""
        # swap out text from nodes, with the original node mappings
        new_nodes = []
        for node_with_score in nodes:
            node = node_with_score.node
            new_text = self.mask_pii(node.get_content(metadata_mode=MetadataMode.LLM))
            new_node = deepcopy(node)
            new_node.excluded_embed_metadata_keys.append(self.pii_node_info_key)
            new_node.excluded_llm_metadata_keys.append(self.pii_node_info_key)
            new_node.metadata[self.pii_node_info_key] = self.mapping
            new_node.set_content(new_text)
            new_nodes.append(NodeWithScore(node=new_node, score=node_with_score.score))

        return new_nodes
