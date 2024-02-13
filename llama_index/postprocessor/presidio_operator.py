from typing import Dict

from presidio_anonymizer.operators import OperatorType, Operator

class EntityTypeCountAnonymizer(Operator):
    """
    Anonymizer which replaces the entity value
    with an type counter per entity.
    """

    REPLACING_FORMAT = "<{entity_type}_{index}>"

    def operate(self, text: str, params: Dict = None) -> str:
        """Anonymize the input text."""

        entity_type: str = params["entity_type"]
        entity_mapping: Dict[str:Dict] = params["entity_mapping"]
        deanonymize_mapping: Dict[str:str] = params["deanonymize_mapping"]

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

    def validate(self, params: Dict = None) -> None:
        """Validate operator parameters."""

        if "entity_mapping" not in params:
            raise ValueError("An input Dict called `entity_mapping` is required.")
        if "entity_type" not in params:
            raise ValueError("An entity_type param is required.")

    def operator_name(self) -> str:
        return self.__class__.__name__

    def operator_type(self) -> OperatorType:
        return OperatorType.Anonymize