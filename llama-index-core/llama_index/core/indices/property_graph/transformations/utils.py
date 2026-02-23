from typing import Any, Dict, List, Optional

try:
    from typing import TypeAlias  # type: ignore
except ImportError:
    # python 3.8 and 3.9 compatibility
    from typing import Any as TypeAlias  # type: ignore

from llama_index.core.bridge.pydantic import ConfigDict, create_model, Field


def _clean_additional_properties(schema: Dict[str, Any]) -> None:
    """
    Recursively set ``additionalProperties: true`` to ``false`` in a JSON schema.

    Pydantic generates ``additionalProperties: true`` for ``Dict[str, Any]``
    fields. This is incompatible with OpenAI structured outputs (which require
    ``false``) and Google Gemini (which rejects the field entirely when set to
    ``true``). Setting it to ``false`` satisfies both APIs.
    """
    if isinstance(schema, dict):
        if schema.get("additionalProperties") is True:
            schema["additionalProperties"] = False
        for value in schema.values():
            _clean_additional_properties(value)
    elif isinstance(schema, list):
        for item in schema:
            _clean_additional_properties(item)


def get_entity_class(
    possible_entities: TypeAlias,
    possible_entity_props: Optional[List[str]],
    strict: bool,
    clean_additional_properties: bool = False,
) -> Any:
    """Get entity class."""
    if not possible_entity_props:
        return create_model(
            "Entity",
            type=(
                possible_entities if strict else str,
                Field(
                    ...,
                    description=(
                        "Entity in a knowledge graph. Only extract entities with types that are listed as valid: "
                        + str(possible_entities)
                    ),
                ),
            ),
            name=(str, ...),
        )
    else:
        config_kwargs = {}
        if clean_additional_properties:
            config_kwargs["__config__"] = ConfigDict(
                json_schema_extra=_clean_additional_properties
            )
        return create_model(
            "Entity",
            **config_kwargs,
            type=(
                possible_entities if strict else str,
                Field(
                    ...,
                    description=(
                        "Entity in a knowledge graph. Only extract entities with types that are listed as valid: "
                        + str(possible_entities)
                    ),
                ),
            ),
            name=(str, ...),
            properties=(
                Optional[Dict[str, Any]],
                Field(
                    None,
                    description=(
                        "Properties of the entity. Only extract the following valid properties: "
                        + "\n".join(possible_entity_props)
                    ),
                ),
            ),
        )


def get_relation_class(
    possible_relations: TypeAlias,
    possible_relation_props: Optional[List[str]],
    strict: bool,
    clean_additional_properties: bool = False,
) -> Any:
    """Get relation class."""
    if not possible_relation_props:
        return create_model(
            "Relation",
            type=(
                possible_relations if strict else str,
                Field(
                    ...,
                    description=(
                        "Relation in a knowledge graph. Only extract relations with types that are listed as valid: "
                        + str(possible_relations)
                    ),
                ),
            ),
        )
    else:
        config_kwargs = {}
        if clean_additional_properties:
            config_kwargs["__config__"] = ConfigDict(
                json_schema_extra=_clean_additional_properties
            )
        return create_model(
            "Relation",
            **config_kwargs,
            type=(
                possible_relations if strict else str,
                Field(
                    ...,
                    description=(
                        "Relation in a knowledge graph. Only extract relations with types that are listed as valid: "
                        + str(possible_relations)
                    ),
                ),
            ),
            properties=(
                Optional[Dict[str, Any]],
                Field(
                    None,
                    description=(
                        "Properties of the relation. Only extract the following valid properties: "
                        + "\n".join(possible_relation_props)
                    ),
                ),
            ),
        )
