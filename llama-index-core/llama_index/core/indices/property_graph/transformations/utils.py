from typing import Any, Dict, List, Optional

try:
    from typing import TypeAlias  # type: ignore
except ImportError:
    # python 3.8 and 3.9 compatibility
    from typing import Any as TypeAlias  # type: ignore

from llama_index.core.bridge.pydantic import create_model, Field


def get_entity_class(
    possible_entities: TypeAlias,
    possible_entity_props: Optional[List[str]],
    strict: bool,
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
