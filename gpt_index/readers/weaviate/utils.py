"""Weaviate utils."""

from typing import Any, Dict, List, Set, cast

from gpt_index.utils import get_new_int_id

DEFAULT_CLASS_PREFIX_STUB = "Gpt_Index"


def get_default_class_prefix(current_id_set: Set = set()) -> str:
    """Get default class prefix."""
    return DEFAULT_CLASS_PREFIX_STUB + "_" + str(get_new_int_id(current_id_set))


def validate_client(client: Any) -> None:
    """Validate client and import weaviate library."""
    try:
        import weaviate  # noqa: F401
        from weaviate import Client

        client = cast(Client, client)
    except ImportError:
        raise ValueError(
            "Weaviate is not installed. "
            "Please install it with `pip install weaviate-client`."
        )
    cast(Client, client)


def parse_get_response(response: Dict) -> Dict:
    """Parse get response from Weaviate."""
    if "errors" in response:
        raise ValueError("Invalid query, got errors: {}".format(response["errors"]))
    data_response = response["data"]
    if "Get" not in data_response:
        raise ValueError("Invalid query response, must be a Get query.")

    return data_response["Get"]


def get_by_id(
    client: Any, object_id: str, class_name: str, properties: List[str]
) -> Dict:
    """Get response by id from Weaviate."""
    validate_client(client)

    where_filter = {"path": ["id"], "operator": "Equal", "valueString": object_id}
    query_result = (
        client.query.get(class_name, properties)
        .with_where(where_filter)
        .with_additional(["id", "vector"])
        .do()
    )

    parsed_result = parse_get_response(query_result)
    entries = parsed_result[class_name]
    if len(entries) == 0:
        raise ValueError("No entry found for the given id")
    elif len(entries) > 1:
        raise ValueError("More than one entry found for the given id")
    return entries[0]
