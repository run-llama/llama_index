import json
from typing import Any, Dict, Optional, Tuple, Callable, Mapping, cast

from llama_index.core.schema import (
    BaseNode,
    ImageNode,
    Node,
    IndexNode,
    NodeRelationship,
    RelatedNodeInfo,
    TextNode,
)
from llama_index.core.vector_stores.types import (
    MetadataFilters,
    FilterOperator,
    FilterCondition,
)


DEFAULT_TEXT_KEY = "text"
DEFAULT_TEXT_RESOURCE_KEY = "text_resource"
DEFAULT_EMBEDDING_KEY = "embedding"
DEFAULT_DOC_ID_KEY = "doc_id"


def _validate_is_flat_dict(metadata_dict: dict) -> None:
    """
    Validate that metadata dict is flat,
    and key is str, and value is one of (str, int, float, None).
    """
    for key, val in metadata_dict.items():
        if not isinstance(key, str):
            raise ValueError("Metadata key must be str!")
        if not isinstance(val, (str, int, float, type(None))):
            raise ValueError(
                f"Value for metadata {key} must be one of (str, int, float, None)"
            )


def node_to_metadata_dict(
    node: BaseNode,
    remove_text: bool = False,
    text_field: str = DEFAULT_TEXT_KEY,
    text_resource_field: str = DEFAULT_TEXT_RESOURCE_KEY,
    flat_metadata: bool = False,
) -> Dict[str, Any]:
    """Common logic for saving Node data into metadata dict."""
    # Using mode="json" here because BaseNode may have fields of type bytes (e.g. images in ImageBlock),
    # which would cause serialization issues.
    node_dict = node.model_dump(mode="json")
    metadata: Dict[str, Any] = node_dict.get("metadata", {})

    if flat_metadata:
        _validate_is_flat_dict(metadata)

    # store entire node as json string - some minor text duplication
    if remove_text and text_field in node_dict:
        node_dict[text_field] = ""
    if remove_text and text_resource_field in node_dict:
        del node_dict[text_resource_field]

    # remove embedding from node_dict
    node_dict["embedding"] = None

    # dump remainder of node_dict to json string
    metadata["_node_content"] = json.dumps(node_dict, ensure_ascii=False)
    metadata["_node_type"] = node.class_name()

    # store ref doc id at top level to allow metadata filtering
    # kept for backwards compatibility, will consolidate in future
    metadata["document_id"] = node.ref_doc_id or "None"  # for Chroma
    metadata["doc_id"] = node.ref_doc_id or "None"  # for Pinecone, Qdrant, Redis
    metadata["ref_doc_id"] = node.ref_doc_id or "None"  # for Weaviate

    return metadata


def metadata_dict_to_node(metadata: dict, text: Optional[str] = None) -> BaseNode:
    """Common logic for loading Node data from metadata dict."""
    node_json = metadata.get("_node_content")
    node_type = metadata.get("_node_type")
    if node_json is None:
        raise ValueError("Node content not found in metadata dict.")

    node: BaseNode
    if node_type == Node.class_name():
        node = Node.from_json(node_json)
    elif node_type == IndexNode.class_name():
        node = IndexNode.from_json(node_json)
    elif node_type == ImageNode.class_name():
        node = ImageNode.from_json(node_json)
    else:
        node = TextNode.from_json(node_json)

    if text is not None:
        node.set_content(text)

    return node


def build_metadata_filter_fn(
    metadata_lookup_fn: Callable[[str], Mapping[str, Any]],
    metadata_filters: Optional[MetadataFilters] = None,
) -> Callable[[str], bool]:
    """Build metadata filter function."""
    filter_list = metadata_filters.filters if metadata_filters else []
    if not filter_list or not metadata_filters:
        return lambda _: True

    filter_condition = cast(MetadataFilters, metadata_filters.condition)

    def filter_fn(node_id: str) -> bool:
        def _process_filter_match(
            operator: FilterOperator, value: Any, metadata_value: Any
        ) -> bool:
            if metadata_value is None:
                return False
            if operator == FilterOperator.EQ:
                return metadata_value == value
            if operator == FilterOperator.NE:
                return metadata_value != value
            if operator == FilterOperator.GT:
                return metadata_value > value
            if operator == FilterOperator.GTE:
                return metadata_value >= value
            if operator == FilterOperator.LT:
                return metadata_value < value
            if operator == FilterOperator.LTE:
                return metadata_value <= value
            if operator == FilterOperator.IN:
                return metadata_value in value
            if operator == FilterOperator.NIN:
                return metadata_value not in value
            if operator == FilterOperator.CONTAINS:
                return value in metadata_value
            if operator == FilterOperator.TEXT_MATCH:
                return value.lower() in metadata_value.lower()
            if operator == FilterOperator.ALL:
                return all(val in metadata_value for val in value)
            if operator == FilterOperator.ANY:
                return any(val in metadata_value for val in value)
            raise ValueError(f"Invalid operator: {operator}")

        metadata = metadata_lookup_fn(node_id)

        filter_matches_list = []
        for filter_ in filter_list:
            if isinstance(filter_, MetadataFilters):
                raise ValueError("Nested MetadataFilters are not supported.")

            filter_matches = True
            metadata_value = metadata.get(filter_.key, None)
            if filter_.operator == FilterOperator.IS_EMPTY:
                filter_matches = (
                    metadata_value is None
                    or metadata_value == ""
                    or metadata_value == []
                )
            else:
                filter_matches = _process_filter_match(
                    operator=filter_.operator,
                    value=filter_.value,
                    metadata_value=metadata_value,
                )

            filter_matches_list.append(filter_matches)

        if filter_condition == FilterCondition.AND:
            return all(filter_matches_list)
        elif filter_condition == FilterCondition.OR:
            return any(filter_matches_list)
        else:
            raise ValueError(f"Invalid filter condition: {filter_condition}")

    return filter_fn


# TODO: Deprecated conversion functions
def legacy_metadata_dict_to_node(
    metadata: dict, text_key: str = DEFAULT_TEXT_KEY
) -> Tuple[dict, dict, dict]:
    """Common logic for loading Node data from metadata dict."""
    # make a copy first
    if metadata is None:
        metadata = {}
    else:
        metadata = metadata.copy()

    # load node_info from json string
    node_info_str = metadata.pop("node_info", "")
    if node_info_str == "":
        node_info = {}
    else:
        node_info = json.loads(node_info_str)

    # load relationships from json string
    relationships_str = metadata.pop("relationships", "")
    relationships: Dict[NodeRelationship, RelatedNodeInfo]
    if relationships_str == "":
        relationships = {}
    else:
        relationships = {
            NodeRelationship(k): RelatedNodeInfo(node_id=str(v))
            for k, v in json.loads(relationships_str).items()
        }

    # remove other known fields
    metadata.pop(text_key, None)

    id_ = metadata.pop("id", None)
    document_id = metadata.pop("document_id", None)
    doc_id = metadata.pop("doc_id", None)
    ref_doc_id = metadata.pop("ref_doc_id", None)

    # don't remove id's from metadata that llama-index doesn't know about
    ref_doc_id_info = relationships.get(NodeRelationship.PARENT, None)
    if ref_doc_id_info is not None:
        ref_doc_id = ref_doc_id_info.node_id

    if id_ is not None and id_ != ref_doc_id:
        metadata["id"] = id_
    if document_id is not None and document_id != ref_doc_id:
        metadata["document_id"] = document_id
    if doc_id is not None and doc_id != ref_doc_id:
        metadata["doc_id"] = doc_id

    # remaining metadata is metadata or node_info
    new_metadata = {}
    for key, val in metadata.items():
        # don't enforce types on metadata anymore (we did in the past)
        # since how we store this data now has been updated
        new_metadata[key] = val

    return new_metadata, node_info, relationships
