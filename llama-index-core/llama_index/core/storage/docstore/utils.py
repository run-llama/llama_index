from llama_index.core.constants import DATA_KEY, TYPE_KEY
from llama_index.core.schema import (
    BaseNode,
    Document,
    ImageDocument,
    ImageNode,
    IndexNode,
    Node,
    NodeRelationship,
    RelatedNodeInfo,
    TextNode,
)


def doc_to_json(doc: BaseNode) -> dict:
    return {
        DATA_KEY: doc.to_dict(),
        TYPE_KEY: doc.get_type(),
    }


def json_to_doc(doc_dict: dict) -> BaseNode:
    doc_type = doc_dict[TYPE_KEY]
    data_dict = doc_dict[DATA_KEY]
    doc: BaseNode

    if "extra_info" in data_dict:
        return legacy_json_to_doc(doc_dict)
    else:
        if doc_type == Document.get_type():
            if data_dict["class_name"] == ImageDocument.class_name():
                doc = ImageDocument.from_dict(data_dict)
            else:
                doc = Document.from_dict(data_dict)
        elif doc_type == TextNode.get_type():
            doc = TextNode.from_dict(data_dict)
        elif doc_type == ImageNode.get_type():
            doc = ImageNode.from_dict(data_dict)
        elif doc_type == IndexNode.get_type():
            doc = IndexNode.from_dict(data_dict)
        elif doc_type == Node.get_type():
            doc = Node.from_dict(data_dict)
        else:
            raise ValueError(f"Unknown doc type: {doc_type}")

        return doc


def legacy_json_to_doc(doc_dict: dict) -> BaseNode:
    """Todo: Deprecated legacy support for old node versions."""
    doc_type = doc_dict[TYPE_KEY]
    data_dict = doc_dict[DATA_KEY]
    doc: BaseNode

    text = data_dict.get("text", "")
    metadata = data_dict.get("extra_info", {}) or {}
    id_ = data_dict.get("doc_id", None)

    relationships = data_dict.get("relationships", {})
    relationships = {
        NodeRelationship(k): RelatedNodeInfo(node_id=str(v))
        for k, v in relationships.items()
    }

    if doc_type == Document.get_type():
        doc = Document(
            text=text, metadata=metadata, id=id_, relationships=relationships
        )
    elif doc_type == TextNode.get_type():
        doc = TextNode(
            text=text, metadata=metadata, id=id_, relationships=relationships
        )
    elif doc_type == ImageNode.get_type():
        image = data_dict.get("image", None)
        doc = ImageNode(
            text=text,
            metadata=metadata,
            id=id_,
            relationships=relationships,
            image=image,
        )
    elif doc_type == IndexNode.get_type():
        index_id = data_dict.get("index_id", None)
        doc = IndexNode(
            text=text,
            metadata=metadata,
            id=id_,
            relationships=relationships,
            index_id=index_id,
        )
    else:
        raise ValueError(f"Unknown doc type: {doc_type}")

    return doc
