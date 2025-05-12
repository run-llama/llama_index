from llama_index.core.constants import DATA_KEY, TYPE_KEY
from llama_index.core.schema import (
    BaseNode,
    Document,
    ImageDocument,
    ImageNode,
    IndexNode,
    NodeRelationship,
    RelatedNodeInfo,
    TextNode,
)


def doc_to_json(doc: BaseNode) -> dict:
    return {
        DATA_KEY: doc.to_dict(),
        TYPE_KEY: doc.get_type(),
    }

def json_to_doc(doc_dict: dict) -> BaseNode | Document:
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
        elif doc_type == TextNode.get_type() or doc_type == ImageNode.get_type() or doc_type == IndexNode.get_type():
            rels: dict = data_dict.get("relationships", None)
            _is_document = False
            if rels is not None:
                source = rels.get("1")
                if source is not None:
                    if source["node_type"] == Document.get_type():
                        doc = Document.from_dict(data_dict)
                        _is_document = True
            if not _is_document:
                if doc_type == TextNode.get_type():
                    return TextNode.from_dict(data_dict)
                elif doc_type == ImageNode.get_type():
                    return ImageNode.from_dict(data_dict)
                else:
                    return IndexNode.from_dict(data_dict)
        else:
            raise ValueError(f"Unknown doc type: {doc_type}")
        return doc

def legacy_json_to_doc(doc_dict: dict) -> Document:
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
