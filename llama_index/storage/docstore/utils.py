from llama_index.constants import DATA_KEY, TYPE_KEY
from llama_index.readers.schema.base import Document
from llama_index.schema import BaseNode, ImageNode, IndexNode, TextNode


def doc_to_json(doc: BaseNode) -> dict:
    return {
        DATA_KEY: doc.dict(),
        TYPE_KEY: doc.get_type(),
    }


def json_to_doc(doc_dict: dict) -> BaseNode:
    doc_type = doc_dict[TYPE_KEY]
    data_dict = doc_dict[DATA_KEY]
    doc: BaseNode
    if doc_type == Document.get_type():
        doc = Document.parse_obj(data_dict)
    elif doc_type == TextNode.get_type():
        doc = TextNode.parse_obj(data_dict)
    elif doc_type == ImageNode.get_type():
        doc = ImageNode.parse_obj(data_dict)
    elif doc_type == IndexNode.get_type():
        doc = IndexNode.parse_obj(data_dict)
    else:
        raise ValueError(f"Unknown doc type: {doc_type}")

    return doc
