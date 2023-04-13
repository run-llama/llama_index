from gpt_index.constants import TYPE_KEY
from gpt_index.data_structs.node_v2 import ImageNode, IndexNode, Node
from gpt_index.readers.schema.base import Document
from gpt_index.schema import BaseDocument


def doc_to_json(doc: BaseDocument) -> dict:
    doc_dict = doc.to_dict()
    doc_dict[TYPE_KEY] = doc.get_type()
    return doc_dict


def json_to_doc(doc_dict: dict) -> BaseDocument:
    doc_type = doc_dict.pop(TYPE_KEY, None)
    doc: BaseDocument
    if doc_type == "Document" or doc_type is None:
        doc = Document.from_dict(doc_dict)
    elif doc_type == Node.get_type():
        doc = Node.from_dict(doc_dict)
    elif doc_type == ImageNode.get_type():
        doc = ImageNode.from_dict(doc_dict)
    elif doc_type == IndexNode.get_type():
        doc = IndexNode.from_dict(doc_dict)
    else:
        raise ValueError(f"Unknown doc type: {doc_type}")

    return doc
