from llama_index.constants import DATA_KEY, TYPE_KEY
from llama_index.data_structs.node import ImageNode, IndexNode, Node
from llama_index.readers.schema.base import Document
from llama_index.schema import BaseDocument


def doc_to_json(doc: BaseDocument) -> dict:
    return {
        DATA_KEY: doc.to_dict(),
        TYPE_KEY: doc.get_type(),
    }


def json_to_doc(doc_dict: dict) -> BaseDocument:
    doc_type = doc_dict[TYPE_KEY]
    data_dict = doc_dict[DATA_KEY]
    doc: BaseDocument
    if doc_type == Document.get_type():
        doc = Document.from_dict(data_dict)
    elif doc_type == Node.get_type():
        doc = Node.from_dict(data_dict)
    elif doc_type == ImageNode.get_type():
        doc = ImageNode.from_dict(data_dict)
    elif doc_type == IndexNode.get_type():
        doc = IndexNode.from_dict(data_dict)
    else:
        raise ValueError(f"Unknown doc type: {doc_type}")

    return doc
