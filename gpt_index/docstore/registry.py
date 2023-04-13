from enum import Enum
from typing import Any, Dict, Optional, Sequence, Type

from gpt_index.constants import DATA_KEY, TYPE_KEY
from gpt_index.docstore.simple_docstore import SimpleDocumentStore
from gpt_index.docstore.mongo_docstore import MongoDocumentStore
from gpt_index.docstore.types import DocumentStore


class DocumentStoreType(str, Enum):
    MONGO = "mongo"
    SIMPLE = "simple"


DOCSTORE_TYPE_TO_CLASS: Dict[DocumentStoreType, Type[DocumentStore]] = {
    DocumentStoreType.MONGO: MongoDocumentStore,
    DocumentStoreType.SIMPLE: SimpleDocumentStore,
}


DOCSTORE_CLASS_TO_TYPE: Dict[Type[DocumentStore], DocumentStoreType] = {
    cls_: type_ for type_, cls_ in DOCSTORE_TYPE_TO_CLASS.items()
}


def get_default_docstore() -> DocumentStore:
    return SimpleDocumentStore()


def load_docstore_from_dict(
    docstore_dict: Dict[str, Any],
    type_to_cls: Optional[Dict[DocumentStoreType, Type[DocumentStore]]] = None,
    **kwargs: Any,
):
    type_to_cls = type_to_cls or DOCSTORE_TYPE_TO_CLASS
    type = docstore_dict[TYPE_KEY]
    config_dict: Dict[str, Any] = docstore_dict[DATA_KEY]

    # Inject kwargs into data dict.
    # This allows us to explicitly pass in unserializable objects
    # like the data storage (e.g. MongoDB) client.
    config_dict.update(kwargs)

    cls = type_to_cls[type]
    return cls.from_dict(config_dict)


def save_docstore_to_dict(
    docstore: DocumentStore,
    cls_to_type: Optional[Dict[Type[DocumentStore], DocumentStoreType]] = None,
) -> Dict[str, Any]:
    cls_to_type = cls_to_type or DOCSTORE_CLASS_TO_TYPE
    type_ = cls_to_type[type(docstore)]
    return {TYPE_KEY: type_, DATA_KEY: docstore.to_dict()}


def merge_docstores(docstores: Sequence[DocumentStore]) -> DocumentStore:
    if all(isinstance(docstore, SimpleDocumentStore) for docstore in docstores):
        merged_docstore = SimpleDocumentStore()
        for docstore in docstores:
            merged_docstore.update_docstore(docstore)
        return merged_docstore
    else:
        raise NotImplementedError()
