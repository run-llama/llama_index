from enum import Enum
from typing import Any, Dict, Optional, Sequence, Type

from gpt_index.constants import DATA_KEY, TYPE_KEY
from gpt_index.storage.docstore.simple_docstore import SimpleDocumentStore
from gpt_index.storage.docstore.mongo_docstore import MongoDocumentStore
from gpt_index.storage.docstore.types import BaseDocumentStore


class DocumentStoreType(str, Enum):
    MONGO = "mongo"
    SIMPLE = "simple"


DOCSTORE_TYPE_TO_CLASS: Dict[DocumentStoreType, Type[BaseDocumentStore]] = {
    DocumentStoreType.MONGO: MongoDocumentStore,
    DocumentStoreType.SIMPLE: SimpleDocumentStore,
}


DOCSTORE_CLASS_TO_TYPE: Dict[Type[BaseDocumentStore], DocumentStoreType] = {
    cls_: type_ for type_, cls_ in DOCSTORE_TYPE_TO_CLASS.items()
}


def get_default_docstore() -> BaseDocumentStore:
    return SimpleDocumentStore()


def load_docstore_from_dict(
    docstore_dict: Dict[str, Any],
    type_to_cls: Optional[Dict[DocumentStoreType, Type[BaseDocumentStore]]] = None,
    **kwargs: Any,
) -> BaseDocumentStore:
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
    docstore: BaseDocumentStore,
    cls_to_type: Optional[Dict[Type[BaseDocumentStore], DocumentStoreType]] = None,
) -> Dict[str, Any]:
    cls_to_type = cls_to_type or DOCSTORE_CLASS_TO_TYPE
    type_ = cls_to_type[type(docstore)]
    return {TYPE_KEY: type_, DATA_KEY: docstore.to_dict()}
