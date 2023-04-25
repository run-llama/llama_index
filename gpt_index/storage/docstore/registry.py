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
