from typing import Optional

from llama_index.core.storage.docstore.keyval_docstore import KVDocumentStore
from llama_index.core.storage.docstore.types import DEFAULT_BATCH_SIZE
from llama_index.storage.kvstore.firestore import FirestoreKVStore


class FirestoreDocumentStore(KVDocumentStore):
    """
    Firestore Document (Node) store.

    A Firestore store for Document and Node objects.

    Args:
        firestore_kvstore (FirestoreKVStore): Firestore key-value store
        namespace (str): namespace for the docstore

    """

    def __init__(
        self,
        firestore_kvstore: FirestoreKVStore,
        namespace: Optional[str] = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        """Init a FirestoreDocumentStore."""
        super().__init__(firestore_kvstore, namespace=namespace, batch_size=batch_size)

    @classmethod
    def from_database(
        cls,
        project: str,
        database: str,
        namespace: Optional[str] = None,
    ) -> "FirestoreDocumentStore":
        """
        Args:
            project (str): The project which the client acts on behalf of.
            database (str): The database name that the client targets.
            namespace (str): namespace for the docstore.

        """
        firestore_kvstore = FirestoreKVStore(project=project, database=database)
        return cls(firestore_kvstore, namespace)
