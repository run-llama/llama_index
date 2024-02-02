from typing import Optional

from llama_index.storage.index_store.keyval_index_store import KVIndexStore
from llama_index.storage.kvstore.firestore_kvstore import FirestoreKVStore


class FirestoreIndexStore(KVIndexStore):
    """Firestore Index store.

    Args:
        firestore_kvstore (FirestoreKVStore): Firestore key-value store
        namespace (str): namespace for the index store

    """

    def __init__(
        self,
        firestore_kvstore: FirestoreKVStore,
        namespace: Optional[str] = None,
    ) -> None:
        """Init a FirestoreIndexStore."""
        super().__init__(firestore_kvstore, namespace=namespace)

    @classmethod
    def from_database(
        cls,
        project: str,
        database: str,
        namespace: Optional[str] = None,
    ) -> "FirestoreIndexStore":
        """
        Args:
            project (str): The project which the client acts on behalf of.
            database (str): The database name that the client targets.
            namespace (str): namespace for the docstore.
        """
        firestore_kvstore = FirestoreKVStore(project=project, database=database)
        return cls(firestore_kvstore, namespace)
