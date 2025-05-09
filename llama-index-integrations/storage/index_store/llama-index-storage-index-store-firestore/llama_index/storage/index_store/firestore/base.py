from typing import Optional

from llama_index.core.storage.index_store.keyval_index_store import KVIndexStore
from llama_index.storage.kvstore.firestore import FirestoreKVStore


class FirestoreIndexStore(KVIndexStore):
    """
    Firestore Index store.

    Args:
        firestore_kvstore (FirestoreKVStore): Firestore key-value store
        namespace (str): namespace for the index store

    """

    def __init__(
        self,
        firestore_kvstore: FirestoreKVStore,
        namespace: Optional[str] = None,
        collection_suffix: Optional[str] = None,
    ) -> None:
        """Init a FirestoreIndexStore."""
        super().__init__(
            firestore_kvstore, namespace=namespace, collection_suffix=collection_suffix
        )

    @classmethod
    def from_database(
        cls,
        project: str,
        database: str,
        namespace: Optional[str] = None,
        collection_suffix: Optional[str] = None,
    ) -> "FirestoreIndexStore":
        """
        Load a FirestoreIndexStore from a Firestore database.

        Args:
            project (str): The project which the client acts on behalf of.
            database (str): The database name that the client targets.
            namespace (str): namespace for the docstore.
            collection_suffix (str): suffix for the collection name

        """
        firestore_kvstore = FirestoreKVStore(project=project, database=database)
        return cls(firestore_kvstore, namespace, collection_suffix)
