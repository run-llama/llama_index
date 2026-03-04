"""Firebase Realtime Database Loader."""

from typing import Dict, List, Optional

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document


class FirebaseRealtimeDatabaseReader(BaseReader):
    """
    Firebase Realtime Database reader.

    Retrieves data from Firebase Realtime Database and converts it into the Document used by LlamaIndex.

    Args:
        database_url (str): Firebase Realtime Database URL.
        service_account_key_path (Optional[str]): Path to the service account key file.

    """

    def __init__(
        self,
        database_url: str,
        service_account_key_path: Optional[str] = None,
    ) -> None:
        """Initialize with parameters."""
        try:
            import firebase_admin
            from firebase_admin import credentials
        except ImportError:
            raise ImportError(
                "`firebase_admin` package not found, please run `pip install"
                " firebase-admin`"
            )

        if not firebase_admin._apps:
            if service_account_key_path:
                cred = credentials.Certificate(service_account_key_path)
                firebase_admin.initialize_app(
                    cred, options={"databaseURL": database_url}
                )
            else:
                firebase_admin.initialize_app(options={"databaseURL": database_url})

    def load_data(self, path: str, field: Optional[str] = None) -> List[Document]:
        """
        Load data from Firebase Realtime Database and convert it into documents.

        Args:
            path (str): Path to the data in the Firebase Realtime Database.
            field (str, Optional): Key to pick data from

        Returns:
            List[Document]: A list of documents.

        """
        try:
            from firebase_admin import db
        except ImportError:
            raise ImportError(
                "`firebase_admin` package not found, please run `pip install"
                " firebase-admin`"
            )

        ref = db.reference(path)
        data = ref.get()

        documents = []

        if isinstance(data, Dict):
            for key in data:
                entry = data[key]
                extra_info = {
                    "document_id": key,
                }
                if type(entry) is Dict and field in entry:
                    text = entry[field]
                else:
                    text = str(entry)

                document = Document(text=text, extra_info=extra_info)
                documents.append(document)
        elif isinstance(data, str):
            documents.append(Document(text=data))

        return documents
