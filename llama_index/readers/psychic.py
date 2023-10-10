"""Psychic reader."""
import logging
import os
from typing import List, Optional

from llama_index.readers.base import BaseReader
from llama_index.schema import Document

logger = logging.getLogger(__name__)


class PsychicReader(BaseReader):
    """Psychic reader.

    Psychic is a platform that allows syncing data from many SaaS apps through one
        universal API.
    This reader connects to an instance of Psychic and reads data from it, given a
        connector ID, account ID, and API key.

    Learn more at docs.psychic.dev.

    Args:
        psychic_key (str): Secret key for Psychic.
            Get one at https://dashboard.psychic.dev/api-keys.

    """

    def __init__(self, psychic_key: Optional[str] = None) -> None:
        """Initialize with parameters."""
        try:
            from psychicapi import ConnectorId, Psychic
        except ImportError:
            raise ImportError(
                "`psychicapi` package not found, please run `pip install psychicapi`"
            )
        if psychic_key is None:
            psychic_key = os.environ["PSYCHIC_SECRET_KEY"]
            if psychic_key is None:
                raise ValueError(
                    "Must specify `psychic_key` or set environment "
                    "variable `PSYCHIC_SECRET_KEY`."
                )

        self.psychic = Psychic(secret_key=psychic_key)
        self.ConnectorId = ConnectorId

    def load_data(
        self, connector_id: Optional[str] = None, account_id: Optional[str] = None
    ) -> List[Document]:
        """Load data from a Psychic connection.

        Args:
            connector_id (str): The connector ID to connect to
            account_id (str): The account ID to connect to

        Returns:
            List[Document]: List of documents.

        """
        if not connector_id or not account_id:
            raise ValueError("Must specify both `connector_id` and `account_id`.")
        if connector_id not in self.ConnectorId.__members__:
            raise ValueError("Invalid connector ID.")

        # get all the documents in the database
        docs = []
        data = self.psychic.get_documents(self.ConnectorId[connector_id], account_id)
        for resource in data:
            text = resource.get("content")
            doc_id = resource.get("uri")
            docs.append(
                Document(
                    text=text,
                    id_=doc_id,
                    metadata={"connector_id": connector_id, "account_id": account_id},
                )
            )

        return docs


if __name__ == "__main__":
    reader = PsychicReader(psychic_key="public_key")
    logger.info(reader.load_data(connector_id="connector_id", account_id="account_id"))
