from imap_tools import MailBox
from llama_index.core.readers.base import BaseReader
from typing import (Iterable, Optional, List, Dict, Any, Union)
from llama_index.core.schema import Document
from imap_tools import A, O, N, H, U, AND, OR, NOT, Header, UidRange

SearchCriteria = Union[A, O, N, H, U, AND, OR, NOT, Header, UidRange, None]

class ImapReader(BaseReader):
    """
    IMAP reader. Reads email from an IMAP server.
    """

    mailbox: MailBox

    def __init__(
            self,
            host: str,
            username: str,
            password: str
    ):
        self.mailbox = MailBox(host)
        self.mailbox.login(username, password)

    def lazy_load_data(
        self,
        folder: str = "INBOX",
        metadata_names: Optional[List[str]] = None,
        search_criteria: Optional[SearchCriteria] = None
    ) -> Iterable[Document]:
        if metadata_names is None:
            metadata_names = []
        metadata_names.append("date")

        criteria = search_criteria if search_criteria is not None else A(all=True)
        self.mailbox.folder.set(folder)

        for msg in self.mailbox.fetch(criteria=criteria):
            metadata: Dict[str, Any] = {}
            if metadata_names:
                metadata = { key: getattr(msg, key, None) for key in metadata_names }

            text = f"From: {msg.from_}, To: {msg.to[0]}, Subject: {msg.subject}, Message: {msg.text}"

            yield Document(text=text, metadata=metadata)

