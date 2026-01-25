from imap_tools import MailBox, MailAttachment
from llama_index.core.readers.base import BaseReader
from typing import Iterable, Optional, List, Dict, Any, Union, Callable
from llama_index.core.schema import Document
from imap_tools import A, O, N, H, U, AND, OR, NOT, Header, UidRange

SearchCriteria = Union[A, O, N, H, U, AND, OR, NOT, Header, UidRange, None]


class ImapReader(BaseReader):
    """
    IMAP reader. Reads email from an IMAP server.

    Args:
        host (str): IMAP server host
        username (str): email address
        password (str): email password

    """

    mailbox: MailBox

    def __init__(self, host: str, username: str, password: str):
        """Initialize IMAP connection"""
        self.mailbox = MailBox(host)
        self.mailbox.login(username, password)

    def lazy_load_data(
        self,
        folder: str = "INBOX",
        metadata_names: Optional[List[str]] = None,
        search_criteria: Optional[SearchCriteria] = None,
        save_attachment: Callable[[MailAttachment], str] = None,
    ) -> Iterable[Document]:
        """
        Fetch emails from the provided mailbox.

        Args:
            folder (str, optional): Folder where to look for emails. Defaults to "INBOX".
            metadata_names (List[str], optional): Names of metadata fields. Defaults to None. Full list at https://pypi.org/project/imap-tools/#email-attributes
            search_criteria (SearchCriteria, optional): Search criteria. Documentation at https://pypi.org/project/imap-tools/#search-criteria
            save_attachment (Callable[[MailAttachment], str], optional): Save attachments callback. Defaults to None. Must return the saved filename

        """
        if metadata_names is None:
            metadata_names = []
        # Always add "date" in metadata
        metadata_names.append("date")

        # If no criteria are set, all emails are taken into account
        criteria = search_criteria if search_criteria is not None else A(all=True)
        self.mailbox.folder.set(folder)

        for msg in self.mailbox.fetch(criteria=criteria):
            metadata: Dict[str, Any] = {}
            if metadata_names:
                metadata = {key: getattr(msg, key, None) for key in metadata_names}

            to_field = " ".join(msg.to) if msg.to else "(no recipient)"
            if "to" in metadata:
                metadata["to"] = to_field
            if "text" in metadata:
                # Renaming metadata because LlamaIndex uses text as field for node content in vector store
                metadata["email_text"] = metadata.pop("text")

            text = f"From: {msg.from_}, To: {to_field}, Subject: {msg.subject}, Message: {msg.text}"

            if save_attachment:
                metadata["attachments"] = []
                for attachment in msg.attachments:
                    filename = save_attachment(attachment)
                    metadata["attachments"].append(
                        {
                            "filename": filename,
                            "original_filename": attachment.filename,
                        }
                    )

            yield Document(text=text, metadata=metadata)
