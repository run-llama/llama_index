"""
Mbox parser.

Contains simple parser for mbox files.

"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from fsspec import AbstractFileSystem

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document

logger = logging.getLogger(__name__)


class MboxReader(BaseReader):
    """
    Mbox parser.

    Extract messages from mailbox files.
    Returns string including date, subject, sender, receiver and
    content for each message.

    """

    DEFAULT_MESSAGE_FORMAT: str = (
        "Date: {_date}\n"
        "From: {_from}\n"
        "To: {_to}\n"
        "Subject: {_subject}\n"
        "Content: {_content}"
    )

    def __init__(
        self,
        *args: Any,
        max_count: int = 0,
        message_format: str = DEFAULT_MESSAGE_FORMAT,
        **kwargs: Any,
    ) -> None:
        """Init params."""
        try:
            from bs4 import BeautifulSoup  # noqa
        except ImportError:
            raise ImportError(
                "`beautifulsoup4` package not found: `pip install beautifulsoup4`"
            )

        super().__init__(*args, **kwargs)
        self.max_count = max_count
        self.message_format = message_format

    def load_data(
        self,
        file: Path,
        extra_info: Optional[Dict] = None,
        fs: Optional[AbstractFileSystem] = None,
    ) -> List[Document]:
        """Parse file into string."""
        # Import required libraries
        import mailbox
        from email.parser import BytesParser
        from email.policy import default

        from bs4 import BeautifulSoup

        if fs:
            logger.warning(
                "fs was specified but MboxReader doesn't support loading "
                "from fsspec filesystems. Will load from local filesystem instead."
            )

        i = 0
        results: List[str] = []
        # Load file using mailbox
        bytes_parser = BytesParser(policy=default).parse
        mbox = mailbox.mbox(file, factory=bytes_parser)  # type: ignore

        # Iterate through all messages
        for _, _msg in enumerate(mbox):
            try:
                msg: mailbox.mboxMessage = _msg
                # The body is the plain text part, or else the HTML one, such
                # as the only text of an HTML message with an attachment.
                body = msg.get_body(preferencelist=("plain", "html"))
                content = "" if body is None else self._body_text(body)
                if body is not None and body.get_content_type() == "text/html":
                    content = BeautifulSoup(content, "html.parser").get_text()
                # Remove unneeded whitespace
                stripped_content = " ".join(content.split())
                # Format message to include date, sender, receiver and subject
                msg_string = self.message_format.format(
                    _date=msg["date"],
                    _from=msg["from"],
                    _to=msg["to"],
                    _subject=msg["subject"],
                    _content=stripped_content,
                )
                # Add message string to results
                results.append(msg_string)
            except Exception as e:
                logger.warning(f"Failed to parse message:\n{_msg}\n with exception {e}")

            # Increment counter and return if max count is met
            i += 1
            if self.max_count > 0 and i >= self.max_count:
                break

        return [Document(text=result, metadata=extra_info or {}) for result in results]

    @staticmethod
    def _body_text(body: Any) -> str:
        """Decode a body part, as UTF-8 when its charset is not one Python knows."""
        try:
            return body.get_content()
        except LookupError:
            return body.get_payload(decode=True).decode("utf-8", errors="replace")
