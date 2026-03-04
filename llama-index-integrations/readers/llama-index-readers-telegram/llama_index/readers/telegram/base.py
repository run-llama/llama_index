"""Telegram reader that reads posts/chats and comments to post from Telegram channel or chat."""

import asyncio
import re
from typing import List, Optional
import datetime

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document


class TelegramReader(BaseReader):
    """
    Telegram posts/chat messages/comments reader.

    Read posts/chat messages/comments from Telegram channels or chats.

    Before working with Telegram’s API, you need to get your own API ID and hash:

        1. Login to your Telegram account with the phone number of the developer account to use.
        2. Click under API Development tools.
        3. A Create new application window will appear. Fill in your application details.\
            There is no need to enter any URL,\
            and only the first two fields (App title and Short name) can currently be changed later.
        4. Click on Create application at the end.\
            Remember that your API hash is secret and Telegram won’t let you revoke it.\
            Don’t post it anywhere!

    This API ID and hash is the one used by your application, not your phone number.\
        You can use this API ID and hash with any phone number.

    Args:
        session_name (str): The file name of the session file to be used\
            if a string is given (it may be a full path),\
            or the Session instance to be used otherwise.
        api_id (int): The API ID you obtained from https://my.telegram.org.
        api_hash (str): The API hash you obtained from https://my.telegram.org.
        phone_number (str): The phone to which the code will be sent.

    """

    def __init__(
        self,
        session_name: str,
        api_id: int,
        api_hash: str,
        phone_number: str,
    ) -> None:
        """Initialize with parameters."""
        super().__init__()
        self.session_name = session_name
        self.api_id = api_id
        self.api_hash = api_hash
        self.phone_number = phone_number
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def load_data(
        self,
        entity_name: str,
        post_id: Optional[int] = None,
        limit: Optional[int] = None,
        start_date: Optional[datetime.datetime] = None,
        end_date: Optional[datetime.datetime] = None,
    ) -> List[Document]:
        """
        Load posts/chat messages/comments from Telegram channels or chats.

        Since Telethon is an asynchronous library,\
            you need to await coroutine functions to have them run\
            (or otherwise, run the loop until they are complete)

        Args:
            entity_name (str): The entity from whom to retrieve the message history.
            post_id (int): If set to a post ID, \
                the comments that reply to this ID will be returned.\
                Else will get posts/chat messages.
            limit (int): Number of messages to be retrieved.
            start_date (datetime.datetime): Start date of the time period.
            end_date (datetime.datetime): End date of the time period.

        """
        return self.loop.run_until_complete(
            self._load_data(
                entity_name=entity_name,
                post_id=post_id,
                limit=limit,
                start_date=start_date,
                end_date=end_date,
            )
        )

    async def _load_data(
        self,
        entity_name: str,
        post_id: Optional[int] = None,
        limit: Optional[int] = None,
        start_date: Optional[datetime.datetime] = None,
        end_date: Optional[datetime.datetime] = None,
    ) -> List[Document]:
        """
        Load posts/chat messages/comments from Telegram channels or chats.

        Args:
            entity_name (str): The entity from whom to retrieve the message history.
            post_id (int): If set to a post ID, \
                the comments that reply to this ID will be returned.\
                Else will get posts/chat messages.
            limit (int): Number of messages to be retrieved.
            start_date (datetime.datetime): Start date of the time period.
            end_date (datetime.datetime): End date of the time period.

        """
        import telethon

        client = telethon.TelegramClient(self.session_name, self.api_id, self.api_hash)
        await client.start(phone=self.phone_number)

        results = []
        async with client:
            if end_date and start_date:
                # Asynchronously iterate over messages in between start_date and end_date
                async for message in client.iter_messages(
                    entity_name,
                    reply_to=post_id,
                    limit=limit,
                    offset_date=end_date,
                ):
                    if message.date < start_date:
                        break
                    if isinstance(message.text, str) and message.text != "":
                        results.append(Document(text=self._remove_links(message.text)))
            else:
                # Asynchronously iterate over messages
                async for message in client.iter_messages(
                    entity_name,
                    reply_to=post_id,
                    limit=limit,
                ):
                    if isinstance(message.text, str) and message.text != "":
                        results.append(Document(text=self._remove_links(message.text)))
        return results

    def _remove_links(self, string) -> str:
        """Removes all URLs from a given string, leaving only the base domain name."""

        def replace_match(match):
            text = match.group(1)
            return text if text else ""

        url_pattern = r"https?://(?:www\.)?((?!www\.).)+?"
        return re.sub(url_pattern, replace_match, string)
