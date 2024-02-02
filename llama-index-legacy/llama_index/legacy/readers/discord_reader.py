"""Discord reader.

Note: this file is named discord_reader.py to avoid conflicts with the
discord.py module.

"""

import asyncio
import logging
import os
from typing import List, Optional

from llama_index.readers.base import BasePydanticReader
from llama_index.schema import Document

logger = logging.getLogger(__name__)


async def read_channel(
    discord_token: str,
    channel_id: int,
    limit: Optional[int],
    oldest_first: bool,
) -> List[Document]:
    """Async read channel.

    Note: This is our hack to create a synchronous interface to the
    async discord.py API. We use the `asyncio` module to run
    this function with `asyncio.get_event_loop().run_until_complete`.

    """
    import discord

    messages: List[discord.Message] = []

    class CustomClient(discord.Client):
        async def on_ready(self) -> None:
            try:
                logger.info(f"{self.user} has connected to Discord!")
                channel = client.get_channel(channel_id)
                # only work for text channels for now
                if not isinstance(channel, discord.TextChannel):
                    raise ValueError(
                        f"Channel {channel_id} is not a text channel. "
                        "Only text channels are supported for now."
                    )
                # thread_dict maps thread_id to thread
                thread_dict = {}
                for thread in channel.threads:
                    thread_dict[thread.id] = thread
                async for msg in channel.history(
                    limit=limit, oldest_first=oldest_first
                ):
                    messages.append(msg)
                    if msg.id in thread_dict:
                        thread = thread_dict[msg.id]
                        async for thread_msg in thread.history(
                            limit=limit, oldest_first=oldest_first
                        ):
                            messages.append(thread_msg)
            except Exception as e:
                logger.error("Encountered error: " + str(e))
            finally:
                await self.close()

    intents = discord.Intents.default()
    intents.message_content = True
    client = CustomClient(intents=intents)
    await client.start(discord_token)

    ### Wraps each message in a Document containing the text \
    # as well as some useful metadata properties.
    return [
        Document(
            text=msg.content,
            id_=msg.id,
            metadata={
                "message_id": msg.id,
                "username": msg.author.name,
                "created_at": msg.created_at,
                "edited_at": msg.edited_at,
            },
        )
        for msg in messages
    ]


class DiscordReader(BasePydanticReader):
    """Discord reader.

    Reads conversations from channels.

    Args:
        discord_token (Optional[str]): Discord token. If not provided, we
            assume the environment variable `DISCORD_TOKEN` is set.

    """

    is_remote: bool = True
    discord_token: str

    def __init__(self, discord_token: Optional[str] = None) -> None:
        """Initialize with parameters."""
        try:
            import discord  # noqa
        except ImportError:
            raise ImportError(
                "`discord.py` package not found, please run `pip install discord.py`"
            )
        if discord_token is None:
            discord_token = os.environ["DISCORD_TOKEN"]
            if discord_token is None:
                raise ValueError(
                    "Must specify `discord_token` or set environment "
                    "variable `DISCORD_TOKEN`."
                )

        super().__init__(discord_token=discord_token)

    @classmethod
    def class_name(cls) -> str:
        return "DiscordReader"

    def _read_channel(
        self, channel_id: int, limit: Optional[int] = None, oldest_first: bool = True
    ) -> List[Document]:
        """Read channel."""
        return asyncio.get_event_loop().run_until_complete(
            read_channel(
                self.discord_token, channel_id, limit=limit, oldest_first=oldest_first
            )
        )

    def load_data(
        self,
        channel_ids: List[int],
        limit: Optional[int] = None,
        oldest_first: bool = True,
    ) -> List[Document]:
        """Load data from the input directory.

        Args:
            channel_ids (List[int]): List of channel ids to read.
            limit (Optional[int]): Maximum number of messages to read.
            oldest_first (bool): Whether to read oldest messages first.
                Defaults to `True`.

        Returns:
            List[Document]: List of documents.

        """
        results: List[Document] = []
        for channel_id in channel_ids:
            if not isinstance(channel_id, int):
                raise ValueError(
                    f"Channel id {channel_id} must be an integer, "
                    f"not {type(channel_id)}."
                )
            channel_documents = self._read_channel(
                channel_id, limit=limit, oldest_first=oldest_first
            )
            results += channel_documents
        return results


if __name__ == "__main__":
    reader = DiscordReader()
    logger.info("initialized reader")
    output = reader.load_data(channel_ids=[1057178784895348746], limit=10)
    logger.info(output)
