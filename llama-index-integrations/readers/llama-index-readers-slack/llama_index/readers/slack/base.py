"""Slack reader for fetching and processing messages from Slack channels."""

import logging
import os
from typing import Any, Dict, List, Optional
from llama_index.core.readers.base import BasePydanticReader
from llama_index.core.schema import Document

logger = logging.getLogger(__name__)


class SlackReader(BasePydanticReader):
    """Slack reader for processing channel messages and threads.

    Args:
        slack_token (Optional[str]): Slack bot token. If not provided, reads from SLACK_TOKEN env var.
    """

    is_remote: bool = True
    slack_token: str

    def __init__(self, slack_token: Optional[str] = None) -> None:
        """Initialize the Slack reader."""
        try:
            import slack_sdk  # noqa: F401
        except ImportError:
            raise ImportError(
                "Package `slack_sdk` not found. Please install with `pip install slack_sdk`"
            )

        token = slack_token or os.environ.get("SLACK_TOKEN")
        if not token:
            raise ValueError(
                "Slack token not found. Provide `slack_token` or set `SLACK_TOKEN` env var."
            )

        super().__init__(slack_token=token)

    def _process_message(self, message: Dict[str, Any], include_bots: bool) -> Optional[Document]:
        """Convert a Slack message to a Document.
        
        Args:
            message: Raw message from Slack API
            include_bots: Whether to include bot messages
            
        Returns:
            Document if message should be included, None otherwise
        """
        # Skip empty messages
        if message["text"] == "":
            return None

        # Skip bot messages if specified
        if not include_bots and "bot_id" in message:
            return None
        
        return Document(
            text=message["text"],
            id_=message["ts"],
            metadata={
                "message_id": message["ts"],
                "channel_id": message.get("channel", ""),
                "user_id": message["user"],
                "created_at": message["ts"],
                "edited_at": message.get("edited", {}).get("ts"),
            },
        )

    def _get_channel_messages(
        self, 
        client: Any, 
        channel_id: str, 
        limit: Optional[int],
    ) -> List[Dict[str, Any]]:
        """Fetch messages and their thread replies from a channel.
        
        Args:
            client: Slack WebClient instance
            channel_id: Channel to fetch messages from
            limit: Max number of messages to fetch
            
        Returns:
            List of messages including thread replies
        """
        from slack_sdk.errors import SlackApiError

        all_messages = []
        latest_ts = None
        remaining_limit = limit
        
        while True:
            try:
                # Calculate batch size (max 999 per request)
                batch_limit = 999 if limit is None else min(remaining_limit, 999)
                
                result = client.conversations_history(
                    channel=channel_id,
                    limit=batch_limit,
                    latest=latest_ts
                )
                
                messages = result["messages"]
                if not messages:
                    break

                for msg in messages:
                    # Add channel_id to message data
                    msg["channel"] = channel_id
                    
                    # Add non-thread messages immediately
                    if not msg.get("thread_ts") or msg["thread_ts"] == msg["ts"]:
                        all_messages.append(msg)
                    
                    # Batch fetch thread replies
                    if msg.get("reply_count", 0) > 0:
                        try:
                            thread = client.conversations_replies(
                                channel=channel_id,
                                ts=msg["ts"],
                            )
                            all_messages.extend(
                                reply for reply in thread["messages"]
                                if reply["ts"] != msg["ts"]
                            )
                        except SlackApiError as e:
                            logger.warning(f"Failed to fetch thread replies: {e}")

                latest_ts = messages[0]["ts"]
                
                # Only check remaining limit if we have a limit
                if limit is not None:
                    remaining_limit = limit - len(all_messages)
                    if remaining_limit <= 0:
                        return all_messages[:limit]
                
                if not result.get("has_more", False):
                    break
                    
            except SlackApiError as e:
                logger.error(f"Failed to fetch channel messages: {e}")
                break
        
        return all_messages

    def load_data(
        self,
        channel_ids: List[str],
        include_bots: bool = False,
        limit: Optional[int] = 100
    ) -> List[Document]:
        """Load messages from specified Slack channels.

        Args:
            channel_ids: List of channel IDs to fetch messages from
            include_bots: Whether to include bot messages (default True)
            limit: Max messages per channel (Default 100, None = all messages)

        Returns:
            List of Documents containing messages

        Raises:
            ValueError: If channel access fails
        """
        from slack_sdk import WebClient
        from slack_sdk.errors import SlackApiError

        client = WebClient(token=self.slack_token)
        documents = []

        for channel_id in channel_ids:
            if not isinstance(channel_id, str):
                raise ValueError(
                    f"Channel id {channel_id} must be a string, "
                    f"not {type(channel_id)}."
                )
            
            try:
                client.conversations_info(channel=channel_id)
            except SlackApiError as e:
                if "not_in_channel" in str(e):
                    raise ValueError(
                        f"Bot needs to be added to channel {channel_id}. "
                        "Use `/invite @BotName` in channel"
                    )
                elif "channel_not_found" in str(e):
                    raise ValueError(f"Channel {channel_id} not found")
                raise ValueError(f"Failed to access channel {channel_id}: {e}")

            # Fetch and process messages
            messages = self._get_channel_messages(client, channel_id, limit)
            documents.extend(
                doc for msg in messages
                if (doc := self._process_message(msg, include_bots)) is not None
            )

        return documents


if __name__ == "__main__":
    reader = SlackReader()
    logger.info("initialized reader")
    output = reader.load_data(channel_ids=["C08J3PZD5B2"], limit=10)
    logger.info(output)
