"""Slack reader."""

import logging
import os
import re
import time
from datetime import datetime
from ssl import SSLContext
from typing import Any, List, Optional, Dict

from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.readers.base import BasePydanticReader
from llama_index.core.schema import Document


logger = logging.getLogger(__name__)


class SlackReader(BasePydanticReader):
    """
    Slack reader.

    Reads conversations from channels. If an earliest_date is provided, an
    optional latest_date can also be provided. If no latest_date is provided,
    we assume the latest date is the current timestamp.

    Args:
        slack_token (Optional[str]): Slack token. If not provided, we
            assume the environment variable `SLACK_BOT_TOKEN` is set.
        ssl (Optional[str]): Custom SSL context. If not provided, it is assumed
            there is already an SSL context available.
        earliest_date (Optional[datetime]): Earliest date from which
            to read conversations. If not provided, we read all messages.
        latest_date (Optional[datetime]): Latest date from which to
            read conversations. If not provided, defaults to current timestamp
            in combination with earliest_date.

    """

    is_remote: bool = True
    slack_token: str
    earliest_date_timestamp: Optional[float]
    latest_date_timestamp: float
    channel_types: str

    _client: Any = PrivateAttr()

    def __init__(
        self,
        slack_token: Optional[str] = None,
        ssl: Optional[SSLContext] = None,
        earliest_date: Optional[datetime] = None,
        latest_date: Optional[datetime] = None,
        earliest_date_timestamp: Optional[float] = None,
        latest_date_timestamp: Optional[float] = None,
        channel_types: Optional[str] = None,
    ) -> None:
        """Initialize with parameters."""
        from slack_sdk import WebClient

        if slack_token is None:
            slack_token = os.environ["SLACK_BOT_TOKEN"]
        if slack_token is None:
            raise ValueError(
                "Must specify `slack_token` or set environment "
                "variable `SLACK_BOT_TOKEN`."
            )
        if ssl is None:
            client = WebClient(token=slack_token)
        else:
            client = WebClient(token=slack_token, ssl=ssl)
        if latest_date is not None and earliest_date is None:
            raise ValueError(
                "Must specify `earliest_date` if `latest_date` is specified."
            )
        if earliest_date is not None:
            earliest_date_timestamp = earliest_date.timestamp()
        else:
            earliest_date_timestamp = None or earliest_date_timestamp
        if latest_date is not None:
            latest_date_timestamp = latest_date.timestamp()
        else:
            latest_date_timestamp = datetime.now().timestamp() or latest_date_timestamp
        if channel_types is not None:
            channel_types = channel_types
        else:
            channel_types = "public_channel,private_channel"
        res = client.api_test()
        if not res["ok"]:
            raise ValueError(f"Error initializing Slack API: {res['error']}")

        super().__init__(
            slack_token=slack_token,
            earliest_date_timestamp=earliest_date_timestamp,
            latest_date_timestamp=latest_date_timestamp,
            channel_types=channel_types,
        )
        self._client = client

    @classmethod
    def class_name(cls) -> str:
        return "SlackReader"

    def _read_message(self, channel_id: str, message_ts: str) -> str:
        from slack_sdk.errors import SlackApiError

        """Read a message."""

        messages_text: List[str] = []
        next_cursor = None
        while True:
            try:
                # https://slack.com/api/conversations.replies
                # List all replies to a message, including the message itself.
                if self.earliest_date_timestamp is None:
                    result = self._client.conversations_replies(
                        channel=channel_id, ts=message_ts, cursor=next_cursor
                    )
                else:
                    conversations_replies_kwargs = {
                        "channel": channel_id,
                        "ts": message_ts,
                        "cursor": next_cursor,
                        "latest": str(self.latest_date_timestamp),
                    }
                    if self.earliest_date_timestamp is not None:
                        conversations_replies_kwargs["oldest"] = str(
                            self.earliest_date_timestamp
                        )
                    result = self._client.conversations_replies(
                        **conversations_replies_kwargs  # type: ignore
                    )
                messages = result["messages"]
                messages_text.extend(message["text"] for message in messages)
                if not result["has_more"]:
                    break

                next_cursor = result["response_metadata"]["next_cursor"]
            except SlackApiError as e:
                error = e.response["error"]
                if error == "ratelimited":
                    retry_after = int(e.response.headers.get("retry-after", 1))
                    logger.error(
                        f"Rate limit error reached, sleeping for: {retry_after} seconds"
                    )
                    time.sleep(retry_after)
                elif error == "not_in_channel":
                    logger.error(
                        f"Error: Bot not in channel: {channel_id}, cannot read messages."
                    )
                    break
                else:
                    logger.error(
                        f"Error parsing conversation replies for channel {channel_id}: {e}"
                    )
                    break

        return "\n\n".join(messages_text)

    def _read_channel(self, channel_id: str, reverse_chronological: bool) -> str:
        from slack_sdk.errors import SlackApiError

        """Read a channel."""

        result_messages: List[str] = []
        next_cursor = None
        while True:
            try:
                # Call the conversations.history method using the WebClient
                # conversations.history returns the first 100 messages by default
                # These results are paginated,
                # see: https://api.slack.com/methods/conversations.history$pagination
                conversations_history_kwargs = {
                    "channel": channel_id,
                    "cursor": next_cursor,
                    "latest": str(self.latest_date_timestamp),
                }
                if self.earliest_date_timestamp is not None:
                    conversations_history_kwargs["oldest"] = str(
                        self.earliest_date_timestamp
                    )
                result = self._client.conversations_history(
                    **conversations_history_kwargs  # type: ignore
                )
                conversation_history = result["messages"]
                # Print results
                logger.info(
                    f"{len(conversation_history)} messages found in {channel_id}"
                )
                result_messages.extend(
                    self._read_message(channel_id, message["ts"])
                    for message in conversation_history
                )
                if not result["has_more"]:
                    break
                next_cursor = result["response_metadata"]["next_cursor"]

            except SlackApiError as e:
                error = e.response["error"]
                if error == "ratelimited":
                    retry_after = int(e.response.headers.get("retry-after", 1))
                    logger.error(
                        f"Rate limit error reached, sleeping for: {retry_after} seconds"
                    )
                    time.sleep(retry_after)
                elif error == "not_in_channel":
                    logger.error(
                        f"Error: Bot not in channel: {channel_id}, cannot read messages."
                    )
                    break
                else:
                    logger.error(
                        f"Error parsing conversation replies for channel {channel_id}: {e}"
                    )
                    break

        return (
            "\n\n".join(result_messages)
            if reverse_chronological
            else "\n\n".join(result_messages[::-1])
        )

    def load_data(
        self, channel_ids: List[str], reverse_chronological: bool = True
    ) -> List[Document]:
        """
        Load data from the input slack channel ids.

        Args:
            channel_ids (List[str]): List of channel ids to read.

        Returns:
            List[Document]: List of documents.

        """
        results = []
        for channel_id in channel_ids:
            channel_content = self._read_channel(
                channel_id, reverse_chronological=reverse_chronological
            )
            results.append(
                Document(
                    id_=channel_id,
                    text=channel_content,
                    metadata={"channel": channel_id},
                )
            )
        return results

    def _is_regex(self, pattern: str) -> bool:
        """Check if a string is a regex pattern."""
        try:
            re.compile(pattern)
            return True
        except re.error:
            return False

    def _list_channels(self) -> List[Dict[str, Any]]:
        """List channels based on the types."""
        from slack_sdk.errors import SlackApiError

        try:
            result = self._client.conversations_list(types=self.channel_types)
            return result["channels"]
        except SlackApiError as e:
            logger.error(f"Error fetching channels: {e.response['error']}")
            raise

    def _filter_channels(
        self, channels: List[Dict[str, Any]], patterns: List[str]
    ) -> List[Dict[str, Any]]:
        """Filter channels based on the provided names and regex patterns."""
        regex_patterns = [pattern for pattern in patterns if self._is_regex(pattern)]
        exact_names = [pattern for pattern in patterns if not self._is_regex(pattern)]

        # Match Exact Channel names
        filtered_channels = [
            channel for channel in channels if channel["name"] in exact_names
        ]

        # Match Regex Patterns
        for channel in channels:
            for pattern in regex_patterns:
                if re.match(pattern, channel["name"]):
                    filtered_channels.append(channel)
        return filtered_channels

    def get_channel_ids(self, channel_patterns: List[str]) -> List[str]:
        """
        Get list of channel IDs based on names and regex patterns.

        Args:
            channel_patterns List[str]: List of channel name patterns (names or regex) to read.

        Returns:
            List[Document]: List of documents.

        """
        if not channel_patterns:
            raise ValueError("No channel patterns provided.")

        channels = self._list_channels()
        logger.info(f"Total channels fetched: {len(channels)}")

        if not channels:
            logger.info("No channels found in Slack.")
            return []

        filtered_channels = self._filter_channels(
            channels=channels, patterns=channel_patterns
        )
        logger.info(f"Channels matching patterns: {len(filtered_channels)}")

        if not filtered_channels:
            logger.info(
                "None of the channel names or pattern matched with Slack Channels."
            )
            return []

        channel_ids = [channel["id"] for channel in filtered_channels]

        return list(set(channel_ids))


if __name__ == "__main__":
    reader = SlackReader()

    # load data using only channel ids
    logger.info(reader.load_data(channel_ids=["C079KD1M8J3", "C078YQP5B51"]))

    # load data using exact channel names and regex patterns
    # get the channel ids first
    channel_ids = reader.get_channel_ids(
        channel_patterns=["^dev.*", "^qa.*", "test_channel"]
    )
    # load data using above channel_ids
    logger.info(reader.load_data(channel_ids=channel_ids))
