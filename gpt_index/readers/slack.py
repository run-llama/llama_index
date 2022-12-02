"""Slack reader."""
from typing import Any, List

from gpt_index.readers.base import BaseReader
import os
from gpt_index.schema import Document
import logging

logger = logging.getLogger(__name__)


class SlackReader(BaseReader):
    """Slack reader.

    Reads conversations from channels.

    """

    def __init__(self) -> None:
        """Initialize with parameters."""
        try:
            from slack_sdk import WebClient
        except ImportError:
            raise ValueError(
                "`slack_sdk` package not found, please run `pip install slack_sdk`"
            )
        self.client = WebClient(token=os.environ["SLACK_BOT_TOKEN"])
        res = self.client.api_test()


    def _read_channel(self, channel_id: str) -> str:
        """Read a channel."""
        from slack_sdk.errors import SlackApiError

        result_messages = []
        done = False
        next_cursor = None
        while not done:
            result = self.client.conversations_history(channel=channel_id, cursor=next_cursor)
            try:
                # Call the conversations.history method using the WebClient
                # conversations.history returns the first 100 messages by default
                # These results are paginated, see: https://api.slack.com/methods/conversations.history$pagination
                result = self.client.conversations_history(channel=channel_id)
                conversation_history = result["messages"]
                # Print results
                logger.info("{} messages found in {}".format(len(conversation_history), id))
                for message in conversation_history:
                    result_messages.append(message)

                if result["has_more"]:
                    next_cursor = result["response_metadata"]["next_cursor"]
                else:
                    done = True
                    break

            except SlackApiError as e:
                logger.error("Error creating conversation: {}".format(e))

        return result_messages

    def load_data(self, **load_kwargs: Any) -> List[Document]:
        """Load data from the input directory."""
        channels = load_kwargs.pop("channels", None)
        if channels is None:
            raise ValueError('Must specify a "channels" in `load_kwargs`.')
        
        results = []
        for channel in channels:
            channel_content = self._read_channel(channel)
            results.append(Document(channel_content, extra_info={"channel": channel}))
        return results


if __name__ == "__main__":
    reader = SlackReader()
    print(reader.load_data(channels=["C04DC2VUY3F"]))