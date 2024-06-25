"""Google Chat Reader."""

import logging
from datetime import datetime
from typing import Any, List, Dict

from llama_index.core.readers.base import BasePydanticReader
from llama_index.core.schema import Document

logger = logging.getLogger(__name__)

SCOPES = [
    "https://www.googleapis.com/auth/chat.messages.readonly",
    "https://www.googleapis.com/auth/userinfo.profile",
]


class GoogleChatReader(BasePydanticReader):
    """Google Chat Reader.

    Reads messages from Google Chat
    """

    is_remote: bool = True

    @classmethod
    def class_name(cls) -> str:
        """Gets name identifier of class."""
        return "GoogleChatReader"

    def load_data(
        self,
        space_names: List[str],
        num_messages: int = -1,
        after: datetime = None,
        before: datetime = None,
        order_asc: bool = True,
    ) -> List[Document]:
        """Loads documents from Google Chat.

        Args:
            space_name (List[str]): List of Space ID names found at top of URL (without the "space/").
            num_messages (int, optional): Number of messages to load (may exceed this number). If -1, then loads all messages. Defaults to -1.
            after (datetime, optional): Only search for messages after this datetime (UTC). Defaults to None.
            before (datetime, optional): Only search for messages before this datetime (UTC). Defaults to None.
            order_asc (bool, optional): If messages should be ordered by ascending time order. Defaults to True.

        Returns:
            List[Document]: List of document objects
        """
        from googleapiclient.discovery import build

        # get credentials and create chat service
        credentials = self._get_credentials()
        service = build("chat", "v1", credentials=credentials)

        logger.info("Credentials successfully obtained.")

        res = []
        for space_name in space_names:
            all_msgs = self._get_msgs(
                service, space_name, num_messages, after, before, order_asc
            )
            msgs_sorted = self._sort_msgs(space_name, all_msgs, order_asc)
            res.extend(msgs_sorted)
            logger.info(f"Successfully retrieved messages from {space_name}")

        return res

    def _sort_msgs(
        self, space_name: str, all_msgs: List[Dict[str, Any]], order_asc: bool
    ) -> Document:
        """Sorts messages from space and puts them into Document.

        Args:
            space_name (str): Space ID
            all_msgs (List[Dict[str, Any]]): All messages
            order_asc (bool): If ordered by ascending order

        Returns:
            Document: Document with messages
        """
        msgs_reorder = self._reorder_threads(all_msgs, order_asc)

        res = []
        id_to_text = self._id_to_text(all_msgs)
        for msg in msgs_reorder:
            if any(
                i not in msg for i in ("name", "text", "thread", "sender", "createTime")
            ):
                continue

            if "name" not in msg["thread"] or "name" not in msg["sender"]:
                continue

            metadata = {
                "space_id": space_name,
                "thread_id": msg["thread"]["name"],
                "sender_id": msg["sender"]["name"],
                "timestamp": msg["createTime"],
            }

            if "quotedMessageMetadata" in msg:
                metadata["quoted_msg"] = id_to_text[
                    msg["quotedMessageMetadata"]["name"]
                ]

            doc = Document(id_=msg["name"], text=msg["text"], metadata=metadata)
            res.append(doc)

        return res

    def _id_to_text(self, all_msgs: List[Dict[str, Any]]) -> Dict[str, str]:
        """Maps message ID to text, used for quote replies.

        Args:
            all_msgs (List[Dict[str, Any]]): All messages

        Returns:
            Dict[str, str]: Map message ID -> message text
        """
        res = {}

        for msg in all_msgs:
            if "text" not in msg or "name" not in msg:
                continue

            res[msg["name"]] = msg["text"]

        return res

    def _reorder_threads(
        self, all_msgs: List[Dict[str, Any]], order_asc: bool
    ) -> List[Dict[str, Any]]:
        """Reorders threads in message list.

        Args:
            all_msgs (List[Dict[str, Any]]): All messages
            order_asc (bool): If ordered by ascending order

        Returns:
            List[Dict[str, Any]]: Messages, but with threads reordered to be underneath their parent
        """
        if not order_asc:
            all_msgs = all_msgs[::-1]

        # maps thread ID -> list of message objects associated with that ID
        threads_dict = {}
        for msg in all_msgs:
            thread_name = msg["thread"]["name"]
            if thread_name not in threads_dict:
                threads_dict[thread_name] = [msg]
            else:
                threads_dict[thread_name].append(msg)

        # loop through original messages
        # extend res list with a list of messages that have the same thread ID
        # only works if messages are in ascending time order
        res = []
        for msg in all_msgs:
            if "threadReply" in msg:
                continue

            thread_name = msg["thread"]["name"]
            res.extend(threads_dict[thread_name])

        if not order_asc:
            res = res[::-1]

        return res

    def _get_msgs(
        self,
        service: Any,
        space_name: str,
        num_messages: int = -1,
        after: datetime = None,
        before: datetime = None,
        order_asc: bool = True,
    ) -> List[Dict[str, Any]]:
        """Puts raw API output of chat messages from one space into a list.

        Args:
            service (Any): Google Chat API service object
            space_name (str): Space ID name found at top of URL (without the "space/").
            num_messages (int, optional): Number of messages to load (may exceed this number). If -1, then loads all messages. Defaults to -1.
            after (datetime, optional): Only search for messages after this datetime (UTC). Defaults to None.
            before (datetime, optional): Only search for messages before this datetime (UTC). Defaults to None.
            order_asc (bool, optional): If messages should be ordered by ascending time order. Defaults to True.

        Returns:
            List[Dict[str, Any]]: List of message objects
        """
        all_msgs = []

        # API parameters
        parent = f"spaces/{space_name}"
        page_token = ""
        filter_str = ""
        if after is not None:
            offset_str = ""
            if after.utcoffset() is None:
                offset_str = "+00:00"
            filter_str += f'createTime > "{after.isoformat("T") + offset_str}" AND '
        if before is not None:
            offset_str = ""
            if before.utcoffset() is None:
                offset_str = "+00:00"
            filter_str += f'createTime < "{before.isoformat("T") + offset_str}" AND '
        filter_str = filter_str[:-4]
        order_by = f"createTime {'ASC' if order_asc else 'DESC'}"

        # Get all messages from space
        while num_messages == -1 or len(all_msgs) < num_messages:
            result = (
                service.spaces()
                .messages()
                .list(
                    parent=parent,
                    pageSize=num_messages if num_messages != -1 else 1000,
                    pageToken=page_token,
                    filter=filter_str,
                    orderBy=order_by,
                    showDeleted=False,
                )
                .execute()
            )

            if not result or "messages" not in result:
                break

            all_msgs.extend(result["messages"])

            # if no more messages to load
            if "nextPageToken" not in result:
                break

            page_token = result["nextPageToken"]

        return all_msgs

    def _get_credentials(self) -> Any:
        """Get valid user credentials from storage.

        The file token.json stores the user's access and refresh tokens, and is
        created automatically when the authorization flow completes for the first
        time.

        Returns:
            Credentials, the obtained credential.
        """
        import os

        from google_auth_oauthlib.flow import InstalledAppFlow
        from google.auth.transport.requests import Request

        from google.oauth2.credentials import Credentials

        creds = None
        if os.path.exists("token.json"):
            creds = Credentials.from_authorized_user_file("token.json", SCOPES)
        # If there are no (valid) credentials available, let the user log in.
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    "credentials.json", SCOPES
                )
                creds = flow.run_local_server(port=0)
            # Save the credentials for the next run
            with open("token.json", "w") as token:
                token.write(creds.to_json())

        return creds
