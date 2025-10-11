"""GMail tool spec."""

import base64
import email
from email.message import EmailMessage
from typing import Any, List, Optional

from llama_index.core.schema import Document
from llama_index.core.tools.tool_spec.base import BaseToolSpec

SCOPES = [
    "https://www.googleapis.com/auth/gmail.compose",
    "https://www.googleapis.com/auth/gmail.readonly",
]


class GmailToolSpec(BaseToolSpec):
    """
    GMail tool spec.

    Gives the agent the ability to read, draft and send gmail messages

    """

    spec_functions = [
        "load_data",
        "search_messages",
        "create_draft",
        "update_draft",
        "get_draft",
        "send_draft",
    ]
    query: str = None
    use_iterative_parser: bool = False
    max_results: int = 10
    service: Any = None

    def _cache_service(self) -> None:
        from googleapiclient.discovery import build

        credentials = self._get_credentials()
        if not self.service:
            self.service = build("gmail", "v1", credentials=credentials)

    def load_data(self) -> List[Document]:
        """Load emails from the user's account."""
        self._cache_service()

        return self.search_messages(query="")

    def _get_credentials(self) -> Any:
        """
        Get valid user credentials from storage.

        The file token.json stores the user's access and refresh tokens, and is
        created automatically when the authorization flow completes for the first
        time.

        Returns:
            Credentials, the obtained credential.

        """
        import os

        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow

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
                creds = flow.run_local_server(port=8080)
            # Save the credentials for the next run
            with open("token.json", "w") as token:
                token.write(creds.to_json())

        return creds

    def search_messages(self, query: str, max_results: Optional[int] = None):
        """
        Searches email messages given a query string and the maximum number
        of results requested by the user
           Returns: List of relevant message objects up to the maximum number of results.

        Args:
            query (str): The user's query
            max_results (Optional[int]): The maximum number of search results
            to return.

        """
        if not max_results:
            max_results = self.max_results

        self._cache_service()

        messages = (
            self.service.users()
            .messages()
            .list(userId="me", q=query or None, maxResults=int(max_results))
            .execute()
            .get("messages", [])
        )

        results = []
        try:
            for message in messages:
                message_data = self.get_message_data(message)
                text = message_data.pop("body")
                metadata = message_data
                results.append(Document(text=text, metadata=metadata))
        except Exception as e:
            raise Exception("Can't get message data" + str(e))

        return results

    def get_message_data(self, message):
        message_id = message["id"]
        message_data = (
            self.service.users()
            .messages()
            .get(format="raw", userId="me", id=message_id)
            .execute()
        )
        if self.use_iterative_parser:
            body = self.extract_message_body_iterative(message_data)
        else:
            body = self.extract_message_body(message_data)

        if not body:
            return None

        return {
            "id": message_data["id"],
            "threadId": message_data["threadId"],
            "snippet": message_data["snippet"],
            "body": body,
        }

    def extract_message_body_iterative(self, message: dict):
        if message["raw"]:
            body = base64.urlsafe_b64decode(message["raw"].encode("utf8"))
            mime_msg = email.message_from_bytes(body)
        else:
            mime_msg = message

        body_text = ""
        if mime_msg.get_content_type() == "text/plain":
            plain_text = mime_msg.get_payload(decode=True)
            charset = mime_msg.get_content_charset("utf-8")
            body_text = plain_text.decode(charset).encode("utf-8").decode("utf-8")

        elif mime_msg.get_content_maintype() == "multipart":
            msg_parts = mime_msg.get_payload()
            for msg_part in msg_parts:
                body_text += self.extract_message_body_iterative(msg_part)

        return body_text

    def extract_message_body(self, message: dict):
        from bs4 import BeautifulSoup

        try:
            body = base64.urlsafe_b64decode(message["raw"].encode("utf-8"))
            mime_msg = email.message_from_bytes(body)

            # If the message body contains HTML, parse it with BeautifulSoup
            if "text/html" in mime_msg:
                soup = BeautifulSoup(body, "html.parser")
                body = soup.get_text()
            return body.decode("utf-8")
        except Exception as e:
            raise Exception("Can't parse message body" + str(e))

    def _build_draft(
        self,
        to: Optional[List[str]] = None,
        subject: Optional[str] = None,
        message: Optional[str] = None,
    ) -> str:
        email_message = EmailMessage()

        email_message.set_content(message)

        email_message["To"] = to
        email_message["Subject"] = subject

        encoded_message = base64.urlsafe_b64encode(email_message.as_bytes()).decode()

        return {"message": {"raw": encoded_message}}

    def create_draft(
        self,
        to: Optional[List[str]] = None,
        subject: Optional[str] = None,
        message: Optional[str] = None,
    ) -> str:
        """
        Create and insert a draft email.
           Print the returned draft's message and id.
           Returns: Draft object, including draft id and message meta data.

        Args:
            to (Optional[str]): The email addresses to send the message to
            subject (Optional[str]): The subject for the event
            message (Optional[str]): The message for the event

        """
        self._cache_service()
        service = self.service

        return (
            service.users()
            .drafts()
            .create(userId="me", body=self._build_draft(to, subject, message))
            .execute()
        )

    def update_draft(
        self,
        to: Optional[List[str]] = None,
        subject: Optional[str] = None,
        message: Optional[str] = None,
        draft_id: str = None,
    ) -> str:
        """
        Update a draft email.
           Print the returned draft's message and id.
           This function is required to be passed a draft_id that is obtained when creating messages
           Returns: Draft object, including draft id and message meta data.

        Args:
            to (Optional[str]): The email addresses to send the message to
            subject (Optional[str]): The subject for the event
            message (Optional[str]): The message for the event
            draft_id (str): the id of the draft to be updated

        """
        self._cache_service()
        service = self.service

        if draft_id is None:
            return (
                "You did not provide a draft id when calling this function. If you"
                " previously created or retrieved the draft, the id is available in"
                " context"
            )

        draft = self.get_draft(draft_id)
        headers = draft["message"]["payload"]["headers"]
        for header in headers:
            if header["name"] == "To" and not to:
                to = header["value"]
            elif header["name"] == "Subject" and not subject:
                subject = header["value"]

        return (
            service.users()
            .drafts()
            .update(
                userId="me", id=draft_id, body=self._build_draft(to, subject, message)
            )
            .execute()
        )

    def get_draft(self, draft_id: str = None) -> str:
        """
        Get a draft email.
           Print the returned draft's message and id.
           Returns: Draft object, including draft id and message meta data.

        Args:
            draft_id (str): the id of the draft to be updated

        """
        self._cache_service()
        service = self.service
        return service.users().drafts().get(userId="me", id=draft_id).execute()

    def send_draft(self, draft_id: str = None) -> str:
        """
        Sends a draft email.
           Print the returned draft's message and id.
           Returns: Draft object, including draft id and message meta data.

        Args:
            draft_id (str): the id of the draft to be updated

        """
        self._cache_service()
        service = self.service
        return (
            service.users().drafts().send(userId="me", body={"id": draft_id}).execute()
        )
