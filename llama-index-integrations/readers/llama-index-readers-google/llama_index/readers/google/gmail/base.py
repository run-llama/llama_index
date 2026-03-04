"""Google Mail reader."""

import base64
import email
from typing import Any, List, Optional

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document
from pydantic import BaseModel

SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]


class GmailReader(BaseReader, BaseModel):
    """
    Gmail reader.

    Reads emails

    Args:
        max_results (int): Defaults to 10.
        query (str): Gmail query. Defaults to None.
        service (Any): Gmail service. Defaults to None.
        results_per_page (Optional[int]): Max number of results per page. Defaults to 10.
        use_iterative_parser (bool): Use iterative parser. Defaults to False.

    """

    query: str = None
    use_iterative_parser: bool = False
    max_results: int = 10
    service: Any
    results_per_page: Optional[int]

    def load_data(self) -> List[Document]:
        """Load emails from the user's account."""
        from googleapiclient.discovery import build

        credentials = self._get_credentials()
        if not self.service:
            self.service = build("gmail", "v1", credentials=credentials)

        messages = self.search_messages()

        results = []
        for message in messages:
            text = message.pop("body")
            extra_info = message
            results.append(Document(text=text, extra_info=extra_info or {}))

        return results

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
                creds = flow.run_local_server(port=8080)
            # Save the credentials for the next run
            with open("token.json", "w") as token:
                token.write(creds.to_json())

        return creds

    def search_messages(self):
        query = self.query

        max_results = self.max_results
        if self.results_per_page:
            max_results = self.results_per_page

        results = (
            self.service.users()
            .messages()
            .list(userId="me", q=query, maxResults=int(max_results))
            .execute()
        )
        messages = results.get("messages", [])

        if len(messages) < self.max_results:
            # paginate if there are more results
            while "nextPageToken" in results:
                page_token = results["nextPageToken"]
                results = (
                    self.service.users()
                    .messages()
                    .list(
                        userId="me",
                        q=query,
                        pageToken=page_token,
                        maxResults=int(max_results),
                    )
                    .execute()
                )
                messages.extend(results["messages"])
                if len(messages) >= self.max_results:
                    break

        result = []
        try:
            for message in messages:
                message_data = self.get_message_data(message)
                if not message_data:
                    continue
                result.append(message_data)
        except Exception as e:
            raise Exception("Can't get message data" + str(e))

        return result

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

        # https://developers.google.com/gmail/api/reference/rest/v1/users.messages
        return {
            "id": message_data["id"],
            "threadId": message_data["threadId"],
            "snippet": message_data["snippet"],
            "internalDate": message_data["internalDate"],
            "body": body,
        }

    def extract_message_body_iterative(self, message: dict):
        if message["raw"]:
            body = base64.urlsafe_b64decode(message["raw"].encode("utf-8"))
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


if __name__ == "__main__":
    reader = GmailReader(query="from:me after:2023-01-01")
    print(reader.load_data())
