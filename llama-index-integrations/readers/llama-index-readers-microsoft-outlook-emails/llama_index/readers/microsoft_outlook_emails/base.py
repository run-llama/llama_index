import logging
import requests
from typing import List, Optional
from llama_index.core.readers.base import BasePydanticReader
from llama_index.core.bridge.pydantic import PrivateAttr

logger = logging.getLogger(__name__)


class OutlookEmailReader(BasePydanticReader):
    """
    Outlook Emails Reader using Microsoft Graph API.

    Reads emails from a given Outlook mailbox and indexes the subject and body.

    Args:
        client_id (str): The Application ID for the app registered in Microsoft Azure.
        client_secret (str): The application secret for the app registered in Azure.
        tenant_id (str): Unique identifier of the Azure Active Directory Instance.
        user_email (str): Email address of the user whose emails need to be fetched.
        folder (Optional[str]): The email folder to fetch emails from. Defaults to "Inbox".
        num_mails (int): Number of emails to retrieve. Defaults to 10.

    """

    client_id: str
    client_secret: str
    tenant_id: str
    user_email: str
    folder: Optional[str] = "Inbox"
    num_mails: int = 10

    _authorization_headers: Optional[dict] = PrivateAttr(default=None)

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        tenant_id: str,
        user_email: str,
        folder: Optional[str] = "Inbox",
        num_mails: int = 10,
    ):
        super().__init__(
            client_id=client_id,
            client_secret=client_secret,
            tenant_id=tenant_id,
            user_email=user_email,
            folder=folder,
            num_mails=num_mails,
        )

    def _ensure_token(self):
        """Ensures we have a valid access token."""
        if self._authorization_headers is None:
            token = self._get_access_token()
            self._authorization_headers = {"Authorization": f"Bearer {token}"}

    def _get_access_token(self) -> str:
        """Fetches the OAuth token from Microsoft."""
        token_url = f"https://login.microsoftonline.com/{self.tenant_id}/oauth2/token"
        payload = {
            "grant_type": "client_credentials",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "resource": "https://graph.microsoft.com/",
        }
        response = requests.post(token_url, data=payload)
        response.raise_for_status()
        return response.json().get("access_token")

    def _fetch_emails(self) -> List[dict]:
        """Fetches emails from the specified folder."""
        self._ensure_token()
        url = f"https://graph.microsoft.com/v1.0/users/{self.user_email}/mailFolders/{self.folder}/messages?$top={self.num_mails}"
        response = requests.get(url, headers=self._authorization_headers)
        response.raise_for_status()
        return response.json().get("value", [])

    def load_data(self) -> List[str]:
        """Loads emails as texts containing subject and body."""
        emails = self._fetch_emails()
        email_texts = []
        for email in emails:
            subject = email.get("subject", "No Subject")
            body = email.get("body", {}).get("content", "No Content")
            email_texts.append(f"Subject: {subject}\n\n{body}")
        return email_texts
