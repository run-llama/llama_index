"""ICTFax tool spec.

Tools for sending and tracking faxes through an ICTFax / ICTCore server over its
REST API. ICTFax is an open source fax server (https://ictfax.org) built on the
ICTCore framework; the same API also backs ICTPBX and ICTDialer.

The ICTCore send flow is: authenticate for a JWT, create a document record and
upload its file, register the document as a ``sendfax`` program, create a contact
for the recipient number, then create a transmission and send it. These tools wrap
that flow so an agent can send a fax in a couple of calls and poll its status.
"""

from typing import Any, List, Optional

import requests

from llama_index.core.schema import Document
from llama_index.core.tools.tool_spec.base import BaseToolSpec


class ICTFaxToolSpec(BaseToolSpec):
    """ICTFax fax tool spec.

    Args:
        base_url: Base URL of the ICTCore REST API, e.g. ``https://fax.example.com/api``.
        username: API username.
        password: API password.
        account_id: Account/route id used for outbound faxes. Defaults to 1.
        verify_tls: Verify TLS certificates. Set False only for self-signed test boxes.
        timeout: Per-request timeout in seconds.
    """

    spec_functions = [
        "send_fax",
        "get_fax_status",
        "list_faxes",
        "upload_fax_document",
    ]

    def __init__(
        self,
        base_url: str,
        username: str,
        password: str,
        account_id: int = 1,
        verify_tls: bool = True,
        timeout: int = 30,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._username = username
        self._password = password
        self.account_id = account_id
        self.verify_tls = verify_tls
        self.timeout = timeout
        self._token: Optional[str] = None

    # ---- internal helpers -------------------------------------------------
    def _authenticate(self) -> str:
        resp = requests.post(
            f"{self.base_url}/authenticate",
            json={"username": self._username, "password": self._password},
            verify=self.verify_tls,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        token = resp.json().get("token")
        if not token:
            raise RuntimeError("ICTCore authenticate returned no token")
        self._token = token
        return token

    @staticmethod
    def _id(resp: Any) -> Optional[int]:
        if isinstance(resp, bool):
            return None
        if isinstance(resp, (int, float)):
            return int(resp)
        if isinstance(resp, str) and resp.strip().lstrip("-").isdigit():
            return int(resp.strip())
        if isinstance(resp, dict):
            for k in ("document_id", "contact_id", "program_id", "transmission_id", "id"):
                if resp.get(k) is not None:
                    return resp[k]
        return None

    def _request(self, method: str, path: str, content_type: Optional[str] = None, **kwargs: Any) -> Any:
        if self._token is None:
            self._authenticate()

        def _do() -> requests.Response:
            # ICTCore reads the JWT from the standard Authorization header.
            headers = {"Authorization": f"Bearer {self._token}"}
            if content_type:
                headers["Content-Type"] = content_type
            elif "files" not in kwargs and "data" not in kwargs:
                headers["Content-Type"] = "application/json"
            return requests.request(
                method, f"{self.base_url}/{path.lstrip('/')}", headers=headers,
                verify=self.verify_tls, timeout=self.timeout, **kwargs,
            )

        resp = _do()
        if resp.status_code == 401:
            self._authenticate()
            resp = _do()
        resp.raise_for_status()
        if resp.content:
            try:
                return resp.json()
            except ValueError:
                return resp.text
        return None

    # ---- tools ------------------------------------------------------------
    def upload_fax_document(self, file_path: str, name: Optional[str] = None,
                            mime: str = "application/pdf") -> str:
        """
        Upload a document (PDF, TIFF or image) to fax and return its document id.

        Call this first when you have a local file to fax; pass the returned id to
        send_fax as document_id.

        Args:
            file_path (str): Path to the local file to upload.
            name (str): Optional display name for the document.
            mime (str): MIME type of the file. Defaults to application/pdf.

        """
        meta = self._request("POST", "/documents",
                             json={"name": name or file_path.split("/")[-1]})
        doc_id = self._id(meta)
        with open(file_path, "rb") as fh:
            body = fh.read()
        self._request("POST", f"/documents/{doc_id}/media", data=body, content_type=mime)
        return f"Uploaded document id {doc_id}"

    def send_fax(
        self,
        to_number: str,
        document_id: int,
        title: str = "Fax",
        recipient_name: str = "Fax recipient",
    ) -> str:
        """
        Send a fax of an already-uploaded document to a destination number.

        Args:
            to_number (str): Destination fax number in international format.
            document_id (int): Id of a document already uploaded with upload_fax_document.
            title (str): Short label for this fax, shown in the transmission list.
            recipient_name (str): Name to store on the recipient contact.

        """
        contact_id = self._id(self._request(
            "POST", "/contacts", json={"first_name": recipient_name, "phone": to_number}))
        program_id = self._id(self._request(
            "POST", "/programs/sendfax", json={"name": title, "document_id": document_id}))
        transmission_id = self._id(self._request("POST", "/transmissions", json={
            "title": title, "contact_id": contact_id, "program_id": program_id,
            "account_id": self.account_id, "direction": "outbound"}))
        self._request("POST", f"/transmissions/{transmission_id}/send")
        return (
            f"Fax queued to {to_number}. transmission_id={transmission_id}. "
            f"Use get_fax_status({transmission_id}) to track it."
        )

    def get_fax_status(self, transmission_id: int) -> str:
        """
        Get the delivery status of a fax transmission.

        Args:
            transmission_id (int): Id returned by send_fax.

        """
        t = self._request("GET", f"/transmissions/{transmission_id}")
        status = t.get("status", "unknown") if isinstance(t, dict) else "unknown"
        response = t.get("response", "") if isinstance(t, dict) else ""
        line = f"transmission {transmission_id}: status={status}"
        if response:
            line += f" ({response})"
        return line

    def list_faxes(self) -> List[Document]:
        """
        List fax transmissions on the server with their status.

        Returns one Document per transmission so results can be read or indexed.
        """
        items = self._request("GET", "/transmissions")
        if isinstance(items, dict):
            items = items.get("data", items.get("transmissions", [items]))
        docs: List[Document] = []
        for t in items or []:
            docs.append(
                Document(
                    text=(
                        f"transmission {t.get('transmission_id', t.get('id'))}: "
                        f"title={t.get('title')} direction={t.get('direction')} "
                        f"status={t.get('status')} {t.get('response', '')}"
                    )
                )
            )
        return docs
