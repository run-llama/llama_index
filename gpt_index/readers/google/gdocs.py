"""Google docs reader."""

import os
from typing import Any, List

from gpt_index.readers.base import BaseReader
from gpt_index.readers.schema.base import Document

SCOPES = "https://www.googleapis.com/auth/documents.readonly"


# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


class GoogleDocsReader(BaseReader):
    """Google Docs reader.

    Reads a page from Google Docs

    """

    def __init__(self) -> None:
        """Initialize with parameters."""
        try:
            import google  # noqa: F401
            import google_auth_oauthlib  # noqa: F401
            import googleapiclient  # noqa: F401
        except ImportError:
            raise ValueError(
                "`google_auth_oauthlib`, `googleapiclient` and `google` "
                "must be installed to use the GoogleDocsReader.\n"
                "Please run `pip install --upgrade google-api-python-client "
                "google-auth-httplib2 google-auth-oauthlib`."
            )

    def load_data(self, **load_kwargs: Any) -> List[Document]:
        """Load data from the input directory."""
        document_ids: List[str] = load_kwargs.pop("document_ids", None)
        if document_ids is None:
            raise ValueError('Must specify a "document_ids" in `load_kwargs`.')

        results = []
        for document_id in document_ids:
            doc = self._load_doc(document_id)
            results.append(Document(doc, extra_info={"document_id": document_id}))
        return results

    def _load_doc(self, document_id: str) -> str:
        """Load a document from Google Docs.

        Args:
            document_id: the document id.

        Returns:
            The document text.
        """
        import googleapiclient.discovery as discovery

        credentials = self._get_credentials()
        docs_service = discovery.build("docs", "v1", credentials=credentials)
        doc = docs_service.documents().get(documentId=document_id).execute()
        doc_content = doc.get("body").get("content")
        return self._read_structural_elements(doc_content)

    def _get_credentials(self) -> Any:
        """Get valid user credentials from storage.

        The file token.json stores the user's access and refresh tokens, and is
        created automatically when the authorization flow completes for the first
        time.

        Returns:
            Credentials, the obtained credential.
        """
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
                creds = flow.run_local_server(port=0)
            # Save the credentials for the next run
            with open("token.json", "w") as token:
                token.write(creds.to_json())

        return creds

    def _read_paragraph_element(self, element: Any) -> Any:
        """Return the text in the given ParagraphElement.

        Args:
            element: a ParagraphElement from a Google Doc.
        """
        text_run = element.get("textRun")
        if not text_run:
            return ""
        return text_run.get("content")

    def _read_structural_elements(self, elements: List[Any]) -> Any:
        """Recurse through a list of Structural Elements.

        Read a document's text where text may be in nested elements.

        Args:
            elements: a list of Structural Elements.
        """
        text = ""
        for value in elements:
            if "paragraph" in value:
                elements = value.get("paragraph").get("elements")
                for elem in elements:
                    text += self._read_paragraph_element(elem)
            elif "table" in value:
                # The text in table cells are in nested Structural Elements
                # and tables may be nested.
                table = value.get("table")
                for row in table.get("tableRows"):
                    cells = row.get("tableCells")
                    for cell in cells:
                        text += self._read_structural_elements(cell.get("content"))
            elif "tableOfContents" in value:
                # The text in the TOC is also in a Structural Element.
                toc = value.get("tableOfContents")
                text += self._read_structural_elements(toc.get("content"))
        return text


if __name__ == "__main__":
    reader = GoogleDocsReader()
    print(
        reader.load_data(document_ids=["11ctUj_tEf5S8vs_dk8_BNi-Zk8wW5YFhXkKqtmU_4B8"])
    )
