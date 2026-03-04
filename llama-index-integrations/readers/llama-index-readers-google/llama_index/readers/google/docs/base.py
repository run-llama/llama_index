"""Google docs reader."""

import json
import logging
import os
import random
import string
from typing import Any, List, Optional

import googleapiclient.discovery as discovery
from google_auth_oauthlib.flow import InstalledAppFlow

from llama_index.core.bridge.pydantic import Field
from llama_index.core.readers.base import BasePydanticReader
from llama_index.core.schema import Document

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials

SCOPES = ["https://www.googleapis.com/auth/documents.readonly"]

logger = logging.getLogger(__name__)

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


class GoogleDocsReader(BasePydanticReader):
    """
    Google Docs reader.

    Reads a page from Google Docs

    """

    is_remote: bool = True

    split_on_heading_level: Optional[int] = Field(
        default=None,
        description="If set the document will be split on the specified heading level.",
    )

    include_toc: bool = Field(
        default=True, description="Include table of contents elements."
    )

    @classmethod
    def class_name(cls) -> str:
        return "GoogleDocsReader"

    def load_data(self, document_ids: List[str]) -> List[Document]:
        """
        Load data from the input directory.

        Args:
            document_ids (List[str]): a list of document ids.

        """
        if document_ids is None:
            raise ValueError('Must specify a "document_ids" in `load_kwargs`.')

        results = []
        for document_id in document_ids:
            docs = self._load_doc(document_id)
            results.extend(docs)

        return results

    def _load_doc(self, document_id: str) -> str:
        """
        Load a document from Google Docs.

        Args:
            document_id: the document id.

        Returns:
            The document text.

        """
        credentials = self._get_credentials()
        docs_service = discovery.build("docs", "v1", credentials=credentials)
        google_doc = docs_service.documents().get(documentId=document_id).execute()
        google_doc_content = google_doc.get("body").get("content")

        doc_metadata = {"document_id": document_id}

        return self._structural_elements_to_docs(google_doc_content, doc_metadata)

    def _get_credentials(self) -> Any:
        """
        Get valid user credentials from storage.

        The file token.json stores the user's access and refresh tokens, and is
        created automatically when the authorization flow completes for the first
        time.

        Returns:
            Credentials, the obtained credential.

        """
        creds = None
        port = 8080
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

                with open("credentials.json") as json_file:
                    client_config = json.load(json_file)
                    redirect_uris = client_config["web"].get("redirect_uris", [])
                    if len(redirect_uris) > 0:
                        port = int(redirect_uris[0].strip("/").split(":")[-1])

                creds = flow.run_local_server(port=port)
            # Save the credentials for the next run
            with open("token.json", "w") as token:
                token.write(creds.to_json())

        return creds

    def _read_paragraph_element(self, element: Any) -> Any:
        """
        Return the text in the given ParagraphElement.

        Args:
            element: a ParagraphElement from a Google Doc.

        """
        text_run = element.get("textRun")
        if not text_run:
            return ""
        return text_run.get("content")

    def _read_structural_elements(self, elements: List[Any]) -> Any:
        """
        Recurse through a list of Structural Elements.

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

    def _determine_heading_level(self, element):
        """
        Extracts the heading level, label, and ID from a document element.

        Args:
            element: a Structural Element.

        """
        level = None
        heading_key = None
        heading_id = None
        if self.split_on_heading_level and "paragraph" in element:
            style = element.get("paragraph").get("paragraphStyle")
            style_type = style.get("namedStyleType", "")
            heading_id = style.get("headingId", None)
            if style_type == "TITLE":
                level = 0
                heading_key = "title"
            elif style_type.startswith("HEADING_"):
                level = int(style_type.split("_")[1])
                if level > self.split_on_heading_level:
                    return None, None, None

                heading_key = f"Header {level}"

        return level, heading_key, heading_id

    def _generate_doc_id(self, metadata: dict):
        if "heading_id" in metadata:
            heading_id = metadata["heading_id"]
        else:
            heading_id = "".join(
                random.choices(string.ascii_letters + string.digits, k=8)
            )
        return f"{metadata['document_id']}_{heading_id}"

    def _structural_elements_to_docs(
        self, elements: List[Any], doc_metadata: dict
    ) -> Any:
        """
        Recurse through a list of Structural Elements.

        Split documents on heading if split_on_heading_level is set.

        Args:
            elements: a list of Structural Elements.

        """
        docs = []

        current_heading_level = self.split_on_heading_level

        metadata = doc_metadata.copy()
        text = ""
        for value in elements:
            element_text = self._read_structural_elements([value])

            level, heading_key, heading_id = self._determine_heading_level(value)

            if level is not None:
                if level == self.split_on_heading_level:
                    if text.strip():
                        docs.append(
                            Document(
                                id_=self._generate_doc_id(metadata),
                                text=text,
                                metadata=metadata.copy(),
                            )
                        )
                        text = ""
                    if "heading_id" in metadata:
                        metadata["heading_id"] = heading_id
                elif level < current_heading_level:
                    metadata = doc_metadata.copy()

                metadata[heading_key] = element_text
                current_heading_level = level
            else:
                text += element_text

        if text:
            if docs:
                id_ = self._generate_doc_id(metadata)
            else:
                id_ = metadata["document_id"]
            docs.append(Document(id_=id_, text=text, metadata=metadata))

        return docs


if __name__ == "__main__":
    reader = GoogleDocsReader(split_on_heading_level=1)
    docs = reader.load_data(
        document_ids=["1UORoHYBKmOdcv4g94znMF0ildBYWiu3C2M2MEsWN4mM"]
    )
    logger.info(docs)
