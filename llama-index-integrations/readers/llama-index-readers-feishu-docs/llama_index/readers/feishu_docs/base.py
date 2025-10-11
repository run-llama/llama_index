"""Feishu docs reader."""

import json
import os
import time
from typing import List

import requests
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document

# Copyright (2023) Bytedance Ltd. and/or its affiliates
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


class FeishuDocsReader(BaseReader):
    """
    Feishu Docs reader.

    Reads a page from Google Docs

    """

    host = "https://open.feishu.cn"
    documents_raw_content_url_path = "/open-apis/docx/v1/documents/{}/raw_content"
    tenant_access_token_internal_url_path = (
        "/open-apis/auth/v3/tenant_access_token/internal"
    )

    def __init__(self, app_id, app_secret) -> None:
        """

        Args:
            app_id: The unique identifier of the application is obtained after the application is created.
            app_secret: Application key, obtained after creating the application.

        """
        super().__init__()
        self.app_id = app_id
        self.app_secret = app_secret

        self.tenant_access_token = ""
        self.expire = 0

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
            doc = self._load_doc(document_id)
            results.append(Document(text=doc, extra_info={"document_id": document_id}))
        return results

    def _load_doc(self, document_id) -> str:
        """
        Load a document from Feishu Docs.

        Args:
            document_id: the document id.

        Returns:
            The document text.

        """
        url = self.host + self.documents_raw_content_url_path.format(document_id)
        if self.tenant_access_token == "" or self.expire < time.time():
            self._update_tenant_access_token()
        headers = {
            "Authorization": f"Bearer {self.tenant_access_token}",
            "Content-Type": "application/json; charset=utf-8",
        }
        response = requests.get(url, headers=headers)
        return response.json()["data"]["content"]

    def _update_tenant_access_token(self):
        """For update tenant_access_token."""
        url = self.host + self.tenant_access_token_internal_url_path
        headers = {"Content-Type": "application/json; charset=utf-8"}
        data = {"app_id": self.app_id, "app_secret": self.app_secret}
        response = requests.post(url, data=json.dumps(data), headers=headers)
        self.tenant_access_token = response.json()["tenant_access_token"]
        self.expire = time.time() + response.json()["expire"]

    def set_lark_domain(self):
        """The default API endpoints are for Feishu, in order to switch to Lark, we should use set_lark_domain."""
        self.host = "https://open.larksuite.com"


if __name__ == "__main__":
    app_id = os.environ.get("FEISHU_APP_ID")
    app_secret = os.environ.get("FEISHU_APP_SECRET")
    reader = FeishuDocsReader(app_id, app_secret)
    print(reader.load_data(document_ids=[os.environ.get("FEISHU_DOC_ID")]))
