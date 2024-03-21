"""Feishu wiki reader."""
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


class FeishuWikiReader(BaseReader):
    """Feishu Wiki reader.

    Reads pages from Feishu wiki under the space

    """

    host = "https://open.feishu.cn"
    wiki_nodes_url_path = "/open-apis/wiki/v2/spaces/{}/nodes"
    documents_raw_content_url_path = "/open-apis/docx/v1/documents/{}/raw_content"
    tenant_access_token_internal_url_path = (
        "/open-apis/auth/v3/tenant_access_token/internal"
    )

    def __init__(self, app_id: str, app_secret: str) -> None:
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

    def load_data(self, space_id: str, parent_node_token: str = None) -> List[Document]:
        """Load data from the input directory.

        Args:
            space_id (str): a space id.
            parent_node_token (str[optional]): a parent node token of the space
        """
        if space_id is None:
            raise ValueError('Must specify a "space_id" in `load_kwargs`.')

        document_ids = self._load_space(space_id, parent_node_token=parent_node_token)
        document_ids = list(set(document_ids))

        results = []
        for document_id in document_ids:
            doc = self._load_doc(document_id)
            results.append(Document(text=doc, extra_info={"document_id": document_id}))
        return results

    def _load_space(self, space_id: str, parent_node_token: str = None) -> str:
        if self.tenant_access_token == "" or self.expire < time.time():
            self._update_tenant_access_token()
        headers = {
            "Authorization": f"Bearer {self.tenant_access_token}",
            "Content-Type": "application/json; charset=utf-8",
        }

        url = self.host + self.wiki_spaces_url_path.format(space_id)
        if parent_node_token:
            url += f"?parent_node_token={parent_node_token}"
        try:
            response = requests.get(url, headers=headers)
            result = response.json()
        except Exception:
            return []
        if not result.get("data"):
            return []
        obj_token_list = []
        for item in result["data"]["items"]:
            obj_token_list.append(item["obj_token"])
            if item["has_child"]:
                child_obj_token_list = self._load_space(
                    space_id=space_id, parent_node_token=item["node_token"]
                )
                if child_obj_token_list:
                    obj_token_list.extend(child_obj_token_list)
        return obj_token_list

    def _load_doc(self, document_id: str) -> str:
        """Load a document from Feishu Docs.

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
        try:
            response = requests.get(url, headers=headers)
            result = response.json()
        except Exception:
            return None
        if not result.get("data"):
            return None
        return result["data"]["content"]

    def _update_tenant_access_token(self) -> None:
        """For update tenant_access_token."""
        url = self.host + self.tenant_access_token_internal_url_path
        headers = {"Content-Type": "application/json; charset=utf-8"}
        data = {"app_id": self.app_id, "app_secret": self.app_secret}
        response = requests.post(url, data=json.dumps(data), headers=headers)
        self.tenant_access_token = response.json()["tenant_access_token"]
        self.expire = time.time() + response.json()["expire"]

    def set_lark_domain(self, host: str) -> None:
        """Set lark domain."""
        self.host = host


if __name__ == "__main__":
    app_id = os.environ.get("FEISHU_APP_ID")
    app_secret = os.environ.get("FEISHU_APP_SECRET")
    reader = FeishuWikiReader(app_id, app_secret)
    print(
        reader.load_data(
            space_id=os.environ.get("FEISHU_SPACE_ID"),
            parent_node_token=os.environ.get("FEISHU_PARENT_NODE_TOKEN"),
        )
    )
