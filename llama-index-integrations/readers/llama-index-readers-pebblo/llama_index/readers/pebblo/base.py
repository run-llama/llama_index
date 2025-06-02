"""Pebblo's safe dataloader is a wrapper for document loaders."""

import logging
import os
import uuid
from http import HTTPStatus
from typing import Any, Dict, List

import requests
from llama_index.core.schema import Document, MetadataMode
from llama_index.core.readers.base import BaseReader

from llama_index.readers.pebblo.utility import (
    CLASSIFIER_URL,
    PLUGIN_VERSION,
    App,
    Doc,
    get_runtime,
    get_reader_full_path,
    get_reader_type,
)

logger = logging.getLogger(__name__)


class PebbloSafeReader(BaseReader):
    """
    Pebblo Safe Loader class is a wrapper around document loaders enabling the data
    to be scrutinized.
    """

    _discover_sent: bool = False
    _loader_sent: bool = False

    def __init__(
        self,
        llama_reader: BaseReader,
        name: str,
        owner: str = "",
        description: str = "",
    ):
        if not name or not isinstance(name, str):
            raise NameError("Must specify a valid name.")
        self.app_name = name
        self.load_id = str(uuid.uuid4())
        self.reader = llama_reader
        self.owner = owner
        self.description = description
        self.reader_name = str(type(self.reader)).split(".")[-1].split("'")[0]
        self.source_type = get_reader_type(self.reader_name)
        self.docs: List[Document] = []
        self.source_aggr_size = 0
        # generate app
        self.app = self._get_app_details()
        self._send_discover()

    def load_data(self, **kwargs) -> List[Document]:
        """
        Load Documents.

        Returns:
            list: Documents fetched from load method of the wrapped `reader`.

        """
        self.docs = self.reader.load_data(**kwargs)
        self._send_reader_doc(loading_end=True, **kwargs)
        return self.docs

    @classmethod
    def set_discover_sent(cls) -> None:
        cls._discover_sent = True

    @classmethod
    def set_reader_sent(cls) -> None:
        cls._reader_sent = True

    def _set_reader_details(self, **kwargs) -> None:
        self.source_path = get_reader_full_path(self.reader, self.reader_name, **kwargs)
        self.source_owner = PebbloSafeReader.get_file_owner_from_path(self.source_path)
        self.source_path_size = self.get_source_size(self.source_path)
        self.reader_details = {
            "loader": self.reader_name,
            "source_path": self.source_path,
            "source_type": self.source_type,
            **(
                {"source_path_size": str(self.source_path_size)}
                if self.source_path_size > 0
                else {}
            ),
        }

    def _send_reader_doc(self, loading_end: bool = False, **kwargs) -> None:
        """
        Send documents fetched from reader to pebblo-server. Internal method.

        Args:
            loading_end (bool, optional): Flag indicating the halt of data
                                        loading by reader. Defaults to False.

        """
        headers = {"Accept": "application/json", "Content-Type": "application/json"}

        docs = []
        self._set_reader_details(**kwargs)
        for doc in self.docs:
            page_content = str(doc.get_content(metadata_mode=MetadataMode.NONE))
            page_content_size = self.calculate_content_size(page_content)
            self.source_aggr_size += page_content_size
            docs.append(
                {
                    "doc": page_content,
                    "source_path": self.source_path,
                    "last_modified": doc.metadata.get("last_modified", None),
                    "file_owner": self.source_owner,
                    **(
                        {"source_path_size": self.source_path_size}
                        if self.source_path_size is not None
                        else {}
                    ),
                }
            )
        payload: Dict[str, Any] = {
            "name": self.app_name,
            "owner": self.owner,
            "docs": docs,
            "plugin_version": PLUGIN_VERSION,
            "load_id": self.load_id,
            "loader_details": self.reader_details,
            "loading_end": "false",
            "source_owner": self.source_owner,
        }
        if loading_end is True:
            payload["loading_end"] = "true"
            if "loader_details" in payload:
                payload["loader_details"]["source_aggr_size"] = self.source_aggr_size
        payload = Doc(**payload).dict(exclude_unset=True)
        load_doc_url = f"{CLASSIFIER_URL}/v1/loader/doc"
        try:
            resp = requests.post(
                load_doc_url, headers=headers, json=payload, timeout=20
            )
            if resp.status_code not in [HTTPStatus.OK, HTTPStatus.BAD_GATEWAY]:
                logger.warning(
                    f"Received unexpected HTTP response code: {resp.status_code}"
                )
            logger.debug(
                f"send_loader_doc: request \
                    url {resp.request.url}, \
                    body {str(resp.request.body)[:999]} \
                    len {len(resp.request.body if resp.request.body else [])} \
                    response status{resp.status_code} body {resp.json()}"
            )
        except requests.exceptions.RequestException:
            logger.warning("Unable to reach pebblo server.")
        except Exception:
            logger.warning("An Exception caught in _send_loader_doc.")
        if loading_end is True:
            PebbloSafeReader.set_reader_sent()

    @staticmethod
    def calculate_content_size(page_content: str) -> int:
        """
        Calculate the content size in bytes:
        - Encode the string to bytes using a specific encoding (e.g., UTF-8)
        - Get the length of the encoded bytes.

        Args:
            page_content (str): Data string.

        Returns:
            int: Size of string in bytes.

        """
        # Encode the content to bytes using UTF-8
        encoded_content = page_content.encode("utf-8")
        return len(encoded_content)

    def _send_discover(self) -> None:
        """Send app discovery payload to pebblo-server. Internal method."""
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        payload = self.app.dict(exclude_unset=True)
        app_discover_url = f"{CLASSIFIER_URL}/v1/app/discover"
        try:
            resp = requests.post(
                app_discover_url, headers=headers, json=payload, timeout=20
            )
            logger.debug(
                f"send_discover: request \
                    url {resp.request.url}, \
                    headers {resp.request.headers}, \
                    body {str(resp.request.body)[:999]} \
                    len {len(resp.request.body if resp.request.body else [])} \
                    response status{resp.status_code} body {resp.json()}"
            )
            if resp.status_code in [HTTPStatus.OK, HTTPStatus.BAD_GATEWAY]:
                PebbloSafeReader.set_discover_sent()
            else:
                logger.warning(
                    f"Received unexpected HTTP response code: {resp.status_code}"
                )
        except requests.exceptions.RequestException:
            logger.warning("Unable to reach pebblo server.")
        except Exception:
            logger.warning("An Exception caught in _send_discover.")

    def _get_app_details(self) -> App:
        """
        Fetch app details. Internal method.

        Returns:
            App: App details.

        """
        framework, runtime = get_runtime()
        return App(
            name=self.app_name,
            owner=self.owner,
            description=self.description,
            load_id=self.load_id,
            runtime=runtime,
            framework=framework,
            plugin_version=PLUGIN_VERSION,
        )

    @staticmethod
    def get_file_owner_from_path(file_path: str) -> str:
        """
        Fetch owner of local file path.

        Args:
            file_path (str): Local file path.

        Returns:
            str: Name of owner.

        """
        try:
            import pwd

            file_owner_uid = os.stat(file_path).st_uid
            file_owner_name = pwd.getpwuid(file_owner_uid).pw_name
        except Exception:
            file_owner_name = "unknown"
        return file_owner_name

    def get_source_size(self, source_path: str) -> int:
        """
        Fetch size of source path. Source can be a directory or a file.

        Args:
            source_path (str): Local path of data source.

        Returns:
            int: Source size in bytes.

        """
        if not source_path:
            return 0
        size = 0
        if os.path.isfile(source_path):
            size = os.path.getsize(source_path)
        elif os.path.isdir(source_path):
            total_size = 0
            for dirpath, _, filenames in os.walk(source_path):
                for f in filenames:
                    fp = os.path.join(dirpath, f)
                    if not os.path.islink(fp):
                        total_size += os.path.getsize(fp)
            size = total_size
        return size
