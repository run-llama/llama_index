"""Azure ACA Session tool spec."""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from io import BufferedReader, BytesIO
from typing import Callable, List, Optional
from uuid import uuid4
import re
import os

from azure.identity import DefaultAzureCredential
from azure.core.credentials import AccessToken
import requests

from llama_index.core.tools.tool_spec.base import BaseToolSpec


@dataclass
class RemoteFileMetadata:
    """Metadata for a file in the session."""

    filename: str
    """The filename relative to `/mnt/data`."""

    size_in_bytes: int
    """The size of the file in bytes."""

    @property
    def full_path(self) -> str:
        """Get the full path of the file."""
        return f"/mnt/data/{self.filename}"

    @staticmethod
    def from_dict(data: dict) -> "RemoteFileMetadata":
        """Create a RemoteFileMetadata object from a dictionary."""
        return RemoteFileMetadata(
            filename=data["filename"],
            size_in_bytes=data["bytes"]
        )


def _sanitize_input(query: str) -> str:
    """
    Sanitize input and remove whitespace,
    backtick & python (if llm mistakes python console as terminal)

    Args:
        query: The query to sanitize

    Returns:
        str: The sanitized query
    """

    # Removes `, whitespace & python from start
    query = re.sub(r"^(\s|`)*(?i:python)?\s*", "", query)
    # Removes whitespace & ` from end
    query = re.sub(r"(\s|`)*$", "", query)
    return query


class AzureACASessionToolSpec(BaseToolSpec):
    """Azure Container App Session tool spec."""

    spec_functions = ["code_interpreter"]

    def __init__(
        self,
        pool_managment_endpoint: str,
    ) -> None:
        """Initialize with parameters."""
        self.pool_management_endpoint = pool_managment_endpoint

    def _access_token_provider_factory() -> Callable[[], Optional[str]]:
        access_token: AccessToken = None

        def access_token_provider() -> Optional[str]:
            """Create a function that returns an access token."""
            nonlocal access_token
            if access_token is None or datetime.fromtimestamp(access_token.expires_on, timezone.utc) > (datetime.datetime.now() + timedelta(minutes=5)):
                credential = DefaultAzureCredential()
                access_token = credential.get_token("https://dynamicsessions.io/.default")
            return access_token.token

        return access_token_provider

    access_token_provider: Callable[[], Optional[str]] = _access_token_provider_factory()
    """A function that returns the access token to use for the session pool."""

    session_id: str = str(uuid4())
    """The session ID to use for the session pool. Defaults to a random UUID."""

    def code_interpreter(self, sanitize_input: bool = True) -> str:
        """
        This tool is used to execute python commands
        when you need to perform calculations or computations.
        Input should be a valid python command.
        Returns the result, stdout, and stderr.

        Args:
            sanitize_input (bool): Whether to sanitize input.
        """

        def _build_url(self, path) -> str:
            pool_management_endpoint = self.pool_management_endpoint
            if not pool_management_endpoint:
                raise ValueError("pool_management_endpoint is not set")
            if not pool_management_endpoint.endswith("/"):
                pool_management_endpoint += "/"
            return pool_management_endpoint + path

        def _run(self, python_code: str) -> Any:
            if self.sanitize_input:
                python_code = _sanitize_input(python_code)

            access_token = self.access_token_provider()
            api_url = self._build_url("python/execute")
            headers = {
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
            }
            body = {
                "properties": {
                    "identifier": self.session_id,
                    "codeInputType": "inline",
                    "executionType": "synchronous",
                    "pythonCode": python_code,
                }
            }

            response = requests.post(api_url, headers=headers, json=body)
            response.raise_for_status()
            response_json = response.json()

            return f"result:\n{response_json['result']}\n\nstdout:\n{response_json['stdout']}\n\nstderr:\n{response_json['stderr']}"

        def upload_file(self, *, data: BufferedReader = None, remote_file_path: str = None, local_file_path: str = None) -> RemoteFileMetadata:
            """Upload a file to the session pool.

            Args:
                data: The data to upload.
                remote_file_path: The path to upload the file to, relative to `/mnt/data`. If local_file_path is provided, this is defaulted to its filename.
                local_file_path: The path to the local file to upload.

            Returns:
                RemoteFileMetadata: The metadata for the uploaded file
            """

            if data and local_file_path:
                raise ValueError("data and local_file_path cannot be provided together")

            if local_file_path:
                if not remote_file_path:
                    remote_file_path = os.path.basename(local_file_path)
                data = open(local_file_path, "rb")

            access_token = self.access_token_provider()
            api_url = self._build_url(f"python/uploadFile?identifier={self.session_id}")
            headers = {
                "Authorization": f"Bearer {access_token}",
            }
            payload = {}
            files = [
                ('file', (remote_file_path, data, 'application/octet-stream'))
            ]

            response = requests.request("POST", api_url, headers=headers, data=payload, files=files)
            response.raise_for_status()

            response_json = response.json()
            return RemoteFileMetadata.from_dict(response_json)
        
        def download_file(self, *, remote_file_path: str, local_file_path: str = None) -> Optional[BufferedReader]:
            """Download a file from the session pool.

            Args:
                remote_file_path: The path to download the file from, relative to `/mnt/data`.
                local_file_path: The path to save the downloaded file to. If not provided, the file is returned as a BufferedReader.

            Returns:
                BufferedReader: The data of the downloaded file.
            """
            access_token = self.access_token_provider()
            api_url = self._build_url(f"python/downloadFile?identifier={self.session_id}&filename={remote_file_path}")
            headers = {
                "Authorization": f"Bearer {access_token}",
            }

            response = requests.get(api_url, headers=headers)
            response.raise_for_status()

            if local_file_path:
                with open(local_file_path, "wb") as f:
                    f.write(response.content)
                return None

            return BytesIO(response.content)

        def list_files(self) -> list[RemoteFileMetadata]:
            """List the files in the session pool.

            Returns:
                list[RemoteFileMetadata]: The metadata for the files in the session pool
            """
            access_token = self.access_token_provider()
            api_url = self._build_url(f"python/files?identifier={self.session_id}")
            headers = {
                "Authorization": f"Bearer {access_token}",
            }

            response = requests.get(api_url, headers=headers)
            response.raise_for_status()

            response_json = response.json()
            return [RemoteFileMetadata.from_dict(entry) for entry in response_json["$values"]]