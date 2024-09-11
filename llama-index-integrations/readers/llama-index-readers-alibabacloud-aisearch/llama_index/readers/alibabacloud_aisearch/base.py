import base64
import os
import asyncio
from pathlib import Path
import re
import time
from typing import Any, List, Union

from llama_index.core.async_utils import run_jobs
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.readers.base import BasePydanticReader
from llama_index.core.schema import Document

from llama_index.core.base.llms.generic_utils import get_from_param_or_env

try:
    from alibabacloud_searchplat20240529.models import (
        CreateDocumentAnalyzeTaskRequestDocument,
        CreateDocumentAnalyzeTaskRequest,
        CreateDocumentAnalyzeTaskRequestOutput,
        CreateDocumentAnalyzeTaskResponse,
        GetDocumentAnalyzeTaskStatusRequest,
        GetDocumentAnalyzeTaskStatusResponse,
        CreateImageAnalyzeTaskRequestDocument,
        CreateImageAnalyzeTaskRequest,
        CreateImageAnalyzeTaskResponse,
        GetImageAnalyzeTaskStatusRequest,
        GetImageAnalyzeTaskStatusResponse,
    )
    from alibabacloud_tea_openapi.models import Config as AISearchConfig
    from alibabacloud_searchplat20240529.client import Client
    from Tea.exceptions import TeaException
except ImportError:
    raise ImportError(
        "Could not import alibabacloud_searchplat20240529 python package. "
        "Please install it with `pip install alibabacloud-searchplat20240529`."
    )

FilePath = Union[str, Path]


def retry_decorator(func, wait_seconds: int = 1):
    def wrap(*args, **kwargs):
        while True:
            try:
                return func(*args, **kwargs)
            except TeaException as e:
                if e.code == "Throttling.RateQuota":
                    time.sleep(wait_seconds)
                else:
                    raise

    return wrap


def aretry_decorator(func, wait_seconds: int = 1):
    async def wrap(*args, **kwargs):
        while True:
            try:
                return await func(*args, **kwargs)
            except TeaException as e:
                if e.code == "Throttling.RateQuota":
                    await asyncio.sleep(wait_seconds)
                else:
                    raise

    return wrap


class AlibabaCloudAISearchDocumentReader(BasePydanticReader):
    """
    Supported file types include PPT/PPTX, DOC/DOCX, PDF, and more.
    For further details, please visit `https://help.aliyun.com/zh/open-search/search-platform/developer-reference/api-details`.
    """

    _client: Client = PrivateAttr()

    aisearch_api_key: str = Field(default=None, exclude=True)
    endpoint: str = None

    service_id: str = "ops-document-analyze-001"
    workspace_name: str = "default"

    check_interval: int = 3
    num_workers: int = 4
    show_progress: bool = False

    def __init__(
        self, endpoint: str = None, aisearch_api_key: str = None, **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.aisearch_api_key = get_from_param_or_env(
            "aisearch_api_key", aisearch_api_key, "AISEARCH_API_KEY"
        )
        self.endpoint = get_from_param_or_env("endpoint", endpoint, "AISEARCH_ENDPOINT")

        config = AISearchConfig(
            bearer_token=self.aisearch_api_key,
            endpoint=self.endpoint,
            protocol="http",
        )

        self._client = Client(config=config)

    # upload a document and get back a task_id
    @aretry_decorator
    async def _create_task(
        self,
        file_path: str,
        file_type: str,
        **load_kwargs: Any,
    ) -> str:
        if file_path.startswith("http"):
            file_name = os.path.basename(file_path.split("?")[0].split("#")[0])
            if not file_type:
                file_type = os.path.splitext(file_name)[1][1:]
            document = CreateDocumentAnalyzeTaskRequestDocument(
                url=file_path,
                file_name=file_name,
                file_type=file_type,
            )
        else:
            file_name = os.path.basename(file_path)
            if not file_type:
                file_type = os.path.splitext(file_name)[1][1:]
            document = CreateDocumentAnalyzeTaskRequestDocument(
                content=base64.b64encode(open(file_path, "rb").read()).decode(),
                file_name=file_name,
                file_type=file_type,
            )
        if not file_type:
            raise ValueError(
                "The file_type cannot be determined based on the file extension. Please specify it manually."
            )
        output = CreateDocumentAnalyzeTaskRequestOutput(
            image_storage=load_kwargs.get("image_storage", "url")
        )
        request = CreateDocumentAnalyzeTaskRequest(document=document, output=output)
        response: CreateDocumentAnalyzeTaskResponse = (
            await self._client.create_document_analyze_task_async(
                self.workspace_name, self.service_id, request
            )
        )
        return response.body.result.task_id

    async def _get_task_result(self, task_id: str) -> Document:
        request = GetDocumentAnalyzeTaskStatusRequest(task_id=task_id)
        while True:
            response: GetDocumentAnalyzeTaskStatusResponse = (
                await self._client.get_document_analyze_task_status_async(
                    self.workspace_name, self.service_id, request
                )
            )
            status = response.body.result.status
            if status == "PENDING":
                await asyncio.sleep(self.check_interval)
            elif status == "SUCCESS":
                data = response.body.result.data
                return Document(
                    text=data.content,
                    mimetype=f"text/{data.content_type}",
                )
            else:
                raise RuntimeError(
                    f"Failed to parse the file, error: {response.body.result.error}, task id: {task_id}"
                )

    async def _aload_data(
        self,
        file_path: str,
        file_type: str = None,
        **load_kwargs: Any,
    ) -> Document:
        """Load data from the input path."""
        task_id = await self._create_task(file_path, file_type, **load_kwargs)
        return await self._get_task_result(task_id)

    async def aload_data(
        self,
        file_path: Union[List[FilePath], FilePath],
        file_type: Union[List[FilePath], FilePath] = None,
        **load_kwargs: Any,
    ) -> List[Document]:
        """Load data from the input path."""
        if isinstance(file_path, (str, Path)):
            doc = await self._aload_data(str(file_path), file_type, **load_kwargs)
            return [doc]
        elif isinstance(file_path, list):
            if isinstance(file_type, list) and len(file_type) != len(file_path):
                raise ValueError(
                    "The length of file_type must be the same as file_path."
                )
            else:
                file_type = [file_type] * len(file_path)
            jobs = [
                self._aload_data(
                    str(f),
                    t,
                    **load_kwargs,
                )
                for f, t in zip(file_path, file_type)
            ]
            return await run_jobs(
                jobs,
                workers=self.num_workers,
                desc="Parsing files",
                show_progress=self.show_progress,
            )
        else:
            raise ValueError(
                "The input file_path must be a string or a list of strings."
            )

    def load_data(
        self,
        file_path: Union[List[FilePath], FilePath],
        **load_kwargs: Any,
    ) -> List[Document]:
        """Load data from the input path."""
        return asyncio.get_event_loop().run_until_complete(
            self.aload_data(file_path, **load_kwargs)
        )


class AlibabaCloudAISearchImageReader(AlibabaCloudAISearchDocumentReader):
    """
    For further details, please visit `https://help.aliyun.com/zh/open-search/search-platform/developer-reference/opensearch-api-details`.
    """

    service_id: str = "ops-image-analyze-ocr-001"

    # upload a document and get back a task_id
    @aretry_decorator
    async def _create_task(
        self,
        file_path: str,
        file_type: str,
        **load_kwargs: Any,
    ) -> str:
        if file_path.startswith("data:"):
            prefix, content = file_path.split(",")
            if not file_type:
                m = re.match(r"^data:image/(\w+);base64$", prefix)
                file_type = m.group(1)
            file_name = f"image.{file_type}"
            document = CreateImageAnalyzeTaskRequestDocument(
                content=content,
                file_name=file_name,
                file_type=file_type,
            )
        elif file_path.startswith("http"):
            file_name = os.path.basename(file_path.split("?")[0].split("#")[0])
            if not file_type:
                file_type = os.path.splitext(file_name)[1][1:]
            document = CreateImageAnalyzeTaskRequestDocument(
                url=file_path,
                file_name=file_name,
                file_type=file_type,
            )
        else:
            file_name = os.path.basename(file_path)
            if not file_type:
                file_type = os.path.splitext(file_name)[1][1:]
            document = CreateImageAnalyzeTaskRequestDocument(
                content=base64.b64encode(open(file_path, "rb").read()).decode(),
                file_name=file_name,
                file_type=file_type,
            )
        if not file_type:
            raise ValueError(
                "The file_type cannot be determined based on the file extension. Please specify it manually."
            )
        request = CreateImageAnalyzeTaskRequest(document=document)
        response: CreateImageAnalyzeTaskResponse = (
            await self._client.create_image_analyze_task_async(
                self.workspace_name, self.service_id, request
            )
        )
        return response.body.result.task_id

    async def _get_task_result(self, task_id: str) -> Document:
        request = GetImageAnalyzeTaskStatusRequest(task_id=task_id)
        while True:
            response: GetImageAnalyzeTaskStatusResponse = (
                await self._client.get_image_analyze_task_status_async(
                    self.workspace_name, self.service_id, request
                )
            )
            status = response.body.result.status
            if status == "PENDING":
                await asyncio.sleep(self.check_interval)
            elif status == "SUCCESS":
                data = response.body.result.data
                return Document(
                    text=data.content,
                    mimetype=f"text/{data.content_type}",
                )
            else:
                raise RuntimeError(
                    f"Failed to parse the file, error: {response.body.result.error}, task id: {task_id}"
                )
