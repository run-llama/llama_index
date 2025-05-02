import os
import asyncio
import httpx
import time
from pathlib import Path
from tenacity import (
    retry,
    wait_exponential,
    before_sleep_log,
    after_log,
    retry_if_exception_type,
    stop_after_delay,
)
from typing import List, Optional, Union

from llama_index.core.async_utils import run_jobs
from llama_index.core.bridge.pydantic import Field, field_validator
from llama_index.core.readers.base import BasePydanticReader
from llama_index.core.schema import Document
from llama_index.readers.dashscope.utils import *

from llama_index.readers.dashscope.domain.lease_domains import (
    DownloadFileLeaseResult,
    UploadFileLeaseResult,
    AddFileResult,
    QueryFileResult,
    DatahubDataStatusEnum,
)

DASHSCOPE_DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com"
DASHSCOPE_DEFAULT_DC_CATEGORY = os.getenv(
    "DASHSCOPE_DEFAULT_DC_CATEGORY", default="default"
)

logger = get_stream_logger(name=__name__)


class DashScopeParse(BasePydanticReader):
    """A smart-parser for files."""

    api_key: str = Field(
        default="",
        description="The API key for the DashScope API.",
        validate_default=True,
    )
    workspace_id: str = Field(
        default="",
        description="The Workspace  for the DashScope API.If not set, "
        "it will use the default workspace.",
        validate_default=True,
    )
    category_id: str = Field(
        default=DASHSCOPE_DEFAULT_DC_CATEGORY,
        description="The dc category for the DashScope API.If not set, "
        "it will use the default dc category.",
        validate_default=True,
    )
    base_url: str = Field(
        default=DASHSCOPE_DEFAULT_BASE_URL,
        description="The base URL of the DashScope Parsing API.",
        validate_default=True,
    )
    result_type: ResultType = Field(
        default=ResultType.DASHSCOPE_DOCMIND,
        description="The result type for the parser.",
    )
    num_workers: int = Field(
        default=4,
        gt=0,
        lt=10,
        description="The number of workers to use sending API requests for parsing.",
    )
    check_interval: int = Field(
        default=5,
        description="The interval in seconds to check if the parsing is done.",
    )
    max_timeout: int = Field(
        default=3600,
        description="The maximum timeout in seconds to wait for the parsing to finish.",
    )
    verbose: bool = Field(
        default=True, description="Whether to print the progress of the parsing."
    )
    show_progress: bool = Field(
        default=True, description="Show progress when parsing multiple files."
    )
    ignore_errors: bool = Field(
        default=True,
        description="Whether or not to ignore and skip errors raised during parsing.",
    )
    parse_result: bool = Field(
        default=True,
        description="Whether or not to return parsed text content.",
    )

    @field_validator("api_key", mode="before", check_fields=True)
    def validate_api_key(cls, v: str) -> str:
        """Validate the API key."""
        if not v:
            import os

            api_key = os.getenv("DASHSCOPE_API_KEY", None)
            if api_key is None:
                raise ValueError("The API key [DASHSCOPE_API_KEY] is required.")
            return api_key

        return v

    @field_validator("workspace_id", mode="before", check_fields=True)
    def validate_workspace_id(cls, v: str) -> str:
        """Validate the Workspace."""
        if not v:
            import os

            return os.getenv("DASHSCOPE_WORKSPACE_ID", "")

        return v

    @field_validator("category_id", mode="before", check_fields=True)
    def validate_category_id(cls, v: str) -> str:
        """Validate the category."""
        if not v:
            import os

            return os.getenv("DASHSCOPE_CATEGORY_ID", DASHSCOPE_DEFAULT_DC_CATEGORY)
        return v

    @field_validator("base_url", mode="before", check_fields=True)
    def validate_base_url(cls, v: str) -> str:
        """Validate the base URL."""
        if v and v != DASHSCOPE_DEFAULT_BASE_URL:
            return v
        else:
            url = (
                os.getenv("DASHSCOPE_BASE_URL", None)
                or "https://dashscope.aliyuncs.com"
            )
            if url and not url.startswith(("http://", "https://")):
                raise ValueError(
                    "The DASHSCOPE_BASE_URL must start with http or https. "
                )
            return url or DASHSCOPE_DEFAULT_BASE_URL

    def _get_dashscope_header(self):
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-DashScope-WorkSpace": f"{self.workspace_id}",
            "X-DashScope-OpenAPISource": "CloudSDK",
        }

    # upload a document and get back a job_id
    async def _create_job(
        self, file_path: str, extra_info: Optional[dict] = None
    ) -> str:
        file_path = str(file_path)
        UploadFileLeaseResult.is_file_valid(file_path=file_path)

        headers = self._get_dashscope_header()

        # load data
        with open(file_path, "rb") as f:
            upload_file_lease_result = self.__upload_lease(file_path, headers)

            upload_file_lease_result.upload(file_path, f)

            url = f"{self.base_url}/api/v1/datacenter/category/{self.category_id}/add_file"
            async with httpx.AsyncClient(timeout=self.max_timeout) as client:
                response = await client.post(
                    url,
                    headers=headers,
                    json={
                        "lease_id": upload_file_lease_result.lease_id,
                        "parser": ResultType.DASHSCOPE_DOCMIND.value,
                    },
                )
            add_file_result = dashscope_response_handler(
                response, "add_file", AddFileResult, url=url
            )

        return add_file_result.file_id

    @retry(
        stop=stop_after_delay(60),
        wait=wait_exponential(multiplier=5, max=60),
        before_sleep=before_sleep_log(logger, logging.INFO),
        after=after_log(logger, logging.INFO),
        reraise=True,
        retry=retry_if_exception_type(RetryException),
    )
    def __upload_lease(self, file_path, headers):
        url = f"{self.base_url}/api/v1/datacenter/category/{self.category_id}/upload_lease"
        try:
            with httpx.Client(timeout=self.max_timeout) as client:
                response = client.post(
                    url,
                    headers=headers,
                    json={
                        "file_name": os.path.basename(file_path),
                        "size_bytes": os.path.getsize(file_path),
                        "content_md5": get_file_md5(file_path),
                    },
                )
        except httpx.ConnectTimeout:
            raise RetryException("Connect timeout")
        except httpx.ReadTimeout:
            raise RetryException("Read timeout")
        except httpx.NetworkError:
            raise RetryException("Network error")

        upload_file_lease_result = dashscope_response_handler(
            response, "upload_lease", UploadFileLeaseResult, url=url
        )
        logger.info(
            f"{file_path} upload lease result: {upload_file_lease_result.lease_id}"
        )
        return upload_file_lease_result

    async def _get_job_result(
        self, data_id: str, result_type: str, verbose: bool = False
    ) -> dict:
        result_url = f"{self.base_url}/api/v1/datacenter/category/{self.category_id}/file/{data_id}/download_lease"
        status_url = f"{self.base_url}/api/v1/datacenter/category/{self.category_id}/file/{data_id}/query"

        headers = self._get_dashscope_header()

        start = time.time()
        tries = 0
        while True:
            await asyncio.sleep(1)
            tries += 1
            query_file_result = await self._dashscope_query(
                data_id, headers, status_url
            )

            status = query_file_result.status
            if DatahubDataStatusEnum.PARSE_SUCCESS.value == status:
                async with httpx.AsyncClient(timeout=self.max_timeout) as client:
                    response = await client.post(
                        result_url, headers=headers, json={"file_id": data_id}
                    )
                    down_file_lease_result = dashscope_response_handler(
                        response,
                        "download_lease",
                        DownloadFileLeaseResult,
                        url=result_url,
                    )
                    if self.parse_result:
                        return {
                            result_type: down_file_lease_result.download(escape=True),
                            "job_id": data_id,
                        }
                    else:
                        return {result_type: "{}", "job_id": data_id}
            elif (
                DatahubDataStatusEnum.PARSING.value == status
                or DatahubDataStatusEnum.INIT.value == status
            ):
                end = time.time()
                if end - start > self.max_timeout:
                    raise Exception(f"Timeout while parsing the file: {data_id}")
                if verbose and tries % 5 == 0:
                    print(".", end="", flush=True)

                await asyncio.sleep(self.check_interval)

                continue
            else:
                raise Exception(
                    f"Failed to parse the file: {data_id}, status: {status}"
                )

    @retry(
        stop=stop_after_delay(60),
        wait=wait_exponential(multiplier=5, max=60),
        before_sleep=before_sleep_log(logger, logging.INFO),
        after=after_log(logger, logging.INFO),
        reraise=True,
        retry=retry_if_exception_type(RetryException),
    )
    async def _dashscope_query(self, data_id, headers, status_url):
        try:
            async with httpx.AsyncClient(timeout=self.max_timeout) as client:
                response = await client.post(
                    status_url, headers=headers, json={"file_id": data_id}
                )
                return dashscope_response_handler(
                    response, "query", QueryFileResult, url=status_url
                )
        except httpx.ConnectTimeout:
            raise RetryException("Connect timeout")
        except httpx.ReadTimeout:
            raise RetryException("Read timeout")
        except httpx.NetworkError:
            raise RetryException("Network error")

    async def _aload_data(
        self, file_path: str, extra_info: Optional[dict] = None, verbose: bool = False
    ) -> List[Document]:
        """Load data from the input path."""
        try:
            data_id = await self._create_job(file_path, extra_info=extra_info)
            logger.info(f"Started parsing the file [{file_path}] under [{data_id}]")

            result = await self._get_job_result(
                data_id, self.result_type.value, verbose=verbose
            )

            document = Document(
                text=result[self.result_type.value],
                metadata=extra_info or {},
            )
            document.id_ = data_id

            return [document]

        except Exception as e:
            logger.error(f"Error while parsing the file '{file_path}':{e!s}")
            if self.ignore_errors:
                return []
            else:
                raise

    async def aload_data(
        self, file_path: Union[List[str], str], extra_info: Optional[dict] = None
    ) -> List[Document]:
        """Load data from the input path."""
        if isinstance(file_path, (str, Path)):
            return await self._aload_data(
                file_path, extra_info=extra_info, verbose=self.verbose
            )
        elif isinstance(file_path, list):
            jobs = [
                self._aload_data(
                    f,
                    extra_info=extra_info,
                    verbose=self.verbose and not self.show_progress,
                )
                for f in file_path
            ]
            try:
                results = await run_jobs(
                    jobs,
                    workers=self.num_workers,
                    desc="Parsing files",
                    show_progress=self.show_progress,
                )

                # return flattened results
                return [item for sublist in results for item in sublist]
            except RuntimeError as e:
                if nest_asyncio_err in str(e):
                    raise RuntimeError(nest_asyncio_msg)
                else:
                    raise
        else:
            raise ValueError(
                "The input file_path must be a string or a list of strings."
            )

    def load_data(
        self, file_path: Union[List[str], str], extra_info: Optional[dict] = None
    ) -> List[Document]:
        extra_info = {"parse_fmt_type": ResultType.DASHSCOPE_DOCMIND.value}
        """Load data from the input path."""
        try:
            return asyncio.run(self.aload_data(file_path, extra_info))
        except RuntimeError as e:
            if nest_asyncio_err in str(e):
                raise RuntimeError(nest_asyncio_msg)
            else:
                raise

    async def _aget_json(
        self, file_path: str, extra_info: Optional[dict] = None
    ) -> List[dict]:
        """Load data from the input path."""
        try:
            job_id = await self._create_job(file_path, extra_info=extra_info)
            if self.verbose:
                logger.info("Started parsing the file under job_id %s" % job_id)

            result = await self._get_job_result(
                job_id, ResultType.DASHSCOPE_DOCMIND.value
            )
            result["job_id"] = job_id
            result["file_path"] = file_path
            return [result]

        except Exception as e:
            logger.info(f"Error while parsing the file '{file_path}':", e)
            if self.ignore_errors:
                return []
            else:
                raise

    async def aget_json(
        self, file_path: Union[List[str], str], extra_info: Optional[dict] = None
    ) -> List[dict]:
        """Load data from the input path."""
        if isinstance(file_path, (str, Path)):
            return await self._aget_json(file_path, extra_info=extra_info)
        elif isinstance(file_path, list):
            jobs = [self._aget_json(f, extra_info=extra_info) for f in file_path]
            try:
                results = await run_jobs(
                    jobs,
                    workers=self.num_workers,
                    desc="Parsing files",
                    show_progress=self.show_progress,
                )

                # return flattened results
                return [item for sublist in results for item in sublist]
            except RuntimeError as e:
                if nest_asyncio_err in str(e):
                    raise RuntimeError(nest_asyncio_msg)
                else:
                    raise
        else:
            raise ValueError(
                "The input file_path must be a string or a list of strings."
            )

    def get_json_result(
        self, file_path: Union[List[str], str], extra_info: Optional[dict] = None
    ) -> List[dict]:
        extra_info = {"parse_fmt_type": ResultType.DASHSCOPE_DOCMIND.value}
        """Parse the input path."""
        try:
            return asyncio.run(self.aget_json(file_path, extra_info))
        except RuntimeError as e:
            if nest_asyncio_err in str(e):
                raise RuntimeError(nest_asyncio_msg)
            else:
                raise

    def get_images(self, json_result: List[dict], download_path: str) -> List[dict]:
        raise NotImplementedError
