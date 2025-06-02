import os
import json
import requests
from enum import Enum
from llama_index.readers.dashscope.domain.base_domains import DictToObject
from llama_index.readers.dashscope.utils import get_stream_logger

logger = get_stream_logger(name=__name__)


class FileUploadMethod(Enum):
    OSS_PreSignedUrl = "OSS.PreSignedUrl"

    @classmethod
    def from_value(cls, value):
        for member in cls:
            if member.value == value:
                return member
        raise ValueError(f"No enum member found for value '{value}'")


class UploadFileParameter:
    def upload(self, file_name: str, file: object):
        pass


class OssPreSignedUrlParameter(UploadFileParameter, DictToObject):
    def __init__(self, url: str, method: str, headers: dict):
        self.url = url
        self.method = method
        self.headers = headers

    @classmethod
    def from_dict(cls, data: dict) -> "OssPreSignedUrlParameter":
        if "method" not in data:
            raise ValueError("OssPreSignedUrlParameter method key is required")
        if "headers" not in data:
            raise ValueError("OssPreSignedUrlParameter headers key is required")
        if "url" not in data:
            raise ValueError("OssPreSignedUrlParameter url key is required")
        else:
            return OssPreSignedUrlParameter(
                data["url"], data["method"], data["headers"]
            )

    def upload(self, file_name: str, file: object):
        logger.info(f"Start upload {file_name}.")
        try:
            if self.method == "PUT":
                response = requests.put(self.url, data=file, headers=self.headers)
            elif self.method == "POST":
                response = requests.post(self.url, data=file, headers=self.headers)
            else:
                raise Exception(f"Upload {file_name} unsupported method: {self.method}")
            if response.status_code != 200:
                raise Exception(
                    f"Upload {file_name} failed with status code: {response.status_code} \n {self.url} \n {self.headers} \n {response.text}"
                )
            logger.info(f"Upload {file_name} success.")
        except requests.ConnectionError as ce:
            logger.info(f"Upload {file_name} Error connecting to {self.url}: {ce}")
            raise
        except requests.RequestException as e:
            logger.info(
                f"Upload {file_name} An error occurred while uploading to {self.url}: {e}"
            )
            raise
        except Exception as e:
            logger.info(
                f"Upload {file_name} An error occurred while uploading to {self.url}: {e}"
            )
            raise


class UploadFileLeaseResult(DictToObject):
    def __init__(self, type: str, param: UploadFileParameter, lease_id: str):
        self.type: str = type
        self.param: UploadFileParameter = param
        self.lease_id: str = lease_id

    @classmethod
    def from_dict(cls, data: dict) -> "UploadFileLeaseResult":
        if "lease_id" not in data:
            raise ValueError("UploadFileLeaseResult lease_id key is required")
        if "param" not in data:
            raise ValueError("UploadFileLeaseResult param key is required")
        if "type" not in data:
            raise ValueError("UploadFileLeaseResult type key is required")
        else:
            if data["type"] == FileUploadMethod.OSS_PreSignedUrl.value:
                return cls(
                    data["type"],
                    OssPreSignedUrlParameter.from_dict(data["param"]),
                    data["lease_id"],
                )
            else:
                raise ValueError(f"Unsupported upload type: {data['type']}")

    @staticmethod
    def is_file_valid(file_path: str) -> None:
        if file_path is None or file_path.strip() == "":
            raise ValueError(f"file_path can't blank.")
        file_path = str(file_path)

        # file_ext = os.path.splitext(file_path)[1]
        # if file_ext not in SUPPORTED_FILE_TYPES:
        #     raise ValueError(
        #         f"Currently, only the following file types are supported: {SUPPORTED_FILE_TYPES} "
        #         f"Current file type: {file_ext}"
        #     )

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")

    def upload(self, file_name: str, file: object):
        if self.type == FileUploadMethod.OSS_PreSignedUrl.value:
            self.param.upload(file_name, file)
        else:
            raise ValueError(f"Invalid upload method: {self.type}")


class AddFileResult(DictToObject):
    def __init__(self, file_id: str, parser: str):
        self.file_id = file_id
        self.parser = parser

    @classmethod
    def from_dict(cls, data: dict):
        default_values = {"file_id": "", "parser": ""}

        file_id = data.get("file_id", default_values["file_id"])
        parser = data.get("parser", default_values["parser"])

        return cls(file_id, parser)


class QueryFileResult(DictToObject):
    def __init__(
        self,
        file_id: str,
        status: str,
        file_name: str,
        file_type: str,
        parser: str,
        size_bytes: int,
        upload_time: str,
        category: str,
    ):
        self.file_id = file_id
        self.status = status
        self.file_name = file_name
        self.file_type = file_type
        self.parser = parser
        self.size_bytes = size_bytes
        self.upload_time = upload_time
        self.category = category

    @classmethod
    def from_dict(cls, data: dict):
        """
        Creates an instance of `QueryFileResult` from a dictionary.

        Args:
            data (dict): A dictionary containing the necessary keys and values corresponding to the class attributes.

        Returns:
            QueryFileResult: An instance of `QueryFileResult` populated with data from the input dictionary.

        """
        default_values = {
            "file_id": "",
            "status": "",
            "file_name": "",
            "file_type": "",
            "parser": "",
            "size_bytes": 0,
            "upload_time": "",
            "category": "",
        }

        return cls(
            file_id=data.get("file_id", default_values["file_id"]),
            status=data.get("status", default_values["status"]),
            file_name=data.get("file_name", default_values["file_name"]),
            file_type=data.get("file_type", default_values["file_type"]),
            parser=data.get("parser", default_values["parser"]),
            size_bytes=data.get("size_bytes", default_values["size_bytes"]),
            upload_time=data.get("upload_time", default_values["upload_time"]),
            category=data.get("category", default_values["category"]),
        )


class FileDownloadType(Enum):
    HTTP = "HTTP"

    @classmethod
    def from_value(cls, value):
        for member in cls:
            if member.value == value:
                return member
        raise ValueError(f"No enum member found for value '{value}'")


class HttpDownloadParameter(DictToObject):
    def __init__(self, url, method, headers) -> None:
        self.url = url
        self.method = method
        self.headers = headers

    @classmethod
    def from_dict(cls, data: dict):
        """
        Creates an instance of `QueryFileResult` from a dictionary.

        Args:
            data (dict): A dictionary containing the necessary keys and values corresponding to the class attributes.

        Returns:
            QueryFileResult: An instance of `QueryFileResult` populated with data from the input dictionary.

        """
        default_values = {"url": "", "method": "GET", "headers": {}}

        return cls(
            url=data.get("url", default_values["url"]),
            method=data.get("method", default_values["method"]),
            headers=data.get("headers", default_values["headers"]),
        )


class DownloadFileLeaseResult(DictToObject):
    def __init__(self, file_id, lease_id, file_name, type, param) -> None:
        self.file_id = file_id
        self.lease_id = lease_id
        self.file_name = file_name
        self.type = type
        self.param = param

    @classmethod
    def from_dict(cls, data: dict):
        """
        Creates an instance of `QueryFileResult` from a dictionary.

        Args:
            data (dict): A dictionary containing the necessary keys and values corresponding to the class attributes.

        Returns:
            QueryFileResult: An instance of `QueryFileResult` populated with data from the input dictionary.

        """
        if "param" not in data:
            raise ValueError("download_lease result param is required")

        default_values = {
            "file_id": "",
            "lease_id": "",
            "file_name": "",
            "type": FileDownloadType.HTTP.value,
            "param": HttpDownloadParameter.from_dict(data["param"]),
        }

        return cls(
            file_id=data.get("file_id", default_values["file_id"]),
            lease_id=data.get("lease_id", default_values["lease_id"]),
            file_name=data.get("file_name", default_values["file_name"]),
            type=FileDownloadType.from_value(data.get("type", default_values["type"])),
            param=default_values["param"],
        )

    def download(self, escape: bool = False):
        if self.type == FileDownloadType.HTTP:
            if self.param.method == "GET":
                json_bytes = requests.get(
                    url=self.param.url, headers=self.param.headers
                ).content
                json_str = json_bytes.decode("utf-8")
                if escape:
                    return json.dumps(json_str, ensure_ascii=False)
                else:
                    return json_str
            else:
                raise ValueError(f"Invalid download method: {self.param.method}")
        else:
            raise ValueError(f"Invalid download type: {self.type}")


class DatahubDataStatusEnum(Enum):
    INIT = "INIT"
    PARSING = "PARSING"
    PARSE_SUCCESS = "PARSE_SUCCESS"
    PARSE_FAILED = "PARSE_FAILED"

    @classmethod
    def from_value(cls, value):
        for member in cls:
            if member.value == value:
                return member
        raise ValueError(f"No enum member found for value '{value}'")
