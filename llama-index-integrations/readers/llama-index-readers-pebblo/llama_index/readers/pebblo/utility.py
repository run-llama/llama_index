from __future__ import annotations

import logging
import os
import pathlib
import platform
from typing import Optional, Tuple

from pydantic.v1.main import BaseModel

from llama_index.core.readers.base import BaseReader

logger = logging.getLogger(__name__)

PLUGIN_VERSION = "0.1.0"
CLASSIFIER_URL = os.getenv("PEBBLO_CLASSIFIER_URL", "http://localhost:8000")

# Supported loaders for Pebblo safe data loading
file_reader = ["CSVReader", "DocxReader", "PDFReader"]
dir_reader = ["SimpleDirectoryReader"]
in_memory = []

READER_TYPE_MAPPING = {"file": file_reader, "dir": dir_reader, "in-memory": in_memory}

SUPPORTED_LOADERS = (*file_reader, *dir_reader, *in_memory)

logger = logging.getLogger(__name__)


class Runtime(BaseModel):
    """
    This class represents a Runtime.

    Args:
        type (Optional[str]): Runtime type. Defaults to ""
        host (str): Hostname of runtime.
        path (str): Current working directory path.
        ip (Optional[str]): Ip of current runtime. Defaults to ""
        platform (str): Platform details of current runtime.
        os (str): OS name.
        os_version (str): OS version.
        language (str): Runtime kernel.
        language_version (str): version of current runtime kernel.
        runtime (Optional[str]) More runtime details. Defaults to ""

    """

    type: str = "local"
    host: str
    path: str
    ip: Optional[str] = ""
    platform: str
    os: str
    os_version: str
    language: str
    language_version: str
    runtime: str = "local"


class Framework(BaseModel):
    """
    This class represents a Framework instance.

    Args:
        name (str): Name of the Framework.
        version (str): Version of the Framework.

    """

    name: str
    version: str


class App(BaseModel):
    """
    This class represents an AI application.

    Args:
        name (str): Name of the app.
        owner (str): Owner of the app.
        description (Optional[str]): Description of the app.
        load_id (str): Unique load_id of the app instance.
        runtime (Runtime): Runtime details of app.
        framework (Framework): Framework details of the app
        plugin_version (str): Plugin version used for the app.

    """

    name: str
    owner: str
    description: Optional[str]
    load_id: str
    runtime: Runtime
    framework: Framework
    plugin_version: str


class Doc(BaseModel):
    """
    This class represents a pebblo document.

    Args:
        name (str): Name of app originating this document.
        owner (str): Owner of app.
        docs (list): List of documents with its metadata.
        plugin_version (str): Pebblo plugin Version
        load_id (str): Unique load_id of the app instance.
        loader_details (dict): Loader details with its metadata.
        loading_end (bool): Boolean, specifying end of loading of source.
        source_owner (str): Owner of the source of the loader.

    """

    name: str
    owner: str
    docs: list
    plugin_version: str
    load_id: str
    loader_details: dict
    loading_end: bool
    source_owner: str


def get_full_path(path: str) -> str:
    """
    Return absolute local path for a local file/directory,
    for network related path, return as is.

    Args:
        path (str): Relative path to be resolved.

    Returns:
        str: Resolved absolute path.

    """
    if (
        not path
        or ("://" in path)
        or (path[0] == "/")
        or (path in ["unknown", "-", "in-memory"])
    ):
        return path
    full_path = pathlib.Path(path).resolve()
    return str(full_path)


def get_reader_type(reader: str) -> str:
    """
    Return loader type among, file, dir or in-memory.

    Args:
        loader (str): Name of the loader, whose type is to be resolved.

    Returns:
        str: One of the loader type among, file/dir/in-memory.

    """
    for reader_type, readers in READER_TYPE_MAPPING.items():
        if reader in readers:
            return reader_type
    return "unknown"


def get_reader_full_path(reader: BaseReader, reader_type: str, **kwargs) -> str:
    """
    Return absolute source path of source of reader based on the
    keys present in Document object from reader.

    Args:
        reader (BaseReader): Llama document reader, derived from Baseloader.

    """
    location = "-"
    if not isinstance(reader, BaseReader):
        logger.error(
            "loader is not derived from BaseReader, source location will be unknown!"
        )
        return location
    loader_dict = reader.__dict__
    try:
        if reader_type == "SimpleDirectoryReader":
            if loader_dict.get("input_dir", None):
                location = loader_dict.get("input_dir")
        elif kwargs.get("file"):
            location = kwargs.get("file")
        elif kwargs.get("input_file"):
            location = kwargs.get("input_file")
    except Exception:
        pass
    return get_full_path(str(location))


def get_runtime() -> Tuple[Framework, Runtime]:
    """
    Fetch the current Framework and Runtime details.

    Returns:
        Tuple[Framework, Runtime]: Framework and Runtime for the current app instance.

    """
    from importlib.metadata import version

    try:
        reader_version = version("llama-index-readers-pebblo")
    except Exception:
        reader_version = "unknown"
    framework = Framework(name="Llama Pebblo Reader", version=reader_version)
    uname = platform.uname()

    runtime = Runtime(
        host=uname.node,
        path=os.environ["PWD"],
        platform=platform.platform(),
        os=uname.system,
        os_version=uname.version,
        ip=get_ip(),
        language="unknown",
        language_version=platform.python_version(),
    )
    if "Darwin" in runtime.os:
        runtime.type = "desktop"
        runtime.runtime = "Mac OSX"

    logger.debug(f"framework {framework}")
    logger.debug(f"runtime {runtime}")
    return framework, runtime


def get_ip() -> str:
    """
    Fetch local runtime IP address.

    Returns:
        str: IP address

    """
    import socket  # lazy imports

    host = socket.gethostname()
    try:
        public_ip = socket.gethostbyname(host)
    except Exception:
        public_ip = socket.gethostbyname("localhost")
    return public_ip
