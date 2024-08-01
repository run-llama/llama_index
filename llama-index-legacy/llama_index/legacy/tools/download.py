"""Download tool from Llama Hub."""

from typing import Optional, Type

from llama_index.legacy.download.module import (
    LLAMA_HUB_URL,
    MODULE_TYPE,
    download_llama_module,
    track_download,
)
from llama_index.legacy.tools.tool_spec.base import BaseToolSpec


def download_tool(
    tool_class: str,
    llama_hub_url: str = LLAMA_HUB_URL,
    refresh_cache: bool = False,
    custom_path: Optional[str] = None,
) -> Type[BaseToolSpec]:
    """Download a single tool from Llama Hub.

    Args:
        tool_class: The name of the tool class you want to download,
            such as `GmailToolSpec`.
        refresh_cache: If true, the local cache will be skipped and the
            loader will be fetched directly from the remote repo.
        custom_path: Custom dirpath to download loader into.

    Returns:
        A Loader.
    """
    tool_cls = download_llama_module(
        tool_class,
        llama_hub_url=llama_hub_url,
        refresh_cache=refresh_cache,
        custom_dir="tools",
        custom_path=custom_path,
        library_path="tools/library.json",
    )
    if not issubclass(tool_cls, BaseToolSpec):
        raise ValueError(f"Tool class {tool_class} must be a subclass of BaseToolSpec.")
    track_download(tool_class, MODULE_TYPE.TOOL)
    return tool_cls
