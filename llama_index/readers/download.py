"""Download loader from Llama Hub.

NOTE: using `download_loader` is now deprecated.
Please do `pip install llama-hub` instead.

"""

from typing import Optional, Type

from llama_index.download.module import (
    LLAMA_HUB_URL,
    MODULE_TYPE,
    download_llama_module,
    track_download,
)
from llama_index.readers.base import BaseReader


def download_loader(
    loader_class: str,
    loader_hub_url: str = LLAMA_HUB_URL,
    refresh_cache: bool = False,
    use_gpt_index_import: bool = False,
    custom_path: Optional[str] = None,
) -> Type[BaseReader]:
    """Download a single loader from the Loader Hub.

    Args:
        loader_class: The name of the loader class you want to download,
            such as `SimpleWebPageReader`.
        refresh_cache: If true, the local cache will be skipped and the
            loader will be fetched directly from the remote repo.
        use_gpt_index_import: If true, the loader files will use
            llama_index as the base dependency. By default (False),
            the loader files use llama_index as the base dependency.
            NOTE: this is a temporary workaround while we fully migrate all usages
            to llama_index.
        custom_path: Custom dirpath to download loader into.

    Returns:
        A Loader.
    """
    # Only one of the `custom_dir` or `custom_path` is supported.
    if custom_path is not None:
        custom_dir = None
    else:
        custom_dir = "llamahub_modules"

    reader_cls = download_llama_module(
        loader_class,
        llama_hub_url=loader_hub_url,
        refresh_cache=refresh_cache,
        custom_dir=custom_dir,
        custom_path=custom_path,
        use_gpt_index_import=use_gpt_index_import,
    )
    if not issubclass(reader_cls, BaseReader):
        raise ValueError(
            f"Loader class {loader_class} must be a subclass of BaseReader."
        )
    track_download(loader_class, MODULE_TYPE.LOADER)
    return reader_cls
