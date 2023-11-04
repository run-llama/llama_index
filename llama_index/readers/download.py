"""Download loader from the Loader Hub.

NOTE: using `download_loader` is now deprecated.
Please do `pip install llama-hub` instead.

"""

from typing import Optional, Type
from llama_index.download.download_utils import download_llama_module, LLAMA_HUB_URL

from llama_index.readers.base import BaseReader

def download_loader(
    loader_class: str,
    loader_hub_url: str = LLAMA_HUB_URL,
    refresh_cache: Optional[bool] = False,
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
    
    download_llama_module(
        loader_class,
        llama_hub_url=loader_hub_url,
        refresh_cache=refresh_cache,
        suffix="llamahub_modules",
        custom_path=custom_path,
        use_gpt_index_import=use_gpt_index_import,
    )

