"""
Download loader from Llama Hub.

NOTE: using `download_loader` is now deprecated.
Please do `pip install llama-index-reader-<reader_name>` instead.

"""

import json
import os
from typing import Optional, Type

from deprecated import deprecated

from llama_index.core.download.integration import download_integration
from llama_index.core.readers.base import BaseReader


@deprecated(
    "`download_loader()` is deprecated. "
    "Please install tool using pip install directly instead."
)
def download_loader(
    loader_class: str,
    loader_hub_url: str = "",
    refresh_cache: bool = False,
    use_gpt_index_import: bool = False,
    custom_path: Optional[str] = None,
) -> Type[BaseReader]:  # pragma: no cover
    """
    Download a single loader from the Loader Hub.

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
    # maintain during deprecation period
    del loader_hub_url
    del refresh_cache
    del use_gpt_index_import
    del custom_path

    mappings_path = os.path.join(
        os.path.abspath(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
        ),
        "command_line/mappings.json",
    )
    with open(mappings_path) as f:
        mappings = json.load(f)

    if loader_class in mappings:
        new_import_parent = mappings[loader_class]
        new_install_parent = new_import_parent.replace(".", "-").replace("_", "-")
    else:
        raise ValueError(f"Failed to find python package for class {loader_class}")

    reader_cls = download_integration(
        module_str=new_install_parent,
        module_import_str=new_import_parent,
        cls_name=loader_class,
    )
    if not issubclass(reader_cls, BaseReader):
        raise ValueError(
            f"Loader class {loader_class} must be a subclass of BaseReader."
        )

    return reader_cls
