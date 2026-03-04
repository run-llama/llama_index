"""Download tool from Llama Index. Deprecated."""

import json
import os
from typing import Optional, Type

from deprecated import deprecated

from llama_index.core.download.integration import download_integration
from llama_index.core.tools.tool_spec.base import BaseToolSpec


@deprecated(
    "`download_tool()` is deprecated. "
    "Please install tool using pip install directly instead."
)
def download_tool(
    tool_class: str,
    llama_hub_url: str = "",
    refresh_cache: bool = False,
    custom_path: Optional[str] = None,
) -> Type[BaseToolSpec]:
    """
    Download a single tool from Llama Hub.

    Args:
        tool_class: The name of the tool class you want to download,
            such as `GmailToolSpec`.
        refresh_cache: If true, the local cache will be skipped and the
            loader will be fetched directly from the remote repo.
        custom_path: Custom dirpath to download loader into.

    Returns:
        A Loader.

    """
    del llama_hub_url  # maintain during deprecation period
    del refresh_cache
    del custom_path
    mappings_path = os.path.join(
        os.path.abspath(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
        ),
        "command_line/mappings.json",
    )
    with open(mappings_path) as f:
        mappings = json.load(f)

    if tool_class in mappings:
        new_import_parent = mappings[tool_class]
        new_install_parent = new_import_parent.replace(".", "-").replace("_", "-")
    else:
        raise ValueError(f"Failed to find python package for class {tool_class}")

    tool_cls = download_integration(
        module_str=new_install_parent,
        module_import_str=new_import_parent,
        cls_name=tool_class,
    )

    if not issubclass(tool_cls, BaseToolSpec):
        raise ValueError(f"Tool class {tool_class} must be a subclass of BaseToolSpec.")

    return tool_cls
