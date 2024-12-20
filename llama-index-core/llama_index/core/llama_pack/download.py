import json
import os
from typing import Optional, Type

from llama_index.core.download.integration import download_integration
from llama_index.core.download.pack import (
    LLAMA_PACKS_CONTENTS_URL,
    download_llama_pack_template,
    track_download,
)
from llama_index.core.llama_pack.base import BaseLlamaPack


def download_llama_pack(
    llama_pack_class: str,
    download_dir: Optional[str] = None,
    llama_pack_url: str = LLAMA_PACKS_CONTENTS_URL,
    refresh_cache: bool = True,
) -> Optional[Type[BaseLlamaPack]]:
    """Download a single LlamaPack PyPi Package.

    Args:
        llama_pack_class: The name of the LlamaPack class you want to download,
            such as `GmailOpenAIAgentPack`.
        refresh_cache: If true, the local cache will be skipped and the
            loader will be fetched directly from the remote repo.
        download_dir: Custom dirpath to download the pack into.

    Returns:
        A Loader.
    """
    pack_cls = None

    mappings_path = os.path.join(
        os.path.abspath(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
        ),
        "command_line/mappings.json",
    )
    with open(mappings_path) as f:
        mappings = json.load(f)

    if llama_pack_class in mappings:
        new_import_parent = mappings[llama_pack_class]
        new_install_parent = new_import_parent.replace(".", "-").replace("_", "-")
    else:
        raise ValueError(f"Failed to find python package for class {llama_pack_class}")

    if not download_dir:
        pack_cls = download_integration(
            module_str=new_install_parent,
            module_import_str=new_import_parent,
            cls_name=llama_pack_class,
        )
    else:
        pack_cls = download_llama_pack_template(
            new_install_parent=new_install_parent,
            llama_pack_class=llama_pack_class,
            llama_pack_url=llama_pack_url,
            refresh_cache=refresh_cache,
            custom_path=download_dir,
        )
        track_download(llama_pack_class, "llamapack")
        if pack_cls is None:
            return None

        if not issubclass(pack_cls, BaseLlamaPack):
            raise ValueError(
                f"Pack class {pack_cls} must be a subclass of BaseLlamaPack."
            )

    return pack_cls
