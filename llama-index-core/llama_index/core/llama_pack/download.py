import json
import os
from typing import Optional, Type

from llama_index.core.download.integration import download_integration
from llama_index.core.download.pack import (
    LLAMA_PACKS_CONTENTS_URL,
    download_llama_pack_template,
    track_download,
    parse_pack_identifier,
)
from llama_index.core.llama_pack.base import BaseLlamaPack


def download_llama_pack(
    llama_pack_class: str,
    download_dir: Optional[str] = None,
    llama_pack_url: str = LLAMA_PACKS_CONTENTS_URL,
    refresh_cache: bool = True,
) -> Optional[Type[BaseLlamaPack]]:
    """
    Download a single LlamaPack PyPi Package.

    Args:
        llama_pack_class: The name of the LlamaPack class you want to download,
            such as `GmailOpenAIAgentPack`. Can also include marketplace specifier
            in format `PackName@marketplace-name` (e.g., `SkillPack@huggingface-skills`).
        refresh_cache: If true, the local cache will be skipped and the
            loader will be fetched directly from the remote repo.
        download_dir: Custom dirpath to download the pack into.

    Returns:
        A Loader.

    """
    pack_cls = None

    # Parse pack identifier to extract marketplace if specified
    pack_name, marketplace_name = parse_pack_identifier(llama_pack_class)

    mappings_path = os.path.join(
        os.path.abspath(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
        ),
        "command_line/mappings.json",
    )
    with open(mappings_path) as f:
        mappings = json.load(f)

    if pack_name in mappings:
        new_import_parent = mappings[pack_name]
        new_install_parent = new_import_parent.replace(".", "-").replace("_", "-")
    else:
        raise ValueError(f"Failed to find python package for class {pack_name}")

    if not download_dir:
        pack_cls = download_integration(
            module_str=new_install_parent,
            module_import_str=new_import_parent,
            cls_name=pack_name,
        )
    else:
        pack_cls = download_llama_pack_template(
            new_install_parent=new_install_parent,
            llama_pack_class=pack_name,
            llama_pack_url=llama_pack_url,
            refresh_cache=refresh_cache,
            custom_path=download_dir,
            marketplace_name=marketplace_name,
        )
        track_download(pack_name, "llamapack")
        if pack_cls is None:
            return None

        if not issubclass(pack_cls, BaseLlamaPack):
            raise ValueError(
                f"Pack class {pack_cls} must be a subclass of BaseLlamaPack."
            )

    return pack_cls
