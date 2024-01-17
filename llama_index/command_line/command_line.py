import argparse
from typing import Any, Optional

from llama_index.llama_dataset.download import (
    LLAMA_DATASETS_LFS_URL,
    LLAMA_DATASETS_SOURCE_FILES_GITHUB_TREE_URL,
    download_llama_dataset,
)
from llama_index.llama_pack.download import LLAMA_HUB_URL, download_llama_pack


def handle_download_llama_pack(
    llama_pack_class: Optional[str] = None,
    download_dir: Optional[str] = None,
    llama_hub_url: str = LLAMA_HUB_URL,
    **kwargs: Any,
) -> None:
    assert llama_pack_class is not None
    assert download_dir is not None

    download_llama_pack(
        llama_pack_class=llama_pack_class,
        download_dir=download_dir,
        llama_hub_url=llama_hub_url,
    )
    print(f"Successfully downloaded {llama_pack_class} to {download_dir}")


def handle_download_llama_dataset(
    llama_dataset_class: Optional[str] = None,
    download_dir: Optional[str] = None,
    llama_hub_url: str = LLAMA_HUB_URL,
    llama_datasets_lfs_url: str = LLAMA_DATASETS_LFS_URL,
    llama_datasets_source_files_tree_url: str = LLAMA_DATASETS_SOURCE_FILES_GITHUB_TREE_URL,
    **kwargs: Any,
) -> None:
    assert llama_dataset_class is not None
    assert download_dir is not None

    download_llama_dataset(
        llama_dataset_class=llama_dataset_class,
        download_dir=download_dir,
        llama_hub_url=llama_hub_url,
        llama_datasets_lfs_url=llama_datasets_lfs_url,
        llama_datasets_source_files_tree_url=llama_datasets_source_files_tree_url,
        show_progress=True,
        load_documents=False,
    )

    print(f"Successfully downloaded {llama_dataset_class} to {download_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="LlamaIndex CLI tool.")

    # Subparsers for the main commands
    subparsers = parser.add_subparsers(title="commands", dest="command", required=True)

    # download llamapacks command
    llamapack_parser = subparsers.add_parser(
        "download-llamapack", help="Download a llama-pack"
    )
    llamapack_parser.add_argument(
        "llama_pack_class",
        type=str,
        help=(
            "The name of the llama-pack class you want to download, "
            "such as `GmailOpenAIAgentPack`."
        ),
    )
    llamapack_parser.add_argument(
        "-d",
        "--download-dir",
        type=str,
        default="./llama_packs",
        help="Custom dirpath to download the pack into.",
    )
    llamapack_parser.add_argument(
        "--llama-hub-url",
        type=str,
        default=LLAMA_HUB_URL,
        help="URL to llama hub.",
    )
    llamapack_parser.set_defaults(
        func=lambda args: handle_download_llama_pack(**vars(args))
    )

    # download llamadatasets command
    llamadataset_parser = subparsers.add_parser(
        "download-llamadataset", help="Download a llama-dataset"
    )
    llamadataset_parser.add_argument(
        "llama_dataset_class",
        type=str,
        help=(
            "The name of the llama-dataset class you want to download, "
            "such as `PaulGrahamEssayDataset`."
        ),
    )
    llamadataset_parser.add_argument(
        "-d",
        "--download-dir",
        type=str,
        default="./llama_datasets",
        help="Custom dirpath to download the pack into.",
    )
    llamadataset_parser.add_argument(
        "--llama-hub-url",
        type=str,
        default=LLAMA_HUB_URL,
        help="URL to llama hub.",
    )
    llamadataset_parser.add_argument(
        "--llama-datasets-lfs-url",
        type=str,
        default=LLAMA_DATASETS_LFS_URL,
        help="URL to llama datasets.",
    )
    llamadataset_parser.set_defaults(
        func=lambda args: handle_download_llama_dataset(**vars(args))
    )

    # Parse the command-line arguments
    args = parser.parse_args()

    # Call the appropriate function based on the command
    args.func(args)


if __name__ == "__main__":
    main()
