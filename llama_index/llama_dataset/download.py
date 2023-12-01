from typing import List, Tuple, Type

from llama_index import Document
from llama_index.download.dataset import (
    LLAMA_DATASETS_LFS_URL,
    LLAMA_DATASETS_SOURCE_FILES_GITHUB_TREE_URL,
    LLAMA_DATASETS_URL,
    download_llama_dataset as download,
)
from llama_index.download.download_utils import MODULE_TYPE, track_download
from llama_index.llama_dataset.base import BaseLlamaDataset
from llama_index.llama_dataset.rag import LabelledRagDataset
from llama_index.readers import SimpleDirectoryReader


def download_llama_dataset(
    llama_dataset_class: str,
    download_dir: str,
    llama_datasets_url: str = LLAMA_DATASETS_URL,
    llama_datasets_lfs_url: str = LLAMA_DATASETS_LFS_URL,
    llama_datasets_source_files_tree_url: str = LLAMA_DATASETS_SOURCE_FILES_GITHUB_TREE_URL,
    show_progress: bool = False,
) -> Tuple[Type[BaseLlamaDataset], List[Document]]:
    """Download a single LlamaDataset from Llama Hub.

    Args:
        llama_dataset_class: The name of the LlamaPack class you want to download,
            such as `PaulGrahamEssayDataset`.
        refresh_cache: If true, the local cache will be skipped and the
            loader will be fetched directly from the remote repo.
        download_dir: Custom dirpath to download the pack into.

    Returns:
        A Loader.
    """
    filenames: Tuple[str, List[str]] = download(
        llama_dataset_class,
        llama_datasets_url=llama_datasets_url,
        llama_datasets_lfs_url=llama_datasets_lfs_url,
        llama_datasets_source_files_tree_url=llama_datasets_source_files_tree_url,
        refresh_cache=True,
        custom_path=download_dir,
        library_path="library.json",
        disable_library_cache=True,
        override_path=True,
        show_progress=show_progress,
    )
    rag_dataset_filename, source_files_dir = filenames
    track_download(llama_dataset_class, MODULE_TYPE.DATASETS)
    return (
        LabelledRagDataset.from_json(rag_dataset_filename),
        SimpleDirectoryReader(input_dir=source_files_dir).load_data(
            show_progress=show_progress
        ),
    )
