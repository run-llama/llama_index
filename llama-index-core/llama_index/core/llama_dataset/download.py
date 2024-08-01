from typing import List, Tuple, Type

from llama_index.core.download.dataset import (
    LLAMA_DATASETS_LFS_URL,
    LLAMA_DATASETS_SOURCE_FILES_GITHUB_TREE_URL,
    LLAMA_DATASETS_URL,
)
from llama_index.core.download.dataset import download_llama_dataset as download
from llama_index.core.download.module import (
    MODULE_TYPE,
    track_download,
)
from llama_index.core.llama_dataset.base import BaseLlamaDataset
from llama_index.core.llama_dataset.evaluator_evaluation import (
    LabelledEvaluatorDataset,
    LabelledPairwiseEvaluatorDataset,
)
from llama_index.core.llama_dataset.rag import LabelledRagDataset
from llama_index.core.readers import SimpleDirectoryReader
from llama_index.core.schema import Document


def _resolve_dataset_class(filename: str) -> Type[BaseLlamaDataset]:
    """Resolve appropriate llama dataset class based on file name."""
    if "rag_dataset.json" in filename:
        return LabelledRagDataset
    elif "pairwise_evaluator_dataset.json" in filename:
        return LabelledPairwiseEvaluatorDataset
    elif "evaluator_dataset.json" in filename:
        return LabelledEvaluatorDataset
    else:
        raise ValueError("Unknown filename.")


def download_llama_dataset(
    llama_dataset_class: str,
    download_dir: str,
    llama_datasets_url: str = LLAMA_DATASETS_URL,
    llama_datasets_lfs_url: str = LLAMA_DATASETS_LFS_URL,
    llama_datasets_source_files_tree_url: str = LLAMA_DATASETS_SOURCE_FILES_GITHUB_TREE_URL,
    show_progress: bool = False,
    load_documents: bool = True,
) -> Tuple[Type[BaseLlamaDataset], List[Document]]:
    """Download dataset from datasets-LFS and llamahub.

    Args:
        dataset_class: The name of the llamadataset class you want to download,
            such as `PaulGrahamEssayDataset`.
        custom_dir: Custom dir name to download loader into (under parent folder).
        custom_path: Custom dirpath to download loader into.
        llama_datasets_url: Url for getting ordinary files from llama_datasets repo
        llama_datasets_lfs_url: Url for lfs-traced files llama_datasets repo
        llama_datasets_source_files_tree_url: Url for listing source_files contents
        refresh_cache: If true, the local cache will be skipped and the
            loader will be fetched directly from the remote repo.
        source_files_dirpath: The directory for storing source files
        library_path: File name of the library file.
        base_file_name: The rag dataset json file
        disable_library_cache: Boolean to control library cache
        override_path: Boolean to control overriding path
        show_progress: Boolean for showing progress on downloading source files
        load_documents: Boolean for whether or not source_files for LabelledRagDataset should
                        be loaded.

    Returns:
        a `BaseLlamaDataset` and a `List[Document]`
    """
    filenames: Tuple[str, str] = download(
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
    dataset_filename, source_files_dir = filenames
    track_download(llama_dataset_class, MODULE_TYPE.DATASETS)

    dataset = _resolve_dataset_class(dataset_filename).from_json(dataset_filename)
    documents = []

    # for now only rag datasets need to provide the documents
    # in order to build an index over them
    if "rag_dataset.json" in dataset_filename and load_documents:
        documents = SimpleDirectoryReader(input_dir=source_files_dir).load_data(
            show_progress=show_progress
        )

    return (dataset, documents)
