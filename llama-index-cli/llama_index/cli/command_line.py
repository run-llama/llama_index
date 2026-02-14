import argparse
from typing import Any, Optional

from llama_index.cli.rag import RagCLI, default_ragcli_persist_dir
from llama_index.cli.upgrade import upgrade_dir, upgrade_file
from llama_index.core.ingestion import IngestionCache, IngestionPipeline
from llama_index.core.download.module import LLAMA_HUB_URL
from llama_index.core.llama_dataset.download import (
    LLAMA_DATASETS_LFS_URL,
    LLAMA_DATASETS_SOURCE_FILES_GITHUB_TREE_URL,
    download_llama_dataset,
)
from llama_index.core.llama_pack.download import (
    LLAMA_PACKS_CONTENTS_URL,
    download_llama_pack,
)
from llama_index.core.llama_pack.marketplace import MarketplaceManager
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.text_splitter import SentenceSplitter

from llama_index.cli.new_package.base import init_new_package


def handle_init_package(
    name: str, kind: str, prefix: Optional[str] = None, **kwargs: Any
):
    init_new_package(integration_name=name, integration_type=kind, prefix=prefix)
    print(f"Successfully initialized package")


def handle_download_llama_pack(
    llama_pack_class: Optional[str] = None,
    download_dir: Optional[str] = None,
    llama_pack_url: str = LLAMA_PACKS_CONTENTS_URL,
    **kwargs: Any,
) -> None:
    assert llama_pack_class is not None
    assert download_dir is not None

    download_llama_pack(
        llama_pack_class=llama_pack_class,
        download_dir=download_dir or "./custom_llama_pack",
        llama_pack_url=llama_pack_url,
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
        llama_datasets_lfs_url=llama_datasets_lfs_url,
        llama_datasets_source_files_tree_url=llama_datasets_source_files_tree_url,
        show_progress=True,
        load_documents=False,
    )

    print(f"Successfully downloaded {llama_dataset_class} to {download_dir}")


def handle_marketplace_add(
    name: str,
    repository: str,
    branch: str = "main",
    base_path: str = "",
    description: str = "",
    **kwargs: Any,
) -> None:
    """Add a new plugin marketplace."""
    manager = MarketplaceManager()
    success = manager.add_marketplace(
        name=name,
        repository=repository,
        branch=branch,
        base_path=base_path,
        description=description,
    )

    if success:
        print(f"Successfully added marketplace '{name}'")
        print(f"Repository: {repository}")
        print(f"Branch: {branch}")
        if base_path:
            print(f"Base path: {base_path}")
        print(f"\nTo install a plugin, run:")
        print(f"  llamaindex-cli plugin install <PackName>@{name} --download-dir ./path")
    else:
        print(f"Error: Marketplace '{name}' already exists")


def handle_marketplace_list(**kwargs: Any) -> None:
    """List all registered marketplaces."""
    manager = MarketplaceManager()
    marketplaces = manager.list_marketplaces()

    if not marketplaces:
        print("No marketplaces registered")
        return

    print("Registered marketplaces:")
    print()
    for marketplace in marketplaces:
        print(f"  {marketplace.name}")
        print(f"    Repository: {marketplace.repository}")
        print(f"    Branch: {marketplace.branch}")
        if marketplace.base_path:
            print(f"    Base path: {marketplace.base_path}")
        if marketplace.description:
            print(f"    Description: {marketplace.description}")
        print()


def handle_marketplace_remove(name: str, **kwargs: Any) -> None:
    """Remove a marketplace."""
    manager = MarketplaceManager()
    success = manager.remove_marketplace(name)

    if success:
        print(f"Successfully removed marketplace '{name}'")
    else:
        print(f"Error: Marketplace '{name}' not found or cannot be removed")


def handle_plugin_install(
    plugin_identifier: str,
    download_dir: Optional[str] = None,
    **kwargs: Any,
) -> None:
    """Install a plugin from a marketplace."""
    from llama_index.core.llama_pack.download import download_llama_pack

    if "@" not in plugin_identifier:
        print(
            f"Error: Plugin identifier must include a marketplace specifier "
            f"in the format '<PluginName>@<marketplace-name>'.\n"
            f"Example: llamaindex-cli plugin install superpowers@superpowers-marketplace"
        )
        return

    pack_name, marketplace_name = plugin_identifier.split("@", 1)

    # Verify marketplace exists
    manager = MarketplaceManager()
    marketplace = manager.get_marketplace(marketplace_name)
    if marketplace is None:
        print(f"Error: Marketplace '{marketplace_name}' not found.")
        print(f"Register it first with:")
        print(
            f"  llamaindex-cli plugin marketplace add {marketplace_name} "
            f"--name {marketplace_name}"
        )
        return

    target_dir = download_dir or f"./plugins/{pack_name.lower()}"

    print(f"Installing plugin '{pack_name}' from marketplace '{marketplace_name}'...")
    print(f"  Repository: {marketplace.repository}")
    print(f"  Branch: {marketplace.branch}")
    print(f"  Download dir: {target_dir}")
    print()

    try:
        download_llama_pack(
            llama_pack_class=plugin_identifier,
            download_dir=target_dir,
        )
        print(f"Successfully installed '{pack_name}' from '{marketplace_name}'")
    except Exception as e:
        print(f"Error installing plugin: {e}")


def handle_plugin_list(**kwargs: Any) -> None:
    """List available plugins from all registered marketplaces."""
    manager = MarketplaceManager()
    marketplaces = manager.list_marketplaces()

    if not marketplaces:
        print("No marketplaces registered")
        return

    print("Registered marketplaces and available plugins:")
    print()
    for marketplace in marketplaces:
        print(f"  {marketplace.name}")
        print(f"    Repository: {marketplace.repository}")
        if marketplace.description:
            print(f"    Description: {marketplace.description}")
        print(f"    Install: llamaindex-cli plugin install <PluginName>@{marketplace.name}")
        print()


def default_rag_cli() -> RagCLI:
    from llama_index.embeddings.openai import OpenAIEmbedding  # pants: no-infer-dep

    try:
        import chromadb  # pants: no-infer-dep
        from llama_index.vector_stores.chroma import (
            ChromaVectorStore,
        )  # pants: no-infer-dep
    except ImportError:
        raise ImportError(
            "Default RAG pipeline uses chromadb. "
            "Install with `pip install llama-index-vector-stores-chroma "
            "or customize to use a different vector store."
        )

    persist_dir = default_ragcli_persist_dir()
    chroma_client = chromadb.PersistentClient(path=persist_dir)
    chroma_collection = chroma_client.create_collection("default", get_or_create=True)
    vector_store = ChromaVectorStore(
        chroma_collection=chroma_collection, persist_dir=persist_dir
    )
    docstore = SimpleDocumentStore()

    ingestion_pipeline = IngestionPipeline(
        transformations=[SentenceSplitter(), OpenAIEmbedding()],
        vector_store=vector_store,
        docstore=docstore,
        cache=IngestionCache(),
    )
    try:
        from llama_index.embeddings.openai import OpenAIEmbedding  # pants: no-infer-dep
    except ImportError:
        OpenAIEmbedding = None

    try:
        import chromadb

        from llama_index.vector_stores.chroma import (
            ChromaVectorStore,
        )  # pants: no-infer-dep
    except ImportError:
        ChromaVectorStore = None

    if OpenAIEmbedding and ChromaVectorStore:
        persist_dir = default_ragcli_persist_dir()
        chroma_client = chromadb.PersistentClient(path=persist_dir)
        chroma_collection = chroma_client.create_collection(
            "default", get_or_create=True
        )
        vector_store = ChromaVectorStore(
            chroma_collection=chroma_collection, persist_dir=persist_dir
        )
        docstore = SimpleDocumentStore()

        ingestion_pipeline = IngestionPipeline(
            transformations=[SentenceSplitter(), OpenAIEmbedding()],
            vector_store=vector_store,
            docstore=docstore,
            cache=IngestionCache(),
        )
        try:
            ingestion_pipeline.load(persist_dir=persist_dir)
        except FileNotFoundError:
            pass

        return RagCLI(
            ingestion_pipeline=ingestion_pipeline,
            verbose=False,
            persist_dir=persist_dir,
        )
    else:
        print(
            "Default RagCLI was not built. There are packages missing. Please"
            " install required dependencies by running "
            "`pip install llama-index-embeddings-openai llama-index-llms-openai chroma llama-index-vector-stores-chroma`"
        )
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="LlamaIndex CLI tool.")

    # Subparsers for the main commands
    subparsers = parser.add_subparsers(title="commands", dest="command", required=True)

    # llama rag command
    llamarag_parser = subparsers.add_parser(
        "rag", help="Ask a question to a document / a directory of documents."
    )
    RagCLI.add_parser_args(llamarag_parser, default_rag_cli)

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

    # Upgrade command
    upgrade_parser = subparsers.add_parser(
        "upgrade", help="Upgrade a directory containing notebooks or python files."
    )
    upgrade_parser.add_argument(
        "directory",
        type=str,
        help="The directory to upgrade. Will run on only .ipynb or .py files.",
    )
    upgrade_parser.set_defaults(func=lambda args: upgrade_dir(args.directory))

    # Upgrade command
    upgrade_file_parser = subparsers.add_parser(
        "upgrade-file", help="Upgrade a single notebook or python file."
    )
    upgrade_file_parser.add_argument(
        "path",
        type=str,
        help="The directory to upgrade. Will run on only .ipynb or .py files.",
    )
    upgrade_file_parser.set_defaults(func=lambda args: upgrade_file(args.path))

    # init package command
    new_package_parser = subparsers.add_parser(
        "new-package", help="Initialize a new llama-index package"
    )
    new_package_parser.add_argument(
        "-k",
        "--kind",
        type=str,
        help="Kind of package, e.g., llm, embedding, pack, etc.",
    )
    new_package_parser.add_argument(
        "-n",
        "--name",
        type=str,
        help="Name of python package",
    )
    new_package_parser.add_argument(
        "-p",
        "--prefix",
        type=str,
        required=False,
        help="Name of prefix package",
    )
    new_package_parser.set_defaults(func=lambda args: handle_init_package(**vars(args)))

    # marketplace commands (legacy, kept for backwards compatibility)
    marketplace_parser = subparsers.add_parser(
        "marketplace", help="Manage plugin marketplaces"
    )
    marketplace_subparsers = marketplace_parser.add_subparsers(
        title="marketplace commands", dest="marketplace_command", required=True
    )

    # marketplace add command
    marketplace_add_parser = marketplace_subparsers.add_parser(
        "add", help="Add a new plugin marketplace"
    )
    marketplace_add_parser.add_argument(
        "repository",
        type=str,
        help="GitHub repository in format 'owner/repo' (e.g., 'huggingface/skills')",
    )
    marketplace_add_parser.add_argument(
        "-n",
        "--name",
        type=str,
        required=True,
        help="Short name for the marketplace (used in install commands)",
    )
    marketplace_add_parser.add_argument(
        "-b",
        "--branch",
        type=str,
        default="main",
        help="Git branch to use (default: 'main')",
    )
    marketplace_add_parser.add_argument(
        "-p",
        "--base-path",
        type=str,
        default="",
        help="Base path within the repository for packs",
    )
    marketplace_add_parser.add_argument(
        "-d",
        "--description",
        type=str,
        default="",
        help="Human-readable description of the marketplace",
    )
    marketplace_add_parser.set_defaults(
        func=lambda args: handle_marketplace_add(**vars(args))
    )

    # marketplace list command
    marketplace_list_parser = marketplace_subparsers.add_parser(
        "list", help="List all registered marketplaces"
    )
    marketplace_list_parser.set_defaults(
        func=lambda args: handle_marketplace_list(**vars(args))
    )

    # marketplace remove command
    marketplace_remove_parser = marketplace_subparsers.add_parser(
        "remove", help="Remove a marketplace"
    )
    marketplace_remove_parser.add_argument(
        "name",
        type=str,
        help="Name of the marketplace to remove",
    )
    marketplace_remove_parser.set_defaults(
        func=lambda args: handle_marketplace_remove(**vars(args))
    )

    # plugin commands (unified plugin management interface)
    plugin_parser = subparsers.add_parser(
        "plugin", help="Manage plugins and plugin marketplaces"
    )
    plugin_subparsers = plugin_parser.add_subparsers(
        title="plugin commands", dest="plugin_command", required=True
    )

    # plugin install command
    plugin_install_parser = plugin_subparsers.add_parser(
        "install", help="Install a plugin from a marketplace"
    )
    plugin_install_parser.add_argument(
        "plugin_identifier",
        type=str,
        help=(
            "Plugin identifier in format '<PluginName>@<marketplace-name>' "
            "(e.g., 'superpowers@superpowers-marketplace')"
        ),
    )
    plugin_install_parser.add_argument(
        "-d",
        "--download-dir",
        type=str,
        default=None,
        help="Custom directory to download the plugin into.",
    )
    plugin_install_parser.set_defaults(
        func=lambda args: handle_plugin_install(**vars(args))
    )

    # plugin list command
    plugin_list_parser = plugin_subparsers.add_parser(
        "list", help="List available plugins from registered marketplaces"
    )
    plugin_list_parser.set_defaults(
        func=lambda args: handle_plugin_list(**vars(args))
    )

    # plugin marketplace sub-commands
    plugin_marketplace_parser = plugin_subparsers.add_parser(
        "marketplace", help="Manage plugin marketplaces"
    )
    plugin_marketplace_subparsers = plugin_marketplace_parser.add_subparsers(
        title="plugin marketplace commands",
        dest="plugin_marketplace_command",
        required=True,
    )

    # plugin marketplace add
    plugin_mp_add_parser = plugin_marketplace_subparsers.add_parser(
        "add", help="Add a new plugin marketplace"
    )
    plugin_mp_add_parser.add_argument(
        "repository",
        type=str,
        help="GitHub repository in format 'owner/repo' (e.g., 'obra/superpowers-marketplace')",
    )
    plugin_mp_add_parser.add_argument(
        "-n",
        "--name",
        type=str,
        required=True,
        help="Short name for the marketplace (used in install commands)",
    )
    plugin_mp_add_parser.add_argument(
        "-b",
        "--branch",
        type=str,
        default="main",
        help="Git branch to use (default: 'main')",
    )
    plugin_mp_add_parser.add_argument(
        "-p",
        "--base-path",
        type=str,
        default="",
        help="Base path within the repository for packs",
    )
    plugin_mp_add_parser.add_argument(
        "-d",
        "--description",
        type=str,
        default="",
        help="Human-readable description of the marketplace",
    )
    plugin_mp_add_parser.set_defaults(
        func=lambda args: handle_marketplace_add(**vars(args))
    )

    # plugin marketplace list
    plugin_mp_list_parser = plugin_marketplace_subparsers.add_parser(
        "list", help="List all registered marketplaces"
    )
    plugin_mp_list_parser.set_defaults(
        func=lambda args: handle_marketplace_list(**vars(args))
    )

    # plugin marketplace remove
    plugin_mp_remove_parser = plugin_marketplace_subparsers.add_parser(
        "remove", help="Remove a marketplace"
    )
    plugin_mp_remove_parser.add_argument(
        "name",
        type=str,
        help="Name of the marketplace to remove",
    )
    plugin_mp_remove_parser.set_defaults(
        func=lambda args: handle_marketplace_remove(**vars(args))
    )

    # Parse the command-line arguments
    args = parser.parse_args()

    # Call the appropriate function based on the command
    args.func(args)


if __name__ == "__main__":
    main()
