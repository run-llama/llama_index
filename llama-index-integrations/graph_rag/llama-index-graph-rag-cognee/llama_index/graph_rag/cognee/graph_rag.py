import os
import pathlib
import webbrowser
from typing import List, Union

import cognee as cognee_lib

from llama_index.core import Document

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import GraphRAG

# mypy: disable-error-code="attr-defined"


class CogneeGraphRAG:
    """
    Cognee GraphRAG, handles adding, storing, processing and retrieving information from knowledge graphs.

    Unlike traditional RAG models that retrieve unstructured text snippets, graphRAG utilizes knowledge graphs.
    A knowledge graph represents entities as nodes and their relationships as edges, often in a structured semantic format.
    This enables the system to retrieve more precise and structured information about an entity, its relationships, and its properties.

    Attributes:
    llm_api_key: str: API key for desired LLM.
    llm_provider: str: Provider for desired LLM (default: "openai").
    llm_model: str: Model for desired LLM (default: "gpt-4o-mini").
    graph_db_provider: str: The graph database provider (default: "kuzu").
                            Supported providers: "neo4j", "networkx", "kuzu".
    graph_database_url: str: URL for the graph database.
    graph_database_username: str: Username for accessing the graph database.
    graph_database_password: str: Password for accessing the graph database.
    vector_db_provider: str: The vector database provider (default: "lancedb").
                             Supported providers: "lancedb", "pgvector", "qdrant", "weviate".
    vector_db_url: str: URL for the vector database.
    vector_db_key: str: API key for accessing the vector database.
    relational_db_provider: str: The relational database provider (default: "sqlite").
                            Supported providers: "sqlite", "postgres".
    db_name: str: The name of the databases (default: "cognee_db").
    db_host: str: Host for the relational database.
    db_port: str: Port for the relational database.
    db_username: str: Username for the relational database.
    db_password: str: Password for the relational database.

    """

    def __init__(
        self,
        llm_api_key: str,
        llm_provider: str = "openai",
        llm_model: str = "gpt-4o-mini",
        graph_db_provider: str = "kuzu",
        graph_database_url: str = "",
        graph_database_username: str = "",
        graph_database_password: str = "",
        vector_db_provider: str = "lancedb",
        vector_db_url: str = "",
        vector_db_key: str = "",
        relational_db_provider: str = "sqlite",
        relational_db_name: str = "cognee_db",
        relational_db_host: str = "",
        relational_db_port: str = "",
        relational_db_username: str = "",
        relational_db_password: str = "",
    ) -> None:
        cognee_lib.config.set_llm_config(
            {
                "llm_api_key": llm_api_key,
                "llm_provider": llm_provider,
                "llm_model": llm_model,
            }
        )

        cognee_lib.config.set_vector_db_config(
            {
                "vector_db_url": vector_db_url,
                "vector_db_key": vector_db_key,
                "vector_db_provider": vector_db_provider,
            }
        )
        cognee_lib.config.set_relational_db_config(
            {
                "db_path": "",
                "db_name": relational_db_name,
                "db_host": relational_db_host,
                "db_port": relational_db_port,
                "db_username": relational_db_username,
                "db_password": relational_db_password,
                "db_provider": relational_db_provider,
            }
        )

        cognee_lib.config.set_graph_db_config(
            {
                "graph_database_provider": graph_db_provider,
                "graph_database_url": graph_database_url,
                "graph_database_username": graph_database_username,
                "graph_database_password": graph_database_password,
            }
        )

        data_directory_path = str(
            pathlib.Path(
                os.path.join(pathlib.Path(__file__).parent, ".data_storage/")
            ).resolve()
        )

        cognee_lib.config.data_root_directory(data_directory_path)
        cognee_directory_path = str(
            pathlib.Path(
                os.path.join(pathlib.Path(__file__).parent, ".cognee_system/")
            ).resolve()
        )
        cognee_lib.config.system_root_directory(cognee_directory_path)
        cognee_lib.config.data_root_directory(data_directory_path)

    async def add(
        self, data: Union[Document, List[Document]], dataset_name: str = "main_dataset"
    ) -> None:
        """
        Add data to the specified dataset.
        This data will later be processed and made into a knowledge graph.

        Args:
            data (Union[Document, List[Document]]): The document(s) to be added to the graph.
                Can be a single Document or a list of Documents.
            dataset_name (str): Name of the dataset or node set where the data will be added.
                               Note: While cognee supports custom dataset organization, this integration
                               currently adds all data to 'main_dataset'. Full dataset_name support
                               will be added in a future version. This parameter is included to show
                               the intended API design.

        """
        # Convert LlamaIndex Document type to text
        text_data: List[str]
        if isinstance(data, List) and len(data) > 0:
            text_data = [doc.text for doc in data if isinstance(doc, Document)]
        elif isinstance(data, Document):
            text_data = [data.text]
        else:
            raise ValueError(
                "Invalid data type. Please provide a list of Documents or a single Document."
            )

        await cognee_lib.add(text_data, dataset_name)

    async def process_data(self, dataset_name: str = "main_dataset") -> None:
        """
        Process and structure data in the dataset and create a knowledge graph from it.

        This method takes the raw data that was previously added and transforms it into
        a structured knowledge graph with entities, relationships, and properties.

        Args:
            dataset_names (str): The name of the dataset to process into a knowledge graph.
                               Note: While cognee supports multiple datasets, this integration
                               currently processes 'main_dataset' only. Full dataset_names
                               support will be added in a future version. This parameter is
                               included to show the intended API design.

        """
        from cognee.modules.users.methods import get_default_user

        user = await get_default_user()
        await cognee_lib.cognify(dataset_name, user)

    async def rag_search(self, query: str) -> list:
        """
        Answer query using traditional RAG approach with document chunks.

        This method performs retrieval-augmented generation by finding the most
        relevant document chunks and generating a response based on them.

        Args:
            query (str): The question or query to answer.

        Returns:
            list: Search results containing relevant document chunks and generated responses.

        """
        user = await cognee_lib.modules.users.methods.get_default_user()
        return await cognee_lib.search(
            query_type=cognee_lib.SearchType.RAG_COMPLETION,
            query_text=query,
            user=user,
        )

    async def search(self, query: str) -> list:
        """
        Search the knowledge graph for relevant information using graph-based retrieval.

        This method leverages the graph structure to find related entities, relationships,
        and contextual information that traditional RAG might miss.

        Args:
            query (str): The question or search term to match against entities and relationships in the graph.

        Returns:
            list: Search results containing graph-based insights and related information.

        """
        user = await cognee_lib.modules.users.methods.get_default_user()
        return await cognee_lib.search(
            query_type=cognee_lib.SearchType.GRAPH_COMPLETION,
            query_text=query,
            user=user,
        )

    async def get_related_nodes(self, node_id: str) -> list:
        """
        Find nodes and relationships connected to a specific node in the knowledge graph.

        This method explores the graph structure to discover entities and concepts
        that are directly or indirectly related to the specified node.

        Args:
            node_id (str): The identifier or name of the node to find connections for.

        Returns:
            list: Related nodes, relationships, and insights connected to the specified node.

        """
        user = await cognee_lib.modules.users.methods.get_default_user()
        return await cognee_lib.search(
            query_type=cognee_lib.SearchType.INSIGHTS,
            query_text=node_id,
            user=user,
        )

    async def visualize_graph(
        self, open_browser: bool = False, output_file_path: str | None = None
    ) -> str:
        """
        Generate HTML visualization of the graph and optionally open in browser.

        Args:
            open_browser (bool): Whether to automatically open the visualization in the default browser. Defaults to False.
            output_file_path (str | None): Directory path where the HTML file will be saved.
                                         If None, saves to user's home directory. Defaults to None.

        Returns:
            str: Full path to the generated HTML visualization file.

        Raises:
            ValueError: If output_file_path is provided but is not a valid directory.

        """
        # Determine the full file path for the visualization
        if output_file_path:
            if not os.path.isdir(output_file_path):
                raise ValueError(
                    f"The provided path '{output_file_path}' is not a directory"
                )
            full_file_path = os.path.join(output_file_path, "graph_visualization.html")
        else:
            home_dir = os.path.expanduser("~")
            full_file_path = os.path.join(home_dir, "graph_visualization.html")

        # Generate the visualization using cognee
        await cognee_lib.visualize_graph(full_file_path)

        # Open in browser if requested
        if open_browser:
            webbrowser.open(f"file://{os.path.abspath(full_file_path)}")

        return full_file_path


if TYPE_CHECKING:
    _: GraphRAG = CogneeGraphRAG("dummy_key")
