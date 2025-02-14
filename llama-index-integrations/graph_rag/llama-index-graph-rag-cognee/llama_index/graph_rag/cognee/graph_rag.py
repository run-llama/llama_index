import os
import pathlib
from typing import List, Union

import cognee

from llama_index.core import Document

from .base import GraphRAG


class CogneeGraphRAG(GraphRAG):
    """Cognee GraphRAG, handles adding, storing, processing and retrieving information from knowledge graphs.

    Unlike traditional RAG models that retrieve unstructured text snippets, graphRAG utilizes knowledge graphs.
    A knowledge graph represents entities as nodes and their relationships as edges, often in a structured semantic format.
    This enables the system to retrieve more precise and structured information about an entity, its relationships, and its properties.

    Attributes:
    llm_api_key: str: API key for desired LLM.
    llm_provider: str: Provider for desired LLM (default: "openai").
    llm_model: str: Model for desired LLM (default: "gpt-4o-mini").
    graph_db_provider: str: The graph database provider (default: "networkx").
                            Supported providers: "neo4j", "networkx".
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
        graph_db_provider: str = "networkx",
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
        cognee.config.set_llm_config(
            {
                "llm_api_key": llm_api_key,
                "llm_provider": llm_provider,
                "llm_model": llm_model,
            }
        )

        cognee.config.set_vector_db_config(
            {
                "vector_db_url": vector_db_url,
                "vector_db_key": vector_db_key,
                "vector_db_provider": vector_db_provider,
            }
        )
        cognee.config.set_relational_db_config(
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

        cognee.config.set_graph_db_config(
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

        cognee.config.data_root_directory(data_directory_path)
        cognee_directory_path = str(
            pathlib.Path(
                os.path.join(pathlib.Path(__file__).parent, ".cognee_system/")
            ).resolve()
        )
        cognee.config.system_root_directory(cognee_directory_path)

    async def add(
        self, data: Union[Document, List[Document]], dataset_name: str
    ) -> None:
        """Add data to the specified dataset.
        This data will later be processed and made into a knowledge graph.

        Args:
             data (Any): The data to be added to the graph.
             dataset_name (str): Name of the dataset or node set where the data will be added.
        """
        # Convert LlamaIndex Document type to text
        if isinstance(data, List) and len(data) > 0:
            data = [data.text for data in data if type(data) == Document]
        elif type(data) == Document:
            data = [data.text]

        await cognee.add(data, dataset_name)

    async def process_data(self, dataset_names: str) -> None:
        """Process and structure data in the dataset and make a knowledge graph out of it.

        Args:
            dataset_name (str): The dataset name to process.
        """
        user = await cognee.modules.users.methods.get_default_user()
        datasets = await cognee.modules.data.methods.get_datasets_by_name(
            dataset_names, user.id
        )
        await cognee.cognify(datasets, user)

    async def get_graph_url(self, graphistry_password, graphistry_username) -> str:
        """Retrieve the URL or endpoint for visualizing or interacting with the graph.

        Returns:
            str: The URL endpoint of the graph.
        """
        if graphistry_password and graphistry_username:
            cognee.config.set_graphistry_config(
                {"username": graphistry_username, "password": graphistry_password}
            )

        from cognee.shared.utils import render_graph
        from cognee.infrastructure.databases.graph import get_graph_engine
        import graphistry

        graphistry.login(
            username=graphistry_username,
            password=graphistry_password,
        )
        graph_engine = await get_graph_engine()

        graph_url = await render_graph(graph_engine.graph)
        print(graph_url)
        return graph_url

    async def rag_search(self, query: str) -> list:
        """Answer query based on data chunk most relevant to query.

        Args:
            query (str): The query string.
        """
        user = await cognee.modules.users.methods.get_default_user()
        return await cognee.search(
            query_type=cognee.api.v1.search.SearchType.COMPLETION,
            query_text=query,
            user=user,
        )

    async def search(self, query: str) -> list:
        """Search the graph for relevant information based on a query.

        Args:
            query (str): The query string to match against data from the graph.
        """
        user = await cognee.modules.users.methods.get_default_user()
        return await cognee.search(
            query_type=cognee.api.v1.search.SearchType.GRAPH_COMPLETION,
            query_text=query,
            user=user,
        )

    async def get_related_nodes(self, node_id: str) -> list:
        """Search the graph for relevant nodes or relationships based on node id.

        Args:
            node_id (str): The name of the node to match against nodes in the graph.
        """
        user = await cognee.modules.users.methods.get_default_user()
        return await cognee.search(
            query_type=cognee.api.v1.search.SearchType.INSIGHTS,
            query_text=node_id,
            user=user,
        )
