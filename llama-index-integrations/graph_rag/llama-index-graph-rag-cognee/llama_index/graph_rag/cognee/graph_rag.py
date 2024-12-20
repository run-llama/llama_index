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
    llm_api_key: str: Api key for desired llm.
    llm_provider: str: Provider for desired llm.
    llm_model: str: Model for desired llm.
    graph_db_provider: str: The graph database provider.
    vector_db_provider: str: The vector database provider.
    relational_db_provider: str: The relational database provider.
    db_name: str: The name of the databases.
    """

    def __init__(
        self,
        llm_api_key: str,
        llm_provider: str = "openai",
        llm_model: str = "gpt-4o-mini",
        graph_db_provider: str = "networkx",
        vector_db_provider: str = "lancedb",
        relational_db_provider: str = "sqlite",
        db_name: str = "cognee_db",
    ) -> None:
        cognee.config.set_llm_config(
            {
                "llm_api_key": llm_api_key,
                "llm_provider": llm_provider,
                "llm_model": llm_model,
            }
        )

        cognee.config.set_vector_db_config({"vector_db_provider": vector_db_provider})
        cognee.config.set_relational_db_config(
            {"db_provider": relational_db_provider, "db_name": db_name}
        )
        cognee.config.set_graph_database_provider(graph_db_provider)

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

    async def search(self, query: str) -> list:
        """Search the graph for relevant information based on a query.

        Args:
            query (str): The query string to match against data from the graph.
        """
        user = await cognee.modules.users.methods.get_default_user()
        return await cognee.search(
            cognee.api.v1.search.SearchType.SUMMARIES, query, user
        )

    async def get_related_nodes(self, node_id: str) -> list:
        """Search the graph for relevant nodes or relationships based on node id.

        Args:
            node_id (str): The name of the node to match against nodes in the graph.
        """
        user = await cognee.modules.users.methods.get_default_user()
        return await cognee.search(
            cognee.api.v1.search.SearchType.INSIGHTS, node_id, user
        )
