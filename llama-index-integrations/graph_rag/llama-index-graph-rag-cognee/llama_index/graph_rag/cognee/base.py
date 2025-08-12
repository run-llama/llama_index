from abc import abstractmethod
from typing import Protocol, Union, List

from llama_index.core import Document


# NOTE: This is a bare-bone suggestion for an abstract protocol to define GraphRAG for llama-index
# This should be expanded upon and integrated to llama-index-core to support multiple different GraphRAG
# libraries in the future
class GraphRAG(Protocol):
    """
    Abstract graph RAG protocol.

    This protocol defines the interface for a graphRAG, which is responsible
    for adding, storing, processing and retrieving information from knowledge graphs.

    Attributes:
        llm_api_key: str: Api key for desired llm.
        graph_db_provider: str: The graph database provider.
        vector_db_provider: str: The vector database provider.
        relational_db_provider: str: The relational database provider.
        db_name: str: The name of the databases.

    """

    @abstractmethod
    async def add(self, data: Union[Document, List[Document]], dataset_name: str) -> None:
        """
        Add data to the specified dataset.
        This data will later be processed and made into a knowledge graph.

        Args:
             data (Any): The data to be added to the graph.
             dataset_name (str): Name of the dataset or node set where the data will be added.

        """

    @abstractmethod
    async def process_data(self, dataset_name: str) -> None:
        """
        Process and structure data in the dataset and make a knowledge graph out of it.

        Args:
            dataset_name (str): The dataset name to process.

        """

    @abstractmethod
    async def search(self, query: str) -> list:
        """
        Search the graph for relevant information based on a query.

        Args:
            query (str): The query string to match against data from the graph.

        """

    @abstractmethod
    async def get_related_nodes(self, node_id: str) -> list:
        """
        Search the graph for relevant nodes or relationships based on node id.

        Args:
            node_id (str): The name of the node to match against nodes in the graph.

        """
    
    @abstractmethod
    async def visualize_graph(self, open_browser: bool = False, output_file_path: str | None = None) -> str:
        """
        Visualize the graph.

        Args:
            open_browser (bool): Whether to open the graph in a browser.
            output_file_path (str): The path to save the graph to.

        """
