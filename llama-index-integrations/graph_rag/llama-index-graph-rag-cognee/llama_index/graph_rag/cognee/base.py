from abc import abstractmethod
from typing import Protocol

class GraphRAG(Protocol):
    @abstractmethod
    async def add(self, data, dataset_name):
        """Add data to the specified dataset.
        This data will later be processed and made into a knowledge graph.

         Args:
             data (Any): The data to be added to the graph.
             dataset_name (str): Name of the dataset or node set where the data will be added.
        """
        pass

    @abstractmethod
    async def process_data(self, dataset_name: str):
        """Process and structure data in the dataset and make a knowledge graph out of it.

        Args:
            dataset_name (str): The dataset name to process.
        """
        pass

    @abstractmethod
    async def get_graph_url(self):
        """Retrieve the URL or endpoint for visualizing or interacting with the graph.

        Returns:
            str: The URL endpoint of the graph.
        """
        pass

    @abstractmethod
    async def search(self, query: str):
        """Search the graph for relevant information based on a query.

        Args:
            query (str): The query string to match against data from the graph.
        """
        pass

    @abstractmethod
    async def get_related_nodes(self, node_id: str):
        """Search the graph for relevant nodes or relationships based on node id.

        Args:
            node_id (str): The name of the node to match against nodes in the graph.
        """
        pass
