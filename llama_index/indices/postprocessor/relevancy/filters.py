class BaseFilter:
    """
    Filters are in post processors are generally aggregate filters:
    they filter based on the node population.
    """

    def filter(self, nodes: List[NodeWithScores]):
        pass


class TopKFilter(BaseFilter):
    """
    Filters the top k nodes.
    """

    top_k: int = 1
    cross_node_mode: OperatorMode = OperatorMode.INTRA_NODE

    def filter(self, nodes: List[NodeWithScores]):
        pass


class PercentileFilter(BaseFilter):
    """
    Filters the specified percentile of nodes.
    """

    percentile: float = 0.5
    cross_node_mode: OperatorMode = OperatorMode.INTRA_NODE

    def filter(self, nodes: List[NodeWithScores]):
        pass


class MarginalizationFilter(BaseFilter):
    """
    Filters based on similarity function
    """

    similarity_function: Any
    cross_node_mode: OperatorMode = OperatorMode.INTRA_NODE

    def filter(self, nodes: List[NodeWithScores]):
        pass
