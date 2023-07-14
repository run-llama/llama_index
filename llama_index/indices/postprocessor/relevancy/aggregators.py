class Aggregator:
    op_mode: OperatorMode

    def validate(self):
        assert self.op_mode in [MapMode.NODE, MapMode.SUB_NODE, AggMode.INTRA_NODE]

    def aggregate(self, nodes: List[NodeWithScores]) -> List[NodeWithScores]:
        pass

    def aggregate_nested(
        self,
        subnodes: List[List[NodeWithScores]],
        nodes: List[NodeWithScores],
    ) -> Tuple[List[List[NodeWithScores]], List[NodeWithScores]]:
        pass


class ScoreAggregator(Aggregator):
    """
    score modes: old, new_max, new_avg
    """

    score_mode: str = "old"

    def validate(self):
        assert self.score_mode in ["old", "new_max", "new_avg"]
        assert self.op_mode == AggMode.INTRA_NODE

    def aggregate_nested(
        self, subnodes: List[List[NodeWithScores]], nodes: List[NodeWithScores]
    ) -> Tuple[List[List[NodeWithScores]], List[NodeWithScores]]:
        if self.op_mode != AggMode.INTRA_NODE:
            raise ValueError("Invalid aggregator operator mode.")
        if self.score_mode == "old":
            # Do not modify the scores from the nodes
            pass
        if self.score_mode == "new_max":
            for node, ns in zip(nodes, subnodes):
                max_score = max([node.score for node in ns])
                node.score = max_score
        if self.score_mode == "new_avg":
            for node, ns in zip(nodes, subnodes):
                avg_score = sum([node.score for node in ns]) / len(ns)
                node.score = avg_score
        return subnodes, nodes


class FormatterAggregator(Aggregator):
    op_mode: AggMode


class SummarizerAggregator(Aggregator):
    op_mode: MapMode
