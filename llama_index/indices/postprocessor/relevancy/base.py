from llama_index.indices.postprocessor.types import BaseNodePostprocessor
from enum import Enum

"""
post_processor_config: Dict = {
    "node_splitter": { "splitter_type": "sentence" },
    "relevancy_scorer": { 
        "scorer_type": "sentence_transformer", 
        "model_name": "paraphrase-distilroberta-base-v1" 
        "agg_mode": "intra_node",
    },
    "filters": [
        { 
            "filter_type": "top_k", 
            "top_k": 5, 
            "agg_mode": "intra_node", 
        }, 
        { 
            "filter_type": "marginalization", 
            "agg_mode": "intra_node",
            "embedding_function": "default_embedding",
            "similarity_function": "cosine",
        }
    ],
    # node_synthesizers have access to the old nodes and the new nodes
    "synthesizer": {
        "score_mode": "new_avg",
        "node_synthesizers": [
            { 
                "synthesizer_type": "summarizer", 
                "op_mode": "sub_node",
                "summarizer_type": "hugging-face",
                "model_name": "t5-base", 
            },
            # Highlight the top 1 sentence, used in 
            # relevancy highlighting formatter
            { 
                "synthesizer_type": "relevancy_highlights",
                "op_mode": "intra_node", 
                "relevancy_mode": "top_k", 
                "top_k": 1
            },
            { 
                "synthesizer_type": "formatter", 
                "op_mode": "intra_node", 
                "format_mode": "join",
                "join_on": "...",
            },
        ]
    },
}
"""


class BaseRelevancyPostprocessor(BaseNodePostprocessor):
    """
    Base class for relevancy post-processors.

    It is meant to encompass the stages of:
    - Splitting / Chunking
    - Ranking Function / Relevancy Scoring
    - Filtering / Marginalization
    - Node Synthesis: Aggregation, Mapping
    """

    filter_operator_shuffle: OperatorMode
    scorer_operator_shuffle: OperatorMode = filter_operator_shuffle

    _initial_nodes: Optional[List[NodeWithScores]] = None
    _nodes: Optional[List[NodeWithScores]] = None
    _subnodes: Optional[List[List[NodeWithScores]]] = None

    filters = List[BaseFilter]
    scorer: BaseRelevancyScorer  # default is to use old score...?

    def __init__(self, config):
        self.config = config

    def validate(self):
        if self.splitter is None:
            assert self.scorer.agg_mode == AggMode.INTER_NODE
            for filter in self.filters:
                assert filter.agg_mode == AggMode.INTER_NODE
            for synthesizer in self.node_synthesizers:
                assert synthesizer.op_mode == MapMode.NODE
        else:
            assert self.scorer.agg_mode != AggMode.INTER_NODE
            for filter in self.filters:
                assert filter.agg_mode != AggMode.INTER_NODE
            for synthesizer in self.node_synthesizers:
                assert synthesizer.op_mode in [
                    MapMode.SUB_NODE,
                    MapMode.NODE,
                    AggMode.INTRA_NODE,
                ]

    def postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        self._initial_nodes = nodes

        if self.scorer.agg_mode == AggMode.INTER_NODE:
            assert self.splitter is None
            self._nodes = nodes

            # Relevant scores are inter-node
            self._nodes = self.relevancy_scorer(self._nodes)

            # Filter is inter-node
            assert self.filter.agg_mode == AggMode.INTER_NODE
            self._nodes = self.filter.filter(self._nodes)

            # Synthesizer can only be a map from each node to a new node
            assert self.synthesizer.op_mode == MapMode.NODE
            self._nodes = self.synthesizer.synthesize(self._nodes)
        else:
            assert self.splitter is not None
            self._subnodes = self.splitter.split(nodes)

            # Score each subnode
            if self.scorer.agg_mode == AggMode.INTRA_NODE:
                relevancy_scored = []
                for nodes in self._subnodes:
                    relevancy_scored.append(self.scorer.score(nodes))
                self._subnodes = relevancy_scored
            elif self.scorer.agg_mode == AggMode.CROSS_NODE:
                self._subnodes = self.scorer.score_nested(self._subnodes)
            else:
                raise ValueError("Invalid scorer agg mode.")

            # Filter the subnodes by considering the population of subnodes
            for filter in self.filters:
                if filter.agg_mode == AggMode.INTRA_NODE:
                    filtered = []
                    for nodes in self._subnodes:
                        filtered.append(filter.filter(nodes))
                    self._subnodes = filtered
                elif filter.agg_mode == AggMode.CROSS_NODE:
                    self._subnodes = filter.filter_nested(self._subnodes)
                else:
                    raise ValueError("Invalid filter agg mode.")

            # synthesize the subnodes into nodes
            # Modify the original nodes in place, utilizing
            # information from the subnodes.
            self._nodes = self._original_nodes
            for synthesizer in self.node_synthesizers:
                if synthesizer.op_mode == MapMode.SUB_NODE:
                    synthesized = []
                    for nodes in self._subnodes:
                        synthesized.append(synthesizer.synthesize(nodes))
                    self._subnodes = synthesized
                if self.synthesizer.op_mode == MapMode.NODE:
                    self._nodes = synthesizer.synthesize(self._nodes)
                elif synthesizer.op_mode == AggMode.INTRA_NODE:
                    self._subnodes, self._nodes = synthesizer.synthesize_nested(
                        self._subnodes, self._nodes
                    )
                else:
                    raise ValueError("Invalid synthesizer operator mode.")

        return self._nodes

    def _aprocess(self):
        pass

    def _aget_relevancy(self, query, results):
        pass


class OperatorMode(Enum):
    pass


class AggMode(OperatorMode, Enum):
    """
    Performs an agg operation on a node level,
    across all nodes. Splitter must be `None`.
    """

    INTER_NODE = "inter_node"

    """
    Performs an agg operation within a node, 
    on a sub-node level. Splitter must not be `None`.
    """
    INTRA_NODE = "intra_node"

    """
    Performs an agg operation on a sub-node level, 
    across all nodes. Splitter must not be `None`.
    """
    CROSS_NODE = "cross_node"


class MapMode(OperatorMode, Enum):
    """
    Performs a map operation on the node level.
    """

    NODE = "node"

    """
    Performs a map operation on the sub-node level.
    """
    SUB_NODE = "sub_node"


class Splitter:
    # Sentence: tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")
    # Text spltters/chunkers: `TokenTextSplitter(TextSplitter)`
    pass
