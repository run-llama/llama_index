
from llama_index.indices.postprocessor.types import BaseNodePostprocessor
from enum import Enum

"""
post_processor_config: Dict = {
    "node_splitter": { "splitter_type": "sentence" },
    "relevancy_scorer": { 
        "scorer_type": "sentence_transformer", 
        "model_name": "paraphrase-distilroberta-base-v1" 
        "operator_mode": "intra_node",
    },
    "filters": [
        { 
            "filter_type": "top_k", 
            "top_k": 5, 
            "operator_mode": "intra_node", 
        }, 
        { 
            "filter_type": "marginalization", 
            "operator_mode": "intra_node",
            "embedding_function": "default_embedding",
            "similarity_function": "cosine",
        }
    ],
    # Aggregators have access to the old nodes and the new nodes
    "aggregator": {
        "score_mode": "new_avg",
        "aggregators": [
            { 
                "aggregator_type": "summarizer", 
                "operator_mode": "sub_node",
                "summarizer_type": "hugging-face",
                "model_name": "t5-base", 
            },
            # Highlight the top 1 sentence, used in 
            # relevancy highlighting formatter
            { 
                "aggregator_type": "relevancy_highlights",
                "operator_mode": "intra_node", 
                "relevancy_mode": "top_k", 
                "top_k": 1
            },
            { 
                "aggregator_type": "formatter", 
                "operator_mode": "intra_node", 
                "format_mode": "join",
                "join_on": "...",
            },
        ]
    },
}
"""

class BaseRelevancyPostprocessor(BasePostprocessor):
    """
    Base class for relevancy post-processors.

    It is meant to encompass the stages of:
    - Splitting / Chunking
    - Ranking Function / Relevancy Scoring
    - Filtering / Marginalization
    - Aggregation and Node Representation
    """
    filter_operator_shuffle: NodeShuffleMode
    scorer_operator_shuffle: NodeShuffleMode = filter_operator_shuffle

    def __init__(self, config):
        self.config = config

    def validate(self):
        pass

    def process(self, query, results):
        if self.scorer_operator_shuffle == NodeShuffleMode.INTER_NODE:
            assert self.splitter is None
            # No split required
            relevancy_scores = self.relevancy_scorer()
            self.filter.filter(relevancy_scores)
        else:
            assert self.splitter is not None
            # Apply split
            nodes_list = self._splitter.split(nodes) 
            if self.scorer_operator_shuffle == NodeShuffleMode.INTRA_NODE:
                relevancy_scores = []
                for nodes in nodes_list:
                    relevancy_scores.append(self.relevancy_scorer(nodes))
            else:
                relevancy_scores = self.relevancy_scorer(flatten(nodes_list))



    def _aget_relevancy(self, query, results):
        pass


    def unknown(self):
        if self.

class NodeShuffleMode(Enum):
    """
    Performs an agg operation on a node level, across all nodes. Splitter must be `None`.
    """
    AGG_INTER_NODE = "inter_node"

    """
    Performs an agg operation within a node, on a subnode level. Splitter must not be `None`.
    """
    AGG_INTRA_NODE = "intra_node"

    """
    Performs an agg operation across subnodes of all nodes. Splitter must not be `None`.
    """
    AGG_CROSS_NODE = "cross_node"

    """
    Performs a map operation on the node level.
    """
    MAP_NODE = "node"

    """
    Performs a map operation on the sub-node level.
    """
    MAP_SUB_NODE = "sub_node"

class Splitter():
    # Sentence: tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")
    # Text spltters/chunkers: `TokenTextSplitter(TextSplitter)`

class BaseRelevancyScorer():
    metadata_visibility: MetadataMode
    cross_node_mode: NodeShuffleMode


class BaseFilter():
    def filter(self, nodes: List[NodeWithScores]):
        pass

class TopKFilter(BaseFilter):
    """
    Filters the top k nodes.
    """
    top_k: int = 1
    cross_node_mode: NodeShuffleMode = NodeShuffleMode.INTRA_NODE

    def filter(self, nodes: List[NodeWithScores]):
        pass

class PercentileFilter(BaseFilter):
    """
    Filters the specified percentile of nodes.
    """
    percentile: float = 0.5
    cross_node_mode: NodeShuffleMode = NodeShuffleMode.INTRA_NODE

    def filter(self, nodes: List[NodeWithScores]):
        pass

class MarginalizationFilter(BaseFilter):
    """
    Filters based on similarity function
    """
    similarity_function: Any
    cross_node_mode: NodeShuffleMode = NodeShuffleMode.INTRA_NODE

    def filter(self, nodes: List[NodeWithScores]):
        pass

class Aggregator():
    aggregation_parameters: AggregationParameters
    content_aggregator: ContentAggregator

class ContentAggregator():
    """
    Aggregates content from sub-nodes into a single node. 
    Every sub-node has exactly one parent. No sub-node
    can be re-parented to a new parent.
    """
    def

class FormatterAggregator(ContentAggregator):
    pass

class SummarizerAggregator(ContentAggregator):
    aggregator_operator_shuffle: NodeShuffleMode = NodeShuffleMode.INTRA_NODE

@dataclass
class AggregationParameters:
    """
    modes: old, new_max, new_avg
    """
    score_mode: str = "old" 

class SentenceTransformerRanker():
    pass