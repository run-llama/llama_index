from llama_index.indices.postprocessor.relevancy.base import AggMode


class BaseRelevancyScorer:
    metadata_visibility: MetadataMode
    agg_mode: AggMode


class SentenceTransformerRanker(BaseRelevancyScorer):
    pass
