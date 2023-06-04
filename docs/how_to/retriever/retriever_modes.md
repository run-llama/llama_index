# Retriever Modes
Here we show the mapping from `retriever_mode` configuration to the selected retriever class.
> Note that `retriever_mode` can mean different thing for different index classes. 

## Vector Index
Specifying `retriever_mode` has no effect (silently ignored).
`vector_index.as_retriever(...)` always returns a VectorIndexRetriever.


## List Index
* `default`: ListIndexRetriever 
* `embedding`: ListIndexEmbeddingRetriever 
* `llm`: ListIndexLLMRetriever

## Tree Index
* `select_leaf`: TreeSelectLeafRetriever
* `select_leaf_embedding`: TreeSelectLeafEmbeddingRetriever
* `all_leaf`: TreeAllLeafRetriever
* `root`: TreeRootRetriever


## Keyword Table Index
* `default`: KeywordTableGPTRetriever
* `simple`: KeywordTableSimpleRetriever
* `rake`: KeywordTableRAKERetriever


## Knowledge Graph Index
* `keyword`: KGTableRetriever
* `embedding`: KGTableRetriever
* `hybrid`: KGTableRetriever

## Document Summary Index
* `default`: DocumentSummaryIndexRetriever
* `embedding`: DocumentSummaryIndexEmbeddingRetriever