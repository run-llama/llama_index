# Bedrock Managed Knowledge Base Support

## Changes
- Updated `AmazonKnowledgeBasesRetriever` to default to `managedSearchConfiguration` instead of `vectorSearchConfiguration`
- Added `knowledge_base_type` parameter (`MANAGED` | `VECTOR`) to retriever constructor
- When type is `MANAGED`, retrieval calls use `managedSearchConfiguration` with managed embedding/reranking
- Added `use_agentic_retrieval` flag for `AgenticRetrieveStream` support
- Existing VECTOR retrieval paths remain unchanged for backward compatibility

## Design
- MANAGED is the default; VECTOR via explicit `knowledge_base_type="VECTOR"` toggle
- AgenticRetrieveStream used when `use_agentic_retrieval=True` for enhanced results
- Backward compatible: existing VECTOR paths and parameters unchanged
- Retriever auto-detects configuration shape based on KB type

## API Shapes
- KB Creation: `type: MANAGED` + `managedKnowledgeBaseConfiguration.embeddingModelType: MANAGED`
- Retrieval: `managedSearchConfiguration` (not `vectorSearchConfiguration`)
- Agentic: `AgenticRetrieveStream` with `foundationModelType: MANAGED`, `rerankingModelType: MANAGED`

## Configuration
| Variable | Description | Default |
|---|---|---|
| knowledge_base_type | MANAGED or VECTOR | MANAGED |
| use_agentic_retrieval | Enable agentic retrieval | True |
| num_results | Number of results to retrieve | 5 |

## SDK Requirements
- boto3 >= 1.43 for managed search and agentic retrieval
- llama-index-core >= 0.10.0
